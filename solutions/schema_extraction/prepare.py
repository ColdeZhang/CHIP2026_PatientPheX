"""Input-only passage maps. Original offsets are not dense document indices."""
import argparse
import hashlib
import json
from pathlib import Path

INPUT_SHA = '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(doc):
    if 'entities' in doc or 'association' in doc:
        raise ValueError('Only unlabeled input is allowed')
    passages, pieces, cursor = [], [], 0
    for i, p in enumerate(doc['full_text']):
        text = p['text']
        passages.append(dict(index=i, start=cursor, end=cursor+len(text), offset=int(p['offset'])))
        pieces.append(text)
        cursor += len(text)+2
    body = '\n\n'.join(pieces)
    patients = []
    for p in doc['patient']:
        mentions=[]
        for m in p['mention']:
            assert len(m['text']) == m['length']
            mapped=[]
            for segment in passages:
                start = int(m['offset']) - segment['offset'] + segment['start']
                end = start + int(m['length'])
                if segment['start'] <= start < end <= segment['end'] and body[start:end] == m['text']:
                    mapped.append([start,end])
            if not mapped:
                raise ValueError('Patient mention has no exact original source mapping')
            mentions.append(dict(offset=int(m['offset']), length=int(m['length']), text=m['text'], spans=mapped))
        if not mentions:
            raise ValueError('Given patient must have a mention')
        patients.append(dict(patient_id=str(p['patient_id']),mentions=mentions))
    assert len({p['patient_id'] for p in patients})==len(patients)
    return dict(pmc_id=str(doc['pmc_id']), pmid=doc['pmid'], text=body, passages=passages, patients=patients)

def original_span(doc,start,end):
    if not 0 <= start < end <= len(doc['text']):
        return None,'invalid_span'
    matches=[p for p in doc['passages'] if p['start'] <= start < end <= p['end']]
    if len(matches)!=1:
        return None,'cross_passage_span'
    p=matches[0]
    return dict(offset=p['offset']+start-p['start'], length=end-start,
                text=doc['text'][start:end], passage_index=p['index']),None

def patient_for_span(doc,start,end):
    # Coordinate containment only. Never match names globally or guess pronouns.
    physical,reason=original_span(doc,start,end)
    if reason:return None,reason
    a,b=physical['offset'],physical['offset']+physical['length']
    ids={p['patient_id'] for p in doc['patients'] for m in p['mentions']
         if (a <= m['offset'] and m['offset']+m['length'] <= b) or
            (m['offset'] <= a and b <= m['offset']+m['length'])}
    if len(ids)==1:return next(iter(ids)),None
    return None,'ambiguous_given_patient' if ids else 'no_given_patient_containment'

def contract():
    # Identical lexical mentions at different coordinates must stay distinct;
    # duplicated source passages must preserve their same physical patient.
    doc=dict(pmc_id='synthetic',pmid='0',full_text=[dict(offset=100,text='Ann pain. Ann fever.'),dict(offset=100,text='Ann pain. Ann fever.')],patient=[dict(patient_id='P1',mention=[dict(offset=100,length=3,text='Ann')]),dict(patient_id='P2',mention=[dict(offset=110,length=3,text='Ann')])])
    p=build(doc)
    assert patient_for_span(p,0,3)==('P1',None)
    assert patient_for_span(p,10,13)==('P2',None)
    s=p['passages'][1]['start'];assert patient_for_span(p,s,s+3)==('P1',None)
    assert patient_for_span(p,0,13)[1]=='ambiguous_given_patient'
    assert original_span(p,17,22)[1]=='cross_passage_span'
    assert original_span(p,s+4,s+8)[0]['offset']==104
    return dict(checks=6,status='PASS',held_labels_accessed=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    assert sha(args.input)==INPUT_SHA
    docs=[build(json.loads(line)) for line in Path(args.input).read_text().splitlines() if line]
    assert len(docs)==16 and len({d['pmc_id'] for d in docs})==16
    assert sum(len(d['patients']) for d in docs)==39
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'held.jsonl').open('w') as f:
        for d in docs:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    report=dict(contract=contract(),documents=16,patients=39,passages=sum(len(d['passages']) for d in docs),
                characters=sum(len(d['text']) for d in docs),patient_mentions=sum(len(p['mentions']) for d in docs for p in d['patients']),
                duplicate_source_passages=sum(len(d['passages'])-len({(p['offset'],d['text'][p['start']:p['end']]) for p in d['passages']}) for d in docs),
                input=dict(path=args.input,sha256=sha(args.input)),output=dict(path=str(out/'held.jsonl'),sha256=sha(out/'held.jsonl')))
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
