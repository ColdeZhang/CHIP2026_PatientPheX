#!/usr/bin/env python3
"""Input-only patient references with a unique verbatim context for grounding."""
import argparse
import json
from pathlib import Path
from run import read_jsonl, sha
from prepare_coreference import windows

SYSTEM = '''Find references to the supplied target patients in TEXT. Each target is identified by its original mention text and surrounding context. The words surrounding an original mention may describe other people; only the explicitly marked original mention identifies that target.
Resolve explicit patient/case/pedigree identifiers, person noun phrases, and pronouns only when identity is unambiguous. Return only person references, never phenotype names, body parts, genes or measurements. Preserve distinct family members and original target IDs. Do not assign an author's, control's, relative's or generic group's reference to a different target. A supplied target can itself be a relative or a specifically identified historical case; keep that identity, and never merge it with the current report's patients. Do not infer identity from having similar symptoms. Omit ambiguous references. Include group references only when the exact group of supplied targets is explicit.
Return a JSON array of objects with keys patient_id, text, context. patient_id must be supplied. text must be the exact complete person reference as written in TEXT, not a fragment inside another word. context must be a verbatim contiguous quote from TEXT, 20 to 300 characters, containing that exact reference once. Choose enough surrounding words to make this quote unique inside TEXT and establish the referent. Do not copy text from the identity anchors unless it also occurs in TEXT. Do not insert markers, change case or invent text. At most 25 items per patient. Return [] if none is unambiguous. No offsets, occurrence numbers, explanations or markdown.'''


def identities(doc):
    result = []
    for patient in doc['patient']:
        snippets = []
        for mention in patient['mention'][:3]:
            start, end = int(mention['offset']), int(mention['offset']) + int(mention['length'])
            for passage in doc['full_text']:
                offset = int(passage['offset'])
                if offset <= start < end <= offset+len(passage['text']):
                    local = start-offset
                    assert passage['text'][local:end-offset] == mention['text']
                    left, right = max(0, local-180), min(len(passage['text']), end-offset+220)
                    snippets.append({'original_mention': mention['text'], 'global_offset': start,
                                     'context': passage['text'][left:right]})
                    break
        result.append({'patient_id': str(patient['patient_id']), 'original_mentions': snippets})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    docs = read_jsonl(args.input)
    assert all('entities' not in d and 'association' not in d for d in docs)
    rows = []
    for doc in docs:
        anchors = json.dumps(identities(doc), ensure_ascii=False)
        for chunk in windows(doc, 5000, 300):
            rows.append(dict(chunk, row_id=f"{doc['pmc_id']}:coreference-context:{chunk['chunk_index']}",
                             patient_ids=[str(x['patient_id']) for x in doc['patient']],
                             messages=[{'role': 'system', 'content': SYSTEM},
                                       {'role': 'user', 'content': 'TARGET IDENTITIES:\n'+anchors+'\n\nTEXT:\n'+chunk['text']}]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps({
        'configuration': vars(args), 'documents': len(docs), 'chunks': len(rows),
        'source_sha256': sha(__file__), 'window_source_sha256': sha(Path(__file__).with_name('prepare_coreference.py')),
        'input_sha256': sha(args.input), 'output_sha256': sha(out)}, indent=2)+'\n')
    print(json.dumps({'documents': len(docs), 'chunks': len(rows), 'bytes': out.stat().st_size}))


if __name__ == '__main__':
    main()
