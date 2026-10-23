import unittest
from prepare_extraction import windows,gold_quotes
from normalize_extraction import occurrences,parse_json
from prepare_pair_rag import patient_anchors,evidence

class ExtractionTests(unittest.TestCase):
    def test_packed_passages_preserve_original_offsets(self):
        doc={'pmc_id':'x','full_text':[{'offset':10,'text':'A seizure occurred.'},{'offset':100,'text':'Small ears were present.'}],
             'entities':[{'offset':12,'length':7,'text':'seizure','note':None},{'offset':100,'length':10,'text':'Small ears','note':None}]}
        chunks=list(windows(doc,100,0));self.assertEqual(len(chunks),1)
        self.assertEqual(gold_quotes(doc,chunks[0]),['seizure','Small ears'])
        prompt=dict(chunks[0],row_id='x:0');found,stats=occurrences([prompt],[{'row_id':'x:0','response':'["seizure", "Small ears", "invented"]'}])
        self.assertEqual([(x['offset'],x['text']) for x in found['x']],[(12,'seizure'),(100,'Small ears')]);self.assertEqual(stats['unmatched_quote'],1)
    def test_negated_duplicate_not_assigned_a_positive_offset(self):
        text='Seizures developed. No seizures were seen later.'
        prompt={'row_id':'x','pmc_id':'x','segments':[{'offset':40,'text':text}]}
        found,stats=occurrences([prompt],[{'row_id':'x','response':'["Seizures", "seizures"]'}])
        self.assertEqual([x['offset'] for x in found['x']],[40]);self.assertEqual(stats['negated_occurrence'],1)
    def test_invalid_or_fenced_json_is_explicit(self):
        self.assertEqual(parse_json('```json\n["seizures"]\n```'),['seizures']);self.assertIsNone(parse_json('cannot answer'))
    def test_patient_markers_preserve_identity_in_shared_context(self):
        doc={'patient':[{'patient_id':'P1','mention':[{'offset':0,'length':5,'text':'Case1'}]},{'patient_id':'P2','mention':[{'offset':18,'length':5,'text':'Case2'}]}],
             'full_text':[{'offset':0,'text':'Case1 has ptosis. Case2 is normal.'}]}
        anchors=patient_anchors(doc);self.assertIn('[PATIENT P1: Case1]',anchors)
        rows=evidence(doc,[{'offset':10,'length':6,'text':'ptosis'}],500);self.assertIn('[PATIENT P1:',rows[0]['context'])
if __name__=='__main__':unittest.main()
