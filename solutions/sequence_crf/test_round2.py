import pickle
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import run
import run_round2
import run_relation_features
import source_fingerprint
import apply_linker_scores
import merge_pair_rag
import patient_id_anchors
import json
import contextlib
import io
from unittest.mock import patch

class Round2Tests(unittest.TestCase):
    def test_compound_training_targets_keep_natural_text(self):
        doc={'patient':[{'patient_id':'P1'}], 'association':[{'patient_id':'P1','phenotype':['HP:0000252;HP:0001250','small; weak','HP:0000252;bad']} ]}
        self.assertEqual(run.gold_associations(doc)['P1'],{'HP:0000252','HP:0001250','small; weak','HP:0000252;bad'})
    def test_cache_source_or_config_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.pkl'
            p.write_bytes(pickle.dumps({'train_sha256':'t','hpo_sha256':'h','fingerprint':{'iterations':80}}))
            with self.assertRaisesRegex(ValueError,'Cached pipeline'):
                run.load_pipeline(p,'t','h',{'iterations':81})
    def test_expanded_names_avoid_generic_patient_and_input_labels(self):
        doc={'pmc_id':'x','patient':[{'patient_id':'P1','mention':[{'text':'patient','offset':0,'length':7},{'text':'II-1','offset':8,'length':4}]},
             {'patient_id':'P2','mention':[{'text':'patient','offset':13,'length':7}]}],
             'full_text':[{'offset':0,'text':'patient II-1 patient II-1'}],'entities':[{'identifier':'SECRET'}],'association':[{'phenotype':['SECRET']}]}
        out=run_round2.expanded_doc(doc)
        self.assertNotIn('entities',out);self.assertNotIn('association',out)
        self.assertEqual([m['offset'] for m in out['patient'][0]['mention']],[0,8,21])
        self.assertEqual(len(out['patient'][1]['mention']),1)
    def test_clinical_ids_use_complete_generation_and_require_numeric_context(self):
        doc={'pmc_id':'x','patient':[{'patient_id':'OIII-3','mention':[]},{'patient_id':'O1','mention':[]},{'patient_id':'P2','mention':[]}],'full_text':[{'offset':100,'text':'III‐3 and VIII-3, III-30. At age 1, Patient 1 and patient 2.'}],'entities':[{'identifier':'SECRET'}]}
        expanded,added=patient_id_anchors.expand(doc)
        self.assertEqual([(x['patient_id'],x['text']) for x in added],[('OIII-3','III‐3'),('O1','Patient 1')])
        self.assertNotIn('entities',expanded)
        self.assertEqual(doc['patient'][0]['mention'],[])
    def test_ambiguous_clinical_id_spellings_are_not_expanded(self):
        doc={'pmc_id':'x','patient':[{'patient_id':'OIII-3','mention':[]},{'patient_id':'OIII:3','mention':[]}],'full_text':[{'offset':0,'text':'III-3'}]}
        self.assertEqual(patient_id_anchors.expand(doc)[1],[])
    def test_unfinished_thinking_cannot_be_parsed_as_a_decision(self):
        self.assertIsNone(merge_pair_rag.parse_json('<think>Maybe use {"C000":["P1"]}'))
        self.assertEqual(merge_pair_rag.parse_json('<think>draft {"C000":["P1"]}</think>{"C000":[]}'),{'C000':[]})
    def test_relation_candidate_pool_does_not_replace_final_entities(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            objects={
                'input':[{'pmc_id':'x','patient':[{'patient_id':'P1'}]}],
                'pool':[{'pmc_id':'x','entities':[{'identifier':'HP:0000252','text':'tiny head','offset':0,'length':9}],'association':[{'patient_id':'P1','phenotype':[]}]}],
                'final':[{'pmc_id':'x','entities':[]}],
                'prompts':[{'row_id':'x:1','pmc_id':'x','patient_ids':['P1'],'code_map':{'C000':'HP:0000252'}}],
                'responses':[{'row_id':'x:1','response':'{"C000":["P1"]}'}]}
            for name,rows in objects.items():(d/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            argv=['merge_pair_rag','--input',str(d/'input'),'--entities',str(d/'pool'),'--final-entities',str(d/'final'),'--prompts',str(d/'prompts'),'--responses',str(d/'responses'),'--output',str(d/'output')]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):merge_pair_rag.main()
            result=json.loads((d/'output').read_text())
            self.assertEqual(result['entities'],[])
            self.assertEqual(result['association'][0]['phenotype'],['HP:0000252'])
    def test_crossencoder_without_nil_class_cannot_replace_nil(self):
        group={"original_identifier":"-1","protected_exact":False,"candidates":[{"identifier":"HP:0000252","probability":.999}]}
        self.assertEqual(apply_linker_scores.select(group),"-1")
        group["original_identifier"]="HP:0001250"
        self.assertEqual(apply_linker_scores.select(group),"HP:0000252")
        group["protected_exact"]=True
        self.assertEqual(apply_linker_scores.select(group),"HP:0001250")
    def test_source_hashes_preserve_same_named_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"crf").mkdir();(p/"semantic").mkdir()
            a=p/"crf/run.py";b=p/"semantic/run.py"
            a.write_text("CRF");b.write_text("SEMANTIC")
            hashes=source_fingerprint.source_hashes({"crf_base":a,"semantic_encoder":b})
            self.assertEqual(set(hashes),{"crf_base","semantic_encoder"})
            self.assertNotEqual(hashes["crf_base"],hashes["semantic_encoder"])
    def test_cohort_scope_requires_matching_explicit_count(self):
        f=run_relation_features.cohort_evidence
        self.assertTrue(f("Both patients had seizures",2,"CASE"))
        self.assertFalse(f("Both patients had seizures",3,"CASE"))
        self.assertTrue(f("All three siblings had seizures",3,"RESULTS"))
        self.assertFalse(f("All three siblings had seizures",4,"RESULTS"))
        self.assertFalse(f("These patients often present with seizures",2,"INTRO"))
        self.assertTrue(f("Our patients had seizures",2,"DISCUSS"))
    def test_concept_grouping_combines_repeated_mentions(self):
        entities=[{'identifier':'HP:0000252','text':'small head'},{'identifier':'HP:0000252;HP:0001250','text':'combined'},{'identifier':'-1','text':'unmapped; text'}]
        groups=run_round2.concept_groups(entities)
        self.assertEqual(len(groups['HP:0000252']),2)
        self.assertEqual(len(groups['HP:0001250']),1)
        self.assertIn('unmapped; text',groups)
if __name__=='__main__':unittest.main()
