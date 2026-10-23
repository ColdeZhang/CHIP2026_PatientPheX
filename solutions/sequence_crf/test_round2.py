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
