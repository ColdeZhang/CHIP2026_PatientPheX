import unittest
from collections import defaultdict
import enhance
import prepare_context
import merge_context


class EvidenceBoundaryTests(unittest.TestCase):
    def test_compound_relation_only_splits_complete_hpo_codes(self):
        doc={'association':[{'patient_id':'P1','phenotype':['HP:0000001;HP:0000002','pain; weakness','HP:0000001;bad']} ]}
        self.assertEqual(enhance.relation_gold(doc)['P1'],{'HP:0000001','HP:0000002','pain; weakness','HP:0000001;bad'})

    def test_patient_markers_preserve_shared_identity_and_original_text(self):
        doc={'patient':[{'patient_id':'P1','mention':[{'offset':105,'length':4}]},{'patient_id':'P2','mention':[{'offset':105,'length':4}]}]}
        self.assertEqual(prepare_context.mark('Both boys were short.',100,doc),'Both [PATIENT P1/P2: boys] were short.')

    def test_abbreviation_uses_local_long_form_and_rejects_other_meaning(self):
        lex=enhance.OntologyLexicon.__new__(enhance.OntologyLexicon);lex.mode='basic';lex.options={('intellectual','disability'):{('HP:0001249',):{'official':True,'documents':{}}}}
        known,other=lex.local_abbreviations('Intellectual disability (ID) was present. DNA sequencing (DS) was performed.')
        self.assertEqual(known['ID'][0],('HP:0001249',));self.assertIn('DS',other)

    def test_candidate_alias_support_excludes_current_document(self):
        lex=enhance.OntologyLexicon.__new__(enhance.OntologyLexicon)
        lex.options={('rare',):{('HP:0000001',):{'official':False,'documents':{'pmc1':1}}}}
        self.assertIsNone(lex.choose(('rare',),exclude='pmc1'))
        self.assertEqual(lex.choose(('rare',),exclude='pmc2')[0],('HP:0000001',))

    def test_response_parser_rejects_prose_or_truncated_json(self):
        with self.assertRaises(ValueError):merge_context.parse('{"P1": ["C001"]')
        with self.assertRaises(ValueError):merge_context.parse('I believe {"P1":[]}')
        self.assertEqual(merge_context.parse('```json\n{"P1":[]}\n```'),{'P1':[]})

if __name__=='__main__':unittest.main()
