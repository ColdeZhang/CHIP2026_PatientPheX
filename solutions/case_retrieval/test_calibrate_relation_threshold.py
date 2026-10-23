import unittest
from calibrate_relation_threshold import choose_threshold, predictions

class CalibrationContract(unittest.TestCase):
    def test_equal_score_prefers_half_then_larger_equidistant(self):
        def row(step,score=0.7):return {'step':step,'metrics':{'Score':score}}
        self.assertEqual(choose_threshold([row(9),row(11)])['step'],11)
        self.assertEqual(choose_threshold([row(9),row(10),row(11)])['step'],10)
        self.assertEqual(choose_threshold([row(10),row(18,.71)])['step'],18)

    def test_probability_boundary_and_preservation(self):
        inputs=[{'pmc_id':'1','patient':[{'patient_id':'p'},{'patient_id':'empty'}]}]
        entities={'1':{'pmc_id':'1','entities':[{'identifier':'HP:0000001','offset':0,'length':1,'text':'a'}],
                       'association':[{'patient_id':'p','phenotype':['old']},{'patient_id':'empty','phenotype':[]}]}}
        rows=[{'pmc_id':'1','patient_id':'p','concept':'HP:0000001'}]
        pred=predictions(inputs,entities,rows,[.15],.15)
        self.assertEqual(pred[0]['association'],[{'patient_id':'p','phenotype':['HP:0000001']},{'patient_id':'empty','phenotype':[]}])
        self.assertEqual(pred[0]['entities'],entities['1']['entities'])
        self.assertEqual(entities['1']['association'][0]['phenotype'],['old'])
        self.assertEqual(predictions(inputs,entities,rows,[.149],.15)[0]['association'][0]['phenotype'],[])

    def test_unknown_patient_not_silently_dropped(self):
        with self.assertRaises(KeyError):
            predictions([{'pmc_id':'1','patient':[{'patient_id':'p'}]}],{'1':{'entities':[]}},
                        [{'pmc_id':'1','patient_id':'unknown','concept':'HP:0000001'}],[.8],.5)

if __name__=='__main__':unittest.main()
