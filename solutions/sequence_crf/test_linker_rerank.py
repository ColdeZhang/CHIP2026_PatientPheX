"""Requires the private GPU environment's Python; tests themselves use CPU only."""
import unittest
from collections import Counter,defaultdict
import run_linker_rerank as rr

class Linker:
    allowed={'HP:0000252','HP:0001250'}
    terms={'HP:0000252':{'names':['Microcephaly']},'HP:0001250':{'names':['Seizure']}}
    def canonical(self,x):return x if x in self.allowed else None

class RetrievalIsolationTests(unittest.TestCase):
    def setUp(self):
        self.linker=Linker()
        self.doc={'pmc_id':'A','full_text':[{'offset':0,'text':'A tiny head was observed.','section_type':'CASE'}]}
        self.e={'identifier':'HP:0000252','offset':2,'length':9,'text':'tiny head'}
        self.doc['entities']=[self.e]
    def test_own_document_alias_is_not_retrieval_evidence(self):
        names,counts,sources=rr.candidate_bank([self.doc],self.linker)
        retrieved={'tiny head':[(names.index('tiny head'),1.)]}
        rows,_,stats=rr.rows_for(self.doc,[self.e],self.linker,retrieved,names,counts,sources,8,True)
        self.assertEqual(stats['gold_retrieved'],0)
        self.assertEqual(rows[0]['similarity'],0.)  # gold is forced only as a training target
    def test_other_document_alias_remains_available(self):
        other={'pmc_id':'B','entities':[self.e]}
        names,counts,sources=rr.candidate_bank([self.doc,other],self.linker)
        retrieved={'tiny head':[(names.index('tiny head'),1.)]}
        rows,_,stats=rr.rows_for(self.doc,[self.e],self.linker,retrieved,names,counts,sources,8,True)
        self.assertEqual(stats['gold_retrieved'],1)
        self.assertEqual(rows[0]['similarity'],1.)
    def test_input_candidate_preserved_when_outside_topk(self):
        names,counts,sources=rr.candidate_bank([],self.linker)
        retrieved={'tiny head':[(names.index('seizure'),.9)]}
        rows,_,_=rr.rows_for(self.doc,[self.e],self.linker,retrieved,names,counts,sources,1,False)
        self.assertEqual({r['identifier'] for r in rows},{'HP:0000252','HP:0001250'})
        self.assertTrue(all('label' not in r for r in rows))
if __name__=='__main__':unittest.main()
