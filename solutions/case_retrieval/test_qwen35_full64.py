"""Small CPU controls for the new fixed265 protocol; no model/GPU/gold access."""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import train_qwen35_full64 as m

class Full64Controls(unittest.TestCase):
    def rows(self):
        return [{'row_id':f'{i}:r0','pmc_id':str(i),'messages':[{'role':'user','content':'input only'}],
                 'answer':'{"C000": []}'} for i in range(64)]
    def files(self, root, data=None, excluded=None):
        train=Path(root)/'train.jsonl';held=Path(root)/'input.jsonl'
        train.write_text(''.join(json.dumps(x)+'\n' for x in (self.rows() if data is None else data)))
        held.write_text(''.join(json.dumps(x)+'\n' for x in ([{'pmc_id':str(100+i)} for i in range(16)] if excluded is None else excluded)))
        return train,held
    def test_disjoint64_passes_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t,h=self.files(d);_,ids,held=m.training_rows(t,h);self.assertEqual((len(ids),len(held)),(64,16))
            t,h=self.files(d,excluded=[{'pmc_id':'0'}]);self.assertRaises(ValueError,m.training_rows,t,h)
    def test_old48_and_duplicate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t,h=self.files(d,data=self.rows()[:48]);self.assertRaises(ValueError,m.training_rows,t,h)
            t,h=self.files(d,data=self.rows()+self.rows()[:1]);self.assertRaises(ValueError,m.training_rows,t,h)
    def test_excluded_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t,h=self.files(d,excluded=[{'pmc_id':'100','entities':[]}]);self.assertRaises(ValueError,m.training_rows,t,h)
    def test_exact_data_hash_guard_rejects_nonfrozen_rows(self):
        with tempfile.TemporaryDirectory() as d:
            t,h=self.files(d)
            args=argparse.Namespace(train=t,excluded_input=h)
            with self.assertRaisesRegex(ValueError,'exact full64'):
                m.verify_sources_and_data(args)
    def test_accumulation_visits_each_row_once_and_keeps_short_tail(self):
        for epoch in (0,1):
            groups=m.batch_plan(265,1,8,20261023,epoch)
            seen=[i for group in groups for batch in group for i in batch]
            self.assertEqual(len(groups),34);self.assertEqual(len(groups[-1]),1)
            self.assertEqual(sorted(seen),list(range(265)))
        self.assertNotEqual(m.batch_plan(265,1,8,20261023,0),m.batch_plan(265,1,8,20261023,1))
    def test_padding_is_actual_batch_width_not6400(self):
        class CPUShape:
            long='int64'
            @staticmethod
            def tensor(data,**kwargs):return data
        result=m.collate([{'input_ids':[1,2],'labels':[-100,2]},
                          {'input_ids':[1,2,3],'labels':[-100,2,3]}],0,CPUShape)
        self.assertEqual(result['input_ids'],[[1,2,0],[1,2,3]])
        self.assertEqual(result['labels'][0],[-100,2,-100]);self.assertEqual(result['attention_mask'][0],[1,1,0])
    def test_resume_fingerprint_and_epoch34_guard(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);checkpoint=out/'epoch-001';checkpoint.mkdir()
            metadata={'fingerprint':{'schema':'full64'},'completed_epochs':1,'global_step':26,'smoke_only':False}
            (checkpoint/'checkpoint.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'formal265'):
                m.validate_checkpoint(out,checkpoint,{'schema':'full64'})
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                m.validate_checkpoint(out,checkpoint,{'schema':'old48'})
    def test_controls_do_not_change_semantic_fingerprint(self):
        a=argparse.Namespace(train='same',resume=None,stop_after_epoch=None)
        b=argparse.Namespace(train='same',resume='epoch-001',stop_after_epoch=1)
        self.assertEqual(m.semantic_configuration(a),m.semantic_configuration(b))
        self.assertEqual(m.CONFIG['max_seq_length'],6400)
        self.assertEqual(m.epoch_outcome(1,2,1),'paused');self.assertEqual(m.epoch_outcome(2,2,None),'complete')
        self.assertRaises(ValueError,m.validate_stop_epoch,3,2)
    def test_core_helpers_preserved_from_reviewed204(self):
        old=ast.parse(Path(m.__file__).with_name('train_qwen35_lora.py').read_text());new=ast.parse(Path(m.__file__).read_text())
        oldfn={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
        newfn={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
        for name in ['batch_plan','collate','verify_fingerprint','validate_stop_epoch','epoch_outcome','model_files']:
            self.assertEqual(oldfn[name],newfn[name],name)

if __name__=='__main__':unittest.main()
