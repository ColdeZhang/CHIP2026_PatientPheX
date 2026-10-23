"""CPU parser adversaries: never accept thought JSON as a final answer."""
import unittest
from evaluate_fit import strict_final
class Contract(unittest.TestCase):
 def check(self,text,valid,hit=False):
  obj,_=strict_final(text,{'C000','C001'},{'P1','P2'},hit);self.assertEqual(obj is not None,valid)
 def test_closed_final(self):self.check('thought {"C000":[],"C001":[]} </think> {"C000":["P1","P2"],"C001":[]}',True)
 def test_no_close(self):self.check('thought {"C000":["P1"],"C001":[]}',False)
 def test_open_no_close(self):self.check('<think> {"C000":["P1"],"C001":[]}',False)
 def test_cap_even_with_close(self):self.check('</think>{"C000":[],"C001":[]}',False,True)
 def test_duplicate_key(self):self.check('</think>{"C000":[],"C000":[],"C001":[]}',False)
 def test_missing_key(self):self.check('</think>{"C000":[]}',False)
 def test_extra_key(self):self.check('</think>{"C000":[],"C001":[],"C002":[]}',False)
 def test_unknown_patient(self):self.check('</think>{"C000":["P3"],"C001":[]}',False)
 def test_duplicate_patient(self):self.check('</think>{"C000":["P1","P1"],"C001":[]}',False)
 def test_nonstring_patient(self):self.check('</think>{"C000":[1],"C001":[]}',False)
 def test_trailing_text(self):self.check('</think>{"C000":[],"C001":[]} done',False)
 def test_fenced(self):self.check('</think>```json\n{"C000":[],"C001":[]}\n```',False)
 def test_second_close(self):self.check('</think>thought </think>{"C000":[],"C001":[]}',False)
 def test_empty_sets(self):self.check('</think>  {"C000":[],"C001":[]}\n',True)
if __name__=='__main__':unittest.main()
