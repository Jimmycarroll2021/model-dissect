import unittest
from minimind_re.response_probe import parse_answer, position_cost, screen_summary, qualify_summary, generate

class ParserTest(unittest.TestCase):
 def test_simple_and_explicit_answers(self):
  for text in ['5',' 5。 ','答案是5。','答案：5','结果是5。','甲写下的数字是5。']:
   self.assertTrue(parse_answer(text,'COPY','5','eos',[])['correct'],text)
  for text in ['大','答案是大。','大于']:
   self.assertTrue(parse_answer(text,'COMPARE','大','eos',[])['correct'],text)
 def test_adversaries(self):
  for text in ['15','-5','+5','5.0','5或6','5，5','不是5','答案不是5，是6','甲写下数字5，乙写下数字6','2减1等于1','5\n6','答案是5，但不确定','５','五','<think>5</think>5']:
   self.assertFalse(parse_answer(text,'COPY','5','eos',[])['correct'],text)
  for text in ['不大','不是小而是大','大或小','小于，但答案是大','较大','大大']:
   self.assertFalse(parse_answer(text,'COMPARE','大','eos',[])['correct'],text)
 def test_stop_and_controls(self):
  self.assertFalse(parse_answer('5','COPY','5','length',[])['correct'])
  self.assertFalse(parse_answer('5','COPY','5','eos',[1])['correct'])
  self.assertEqual(parse_answer('5','COPY','5','eos',[])['strict'],True)
  self.assertEqual(parse_answer('答案是5。','COPY','5','eos',[])['strict'],False)
 def test_cost(self):
  self.assertEqual(position_cost(128,24),3348)
  self.assertEqual(position_cost(48,24),1428)
  self.assertEqual(position_cost(3,1),3)
 def test_gates(self):
  def rows(n):return [{'task':task,'parse':{'correct':i<n,'strict':False}} for task in ['COPY','SUBTRACT','COMPARE'] for i in range(16)]
  self.assertIsNone(screen_summary(rows(12))['selected'])
  self.assertEqual(screen_summary(rows(13))['selected'],'COPY')
  for n,passed in [(25,False),(26,True)]:self.assertEqual(qualify_summary([{'parse':{'correct':i<n,'strict':False}} for i in range(32)])['qualified'],passed)

class GenerationTest(unittest.TestCase):
 def test_fake_generation_eos_and_prefix(self):
  import torch
  class A:
   def __init__(self):self.lengths=[]
   def __call__(self,**kw):
    self.lengths.append(kw['input_ids'].shape[1]);assert kw['cache'] is None
    z=torch.zeros((1,self.lengths[-1],6400));z[0,-1,51 if len(self.lengths)==1 else 2]=1;return {'logits':z}
  a=A();r=generate(a,[7,8],lambda:None,lambda x:None)
  self.assertEqual(r['tokens'],[51,2]);self.assertEqual(r['positions'],5);self.assertEqual(a.lengths,[2,3]);self.assertEqual(r['stop'],'eos')
 def test_tie_cutoff_and_nonfinite(self):
  import torch
  class A:
   def __init__(self,bad=False):self.bad=bad
   def __call__(self,**kw):
    z=torch.zeros((1,kw['input_ids'].shape[1],6400));z[0,-1,0]=float('nan') if self.bad else 0;return {'logits':z}
  r=generate(A(),[7],lambda:None,lambda x:None)
  self.assertEqual(r['tokens'],[0]*24);self.assertEqual(r['stop'],'length');self.assertEqual(r['positions'],300)
  with self.assertRaises(ValueError):generate(A(True),[7],lambda:None,lambda x:None)
 def test_last_eos(self):
  import torch
  class A:
   def __call__(self,**kw):
    z=torch.zeros((1,kw['input_ids'].shape[1],6400));z[0,-1,2 if kw['input_ids'].shape[1]==24 else 51]=1;return {'logits':z}
  self.assertEqual(generate(A(),[7],lambda:None,lambda x:None)['stop'],'eos')
if __name__=='__main__':unittest.main()
