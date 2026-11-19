import torch,pytest
from minimind_re.softmax_bucket_candidate import cached_bucket_softmax
@pytest.mark.parametrize('width',[1,5,6,8,9])
def test_fixed_rule(width):
 torch.manual_seed(19);s=torch.randn(1,2,width,width,dtype=torch.float32);s=s.masked_fill(torch.triu(torch.ones(width,width,dtype=torch.bool),1),float('-inf'));s[...,0]-=1e9
 before=s.clone();b=1<<(width-1).bit_length();p=torch.full((1,2,width,b),float('-inf'));p[...,:width]=s
 expected=torch.softmax(p,-1)[...,:width].contiguous()
 with torch.inference_mode():actual=cached_bucket_softmax(s)
 assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));assert actual.is_contiguous();assert torch.equal(s,before)
 assert torch.isfinite(actual).all()
def test_full_path_unchanged():
 import ast,pathlib
 a=ast.parse(pathlib.Path('src/minimind_re/shape_cache_matrix_candidate.py').read_text());b=ast.parse(pathlib.Path('src/minimind_re/softmax_bucket_candidate.py').read_text())
 def branch(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='cache is None')
 assert [ast.dump(n) for n in branch(a).body]==[ast.dump(n) for n in branch(b).body]
def test_dtype_rejected():
 with pytest.raises(ValueError):cached_bucket_softmax(torch.zeros(1,1,6,6,dtype=torch.float64))
