# Extra checks: uncentred direct path; sign-symmetry of random controls (norm effect).
import sys; sys.argv=['x']
exec(open('measure.py').read().split('rows = []')[0])
import json
for gpath, nid in [("graphs/gemma-addition","24_14176_6"),("graphs2/gemma-small-big-zh","24_2394_8"),("graphs/gemma-addition","25_14682_6")]:
    g=json.load(open(f"{EXP}/{gpath}.json")); ids=tok(g["metadata"]["prompt"][5:],return_tensors="pt").input_ids.to(dev)
    tn=max([n for n in g["nodes"] if n["feature_type"]=="logit"],key=lambda n:n["token_prob"]); t=int(tn["node_id"].split("_")[1])
    L,f,pos=map(int,nid.split("_")); T=transcoder(L)
    pre0,_,cap0=run(ids); x=cap0["premlp"][L][0,pos]
    pr=rmsnorm_nogain(x)@T["W_enc"][:,f]+T["b_enc"][f]; a=pr.item(); c=a*T["W_dec"][f]
    fin=cap0["final_in"][0,-1]; y=c*torch.rsqrt(fin.pow(2).mean()+eps)*(1+model.model.norm.weight); dl=y@WU.T
    d=T["W_dec"][f]/T["W_dec"][f].norm(); res=[]
    for s in SEEDS:
        v=torch.tensor(np.random.Generator(np.random.PCG64(s)).standard_normal(2304),dtype=torch.float32,device=dev); v=v-(v@d)*d; v=v/v.norm()*c.norm()
        e=[mc(pre0,t)-mc(run(ids,(L,pos,sg*v))[0],t) for sg in (1,-1)]; res.append(e)
    res=np.array(res)
    _,_,capE=run(ids,(L,pos,c)); _,_,capR=run(ids,(L,pos,v))
    print(nid, "direct_centred",(dl[t]-dl.mean()).item(),"direct_uncentred",dl[t].item(),"W",[l["weight"] for l in g["links"] if l["source"]==nid and l["target"]==tn["node_id"]],
          "| C_R(-v) mean",res[:,0].mean(),"C_R(+v) mean",res[:,1].mean(),"| final resid norm base",fin.norm().item(),"after feature removal",capE["final_in"][0,-1].norm().item(),"after random",capR["final_in"][0,-1].norm().item())
