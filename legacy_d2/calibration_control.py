"""A prospective-score rule replay; uses arrived held labels, never final-test tuning."""
from pathlib import Path
import json,gzip,hashlib,time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'inputs/d1/src'));from model import metrics
GRID=np.asarray([.1,.2,.3,.4,.5,.6,.7,.8,.9])
def shifted(p,threshold):
 p=np.clip(p,1e-10,1-1e-10);return 1/(1+np.exp(-(np.log(p/(1-p))-np.log(threshold/(1-threshold)))))
def replay(raw,events):
 n=len(raw);audit={};bytime={};threshold=.5;pred=np.zeros(n)
 for e in events:
  if e['held']:bytime.setdefault(e['t'],[]).append(e)
 for i in range(n):
  pred[i]=shifted(raw[i],threshold)
  for e in bytime.get(i,[]):audit[e['event']]=int(e['label'])
  if (i+1)%100==0 and len(audit)>=20:
   ids=np.asarray(sorted(audit)[-128:],int);y=np.asarray([audit[j] for j in ids]);losses=[]
   for cut in GRID:
    h=raw[ids]>=cut;losses.append(float(np.mean(4*((y==1)&~h)+((y==0)&h))))
   best=min(losses);choices=[v for v,l in zip(GRID,losses) if l==best];threshold=float(sorted(choices,key=lambda v:(abs(v-.5),v))[0])
 return pred,threshold
def main(phase='formal'):
 out=ROOT/f'results/{phase}_calibrated';out.mkdir(exist_ok=True);df=pd.read_csv(ROOT/f'results/{phase}/runs.csv');rows=[]
 for _,r in df[(df.policy=='verified_only')|df.policy.str.startswith('dual_')].iterrows():
  stem=r.run_id.replace(':','_')
  with np.load(ROOT/f'results/{phase}'/f'{stem}.npz') as z:tr={k:z[k] for k in z.files}
  with gzip.open(ROOT/f'results/{phase}'/f'{stem}_ledger.json.gz','rt',encoding='utf8') as f:events=json.load(f)['events']
  tick=time.perf_counter();p,cut=replay(tr['p'],events);controller_s=time.perf_counter()-tick;source=shifted(tr['source_p'],cut);newname='calibrated_verified' if r.policy=='verified_only' else 'cal_'+r.policy
  rr=r.to_dict();rr.update(metrics(tr['y'],p));rr.update(policy=newname,algorithm='issued-audit-threshold-rule+'+r.policy,parent_policy=r.policy,source_BA=float(balanced_accuracy_score(tr['source_y'],source>=.5)),online_loop_s=float(r.online_loop_s)+controller_s,controller_s=controller_s,final_threshold=cut,prediction_sha256=hashlib.sha256(p.tobytes()).hexdigest(),source_prediction_sha256=hashlib.sha256(source.tobytes()).hexdigest(),run_id=r.run_id.replace(str(r.policy),newname))
  tr.update(p=p,source_p=source);np.savez_compressed(out/(rr['run_id'].replace(':','_')+'.npz'),**tr);rows.append(rr)
 pd.DataFrame(rows).to_csv(out/'runs.csv',index=False);print('decision-controller replays',len(rows));print(pd.DataFrame(rows).groupby(['dataset','policy'])[['balanced_accuracy','source_BA','fnr']].mean().to_string())
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'formal')
