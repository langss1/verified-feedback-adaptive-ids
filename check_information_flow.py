from pathlib import Path
import json,copy
import numpy as np
from threadpoolctl import threadpool_limits
from engine import Config
from experiment import load,simulate
ROOT=Path(__file__).resolve().parent
def main():
 (ROOT/'checks').mkdir(exist_ok=True);a=load('nsl',11)
 for k in list(a):
  if k=='test' or k.startswith('test_') or k=='y_test':a[k]=a[k][:1200]
 cfg=Config(pseudo=True)
 with threadpool_limits(limits=1):
  r,tr,log=simulate(a,11,'delay80',cfg);b=copy.deepcopy(a);b['y_test'][~tr['query']]=1-b['y_test'][~tr['query']];_,other,_=simulate(b,11,'delay80',cfg);assert np.array_equal(tr['p'],other['p']) and np.array_equal(tr['query'],other['query'])
  b=copy.deepcopy(a);b['y_cal']=1-b['y_cal'];_,other,_=simulate(b,11,'delay80',cfg);assert np.array_equal(tr['p'],other['p'])
  b=copy.deepcopy(a);b['test'][950:]=np.random.default_rng(2).normal(size=b['test'][950:].shape);b['y_test'][950:]=1-b['y_test'][950:];_,other,_=simulate(b,11,'delay80',cfg);assert np.array_equal(tr['p'][:950],other['p'][:950])
  _,control,_=simulate(a,11,'delay80',Config(route='verified',pseudo=False));assert np.array_equal(tr['query'],control['query']) and np.array_equal(tr['held'],control['held']) and np.array_equal(tr['p2'],control['p2'])
  for f in log['fits']:
   assert all(tr['query'][j] and not tr['held'][j] and tr['arrival'][j]<=f['t'] for j in f['verified_ids'])
   if f['role']=='ML1':assert all(not tr['held'][j] for j in f['pseudo_ids']);assert sum(f['pseudo_weights'])<=.25*len(f['verified_ids'])+1e-8
 result=dict(passed=True,unqueried_gold_noninterference=True,CAL_label_noninterference=True,future_feature_label_prefix_invariance=True,dynamic_query_contract_common_across_matched_policies=True,held_audit_excluded_from_both_classifiers=True,pseudo_weight_cap=True,scope='One1200eventdevelopmentcase metamorphic checks; independentallrunauditfollows; not universalproof')
 (ROOT/'checks/information_flow.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
