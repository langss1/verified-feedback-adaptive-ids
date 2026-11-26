from pathlib import Path
import json,gzip,pickle,time,inspect
import numpy as np
from threadpoolctl import threadpool_limits
from adapter import ROOT,D4,derive_route,event_hash,new_arf,dictionaries,run_arf,ARFAdapter,probabilities

def main():
 start=time.time();checks=[];worst=0.;schedules=0
 # Independently recorded D4 runs verify exact controller replay, including corrections.
 for ds in ['dos_main','bot_main']:
  for seed in [11,23,37,51,79]:
   for scenario in ['clean1','delay80','feedback_corrected','checker_fault']:
    prefix=f'{ds}_{seed}_{scenario}_';folder=D4/'results/confirmation'
    tr=dict(np.load(folder/(prefix+'new_selected.npz')));log=json.loads(gzip.decompress((folder/(prefix+'new_selected_ledger.json.gz')).read_bytes()))
    for route,alias in [('verified','matched_verified'),('max','simple_max'),('frozen','frozen_detector')]:
     ref=dict(np.load(folder/(prefix+alias+'.npz')));rlog=json.loads(gzip.decompress((folder/(prefix+alias+'_ledger.json.gz')).read_bytes()))
     got,cut=derive_route(tr,log,route,ref['p1'],ref['source_p1']);delta=float(np.max(np.abs(got['p']-ref['p'])));worst=max(worst,delta)
     assert np.allclose(got['p'],ref['p'],rtol=0,atol=1e-14),(ds,seed,scenario,alias,delta)
     assert np.array_equal(got['p']>=.5,ref['p']>=.5)
     assert np.allclose(got['source_p'],ref['source_p'],rtol=0,atol=1e-14)
     assert event_hash(log)==event_hash(rlog);schedules+=1
     checks.append(dict(dataset=ds,seed=seed,scenario=scenario,route=alias,max_error=delta))
 # Small synthetic software fixtures, not research measurements.
 rng=np.random.default_rng(17);x=rng.normal(size=(400,4));ys=(x[:,0]+x[:,1]>.0).astype(int);model=new_arf(11)
 for row,y in zip(dictionaries(x[:128]),ys[:128]):model.learn_one(row,int(y))
 blob=pickle.dumps(model);events=[dict(event=j,t=101,label=int(ys[j]),held=j!=0,type='feedback',revision=0) for j in range(21)];events.append(dict(event=0,t=250,label=1-int(ys[0]),held=False,type='correction',revision=1))
 p,detail=run_arf(x,blob,events,x[:32]);p2,detail2=run_arf(x,blob,events,x[:32]);assert np.array_equal(p['p'],p2['p'])
 auditflip=[{**e,'label':1-e['label']} if e['held'] else e for e in events];other,_=run_arf(x,blob,auditflip,x[:32]);assert np.array_equal(p['raw'],other['raw']) and np.array_equal(p['source_p1'],other['source_p1'])
 assert all(not any(j in range(1,21) for j in f['trained_ids']) for f in detail['fits'])
 assert detail['resets']==1
 no_correction,_=run_arf(x,blob,events[:-1],x[:32]);assert np.array_equal(no_correction['p'][:300],p['p'][:300])
 rows=dictionaries(x);core=ARFAdapter(blob,rows);core.raw.extend([.5]*200);core.receive_and_update(events[:21],200);core.raw.extend([.5]*100);core.receive_and_update([events[-1]],300)
 reference=pickle.loads(blob);reference.learn_one(rows[0],1-int(ys[0]));assert np.array_equal(probabilities(core.model,rows[300:]),probabilities(reference,rows[300:]))
 assert 'y_test' not in inspect.signature(run_arf).parameters
 report=dict(passed=True,old_D4_route_replays=len(checks),max_probability_error=worst,exact_decision_replays=True,matched_event_schedules=schedules,native_ARF_deterministic=True,audits_excluded_from_training=True,audit_label_changes_leave_raw_model_scores_unchanged=True,correction_replay_equals_initial_plus_latest_label=True,correction_does_not_rewrite_past_predictions=True,learner_function_has_no_target_gold_argument=True,software_fixtures_are_not_research_results=True,seconds=time.time()-start,details=checks)
 (ROOT/'checks/adapter_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
