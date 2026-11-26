from pathlib import Path
import sys,json,pickle,gzip,time,argparse,hashlib,concurrent.futures
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from adapter import ROOT,D4,new_arf,dictionaries,probabilities,run_arf,derive_route,event_hash,shifted
from engine import initialize,Config,digest
from experiment import metrics,simulate,load,save
SEEDS=[11,23,37,51,79];SCENARIOS=['clean1','delay80','feedback_corrected','checker_fault']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def history(seed):
 a=dict(np.load(D4/f'inputs/history_{seed}.npz'));a.update(dict(np.load(D4/f'inputs/retention_{seed}.npz')));return {k:a[k] for k in ['base','y_base','meta','y_meta','cal','y_cal']}
def cache_seed(seed):
 out=ROOT/'cache';out.mkdir(exist_ok=True);p=out/f'initial_{seed}.pkl'
 if p.exists():return
 with threadpool_limits(limits=1):
  start=time.time();a=history(seed);init=initialize(a,seed,'hgb');model=new_arf(seed)
  for x,y in zip(dictionaries(a['base'][init['source']]),a['y_base'][init['source']]):model.learn_one(x,int(y))
  p.write_bytes(pickle.dumps(dict(init=init,arf=pickle.dumps(model)),protocol=5));print('INITIAL SOURCE CACHE',seed,'seconds',round(time.time()-start),flush=True)
def old_reference(ds,seed,scenario):
 stem=f'{ds}_{seed}_{scenario}_new_selected';out=D4/'results/confirmation'
 return dict(np.load(out/(stem+'.npz'))),json.loads(gzip.decompress((out/(stem+'_ledger.json.gz')).read_bytes()))
def case(ds,seed):
 out=ROOT/'results';rows=[];start=time.time();cached=pickle.loads((ROOT/f'cache/initial_{seed}.pkl').read_bytes());init=cached['init'];cfg=Config(**json.loads((D4/'protocol/selected_method.json').read_text())['config']);a=history(seed)
 if ds in ['dos_main','bot_main']:a.update({k:v for k,v in load(ds,seed).items() if k=='test' or k=='y_test' or k.startswith('test_')})
 else:a.update(dict(np.load(ROOT/f'data/{ds}_{seed}.npz')))
 with threadpool_limits(limits=1):
  for scenario in SCENARIOS:
   marker=out/f'case_{ds}_{seed}_{scenario}.csv'
   if marker.exists():rows.extend(pd.read_csv(marker).to_dict('records'));continue
   case_rows=[]
   if ds in ['dos_main','bot_main']:tr,ledger=old_reference(ds,seed,scenario)
   else:
    r,tr,ledger=simulate(a,seed,scenario,cfg,init);r.update(dataset=ds,policy='new_selected',run_id=f'{ds}:{seed}:{scenario}:new_selected',feedback_schedule_sha256=event_hash(ledger),evidence='new_capture_confirmation');save(out,r,tr,ledger);case_rows.append(r)
    for route,alias in [('verified','matched_verified'),('max','simple_max'),('frozen','frozen_detector')]:
     t,cut=derive_route(tr,ledger,route,init['m1'].predict_proba(a['test'])[:,1] if route=='frozen' else None,init['m1'].predict_proba(a['cal'][init['cal']])[:,1] if route=='frozen' else None);rr=metrics(t['y'],t['p']);rr.update(dataset=ds,seed=seed,scenario=scenario,policy=alias,run_id=f'{ds}:{seed}:{scenario}:{alias}',source_BA=metrics(t['source_y'],t['source_p'])['balanced_accuracy'],source1_BA=metrics(t['source_y'],t['source_p1'])['balanced_accuracy'],initial_source1_BA=r['initial_source1_BA'],queried=r['queried'],held_requested=r['held_requested'],arrived_train=r['arrived_train'],arrived_audit=r['arrived_audit'],updates=0 if route=='frozen' else r['updates'],recoveries=0 if route=='frozen' else r['recoveries'],training_rows=0 if route=='frozen' else r['training_rows'],feedback_schedule_sha256=event_hash(ledger),prediction_sha256=digest(t['p']),source_prediction_sha256=digest(t['source_p']),derived_routing=True,evidence='new_capture_confirmation',final_threshold=cut)
     save(out,rr,t,dict(requests=ledger['requests'],events=ledger['events'],shared_fit_reference=f'{ds}:{seed}:{scenario}:new_selected',scope='Exact routing/controller replay; shared model fits, not independent training runs.'));case_rows.append(rr)
   prediction,detail=run_arf(a['test'],cached['arf'],ledger['events'],a['cal'][init['cal']]);r=metrics(a['y_test'],prediction['p']);r.update(dataset=ds,seed=seed,scenario=scenario,policy='arf_native_matched',run_id=f'{ds}:{seed}:{scenario}:arf_native_matched',source_BA=metrics(a['y_cal'][init['cal']],prediction['source_p'])['balanced_accuracy'],source1_BA=metrics(a['y_cal'][init['cal']],prediction['source_p1'])['balanced_accuracy'],initial_source1_BA=metrics(a['y_cal'][init['cal']],prediction['initial_source_p1'])['balanced_accuracy'],queried=int(tr['query'].sum()),held_requested=int(tr['held'].sum()),arrived_train=detail['progress'][-1]['training_arrived'],arrived_audit=detail['progress'][-1]['audit_arrived'],updates=len(detail['fits']),recoveries=detail['resets'],training_rows=detail['training_work'],fit_s=detail['fit_s'],online_s=detail['online_s'],drifts=detail['drifts_in_final_state'],feedback_schedule_sha256=event_hash(ledger),prediction_sha256=digest(prediction['p']),source_prediction_sha256=digest(prediction['source_p']),feedback_origin='simulator',human_origin_received=0,evidence='retrospective_baseline_extension' if ds in ['dos_main','bot_main'] else 'new_capture_confirmation')
   prediction.update(y=a['y_test'],query=tr['query'],held=tr['held'],arrival=tr['arrival'],source_y=a['y_cal'][init['cal']],source_indices=init['cal'])
   for key in ['test_family','test_feature_key','test_timestamps','test_raw_rows']:prediction[key]=a[key]
   detail.update(requests=ledger['requests'],events=ledger['events'],initial_source_rows=init['source'].tolist(),feedback_schedule_sha256=event_hash(ledger));save(out,r,prediction,detail);case_rows.append(r);pd.DataFrame(case_rows).to_csv(marker,index=False);rows.extend(case_rows)
   print('CASE',ds,seed,scenario,'ARF_BA',round(r['balanced_accuracy'],4),'elapsed',round(time.time()-start),flush=True)
 return len(rows)
def lock():
 assert not (ROOT/'protocol/confirmation_lock.json').exists()
 assert json.loads((ROOT/'checks/adapter_checks.json').read_text())['passed']
 original_lock=json.loads((D4/'protocol/confirmation_lock.json').read_text())
 for e in original_lock['files']:assert sha(D4/e['path'])==e['sha256'],e['path']
 preparation=[json.loads((ROOT/f'data/{d}_preparation.json').read_text()) for d in ['dos16_new','ddos20_new']]
 for p in preparation:assert p['model_scores_computed'] is False
 files=[ROOT/n for n in ['adapter.py','run_extension.py','prepare_new.py','check_adapter.py','protocol/design.json','protocol/environment.json','requirements.txt','checks/adapter_checks.json']]+list((ROOT/'data').glob('*.npz'))+list((ROOT/'data').glob('*preparation.json'))+list((ROOT/'cache').glob('*.pkl'))
 d4files=[D4/n for n in ['engine.py','experiment.py','protocol/selected_method.json','protocol/confirmation_lock.json']]+list((D4/'inputs').glob('history_*.npz'))+list((D4/'inputs').glob('retention_*.npz'))
 record=dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),method_unchanged=True,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files],d4_files=[dict(path=p.relative_to(D4).as_posix(),sha256=sha(p)) for p in d4files],new_captures=[{k:p[k] for k in ['dataset','records','binary_counts','families','adequate_primary_support']} for p in preparation],expected_result_rows=240,expected_native_ARF_runs=80,expected_new_shared_HGB_NN_runs=40,prior_results_known=True)
 (ROOT/'protocol/confirmation_lock.json').write_text(json.dumps(record,indent=2));print('D5 LOCKED',record['created_utc'],flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',action='store_true');ap.add_argument('--freeze',action='store_true');ap.add_argument('--workers',type=int,default=5);args=ap.parse_args()
 if args.freeze:lock();return
 if args.cache:
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(cache_seed,SEEDS))
  return
 plan=json.loads((ROOT/'protocol/confirmation_lock.json').read_text())
 for e in plan['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
 for e in plan['d4_files']:assert sha(D4/e['path'])==e['sha256'],e['path']
 start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  jobs=[pool.submit(case,d,s) for d in ['dos_main','bot_main','dos16_new','ddos20_new'] for s in SEEDS]
  for job in concurrent.futures.as_completed(jobs):print('FINISHED',job.result(),flush=True)
 data=pd.concat([pd.read_csv(p) for p in (ROOT/'results').glob('case_*.csv')],ignore_index=True);assert len(data)==240 and data.run_id.is_unique;data.to_csv(ROOT/'results/runs.csv',index=False);(ROOT/'results/receipt.json').write_text(json.dumps(dict(completed=True,result_rows=len(data),native_ARF_runs=int((data.policy=='arf_native_matched').sum()),shared_HGB_NN_runs=int((data.policy=='new_selected').sum()),seconds=time.time()-start),indent=2));print('EXTENSION COMPLETE',len(data),flush=True)
if __name__=='__main__':main()
