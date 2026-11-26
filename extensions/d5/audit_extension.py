from pathlib import Path
import json,gzip,hashlib,itertools,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import pandas as pd
from adapter import ROOT,D4,event_hash
from engine import digest
from audit_results import metrics
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 lock=json.loads((ROOT/'protocol/confirmation_lock.json').read_text())
 for e in lock['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
 for e in lock['d4_files']:assert sha(D4/e['path'])==e['sha256'],e['path']
 original=json.loads((D4/'protocol/confirmation_lock.json').read_text())
 for e in original['files']:assert sha(D4/e['path'])==e['sha256'],e['path']
 df=pd.read_csv(ROOT/'results/runs.csv');assert len(df)==240 and df.run_id.is_unique
 worst=0.;fit_checks=label_checks=matched=hgb_fit_checks=hgb_label_checks=controller_blocks=0;reports=[];base_labels={}
 for row in df.to_dict('records'):
  stem=row['run_id'].replace(':','_');p=ROOT/'results'/(stem+'.npz');tr=dict(np.load(p));log=json.loads(gzip.decompress((ROOT/'results'/(stem+'_ledger.json.gz')).read_bytes()));assert len(tr['p'])==12000
  assert np.isfinite(tr['p']).all() and ((tr['p']>=0)&(tr['p']<=1)).all();assert digest(tr['p'])==row['prediction_sha256'];assert event_hash(log)==row['feedback_schedule_sha256']
  for key,value in metrics(tr['y'],tr['p']).items():
   if key in row and pd.notna(row[key]):error=abs(float(row[key])-float(value));worst=max(worst,error);assert error<1e-12,(stem,key,error)
  assert abs(metrics(tr['source_y'],tr['source_p'])['balanced_accuracy']-row['source_BA'])<1e-12
  req={e['event']:e for e in log['requests']};assert len(req)==600 and sum(e['held'] for e in req.values())==120
  events=sorted(log['events'],key=lambda e:e['t']);target={};audit={}
  for e in events:
   j=e['event'];assert j in req and req[j]['held']==e['held'] and e['t']>=req[j]['request_t'] and j<e['t']<12000
   expected_delay=1 if row['scenario']=='clean1' else 80;expected_t=req[j]['request_t']+expected_delay+(400 if e['type']=='correction' else 0);assert e['t']==expected_t
   (audit if e['held'] else target)[j]=e['label']
  assert len(target)==row['arrived_train'] and len(audit)==row['arrived_audit']
  # Reconstruct the controller independently from the adapter/learner functions.
  cursor=0;held_state={};cut=.5
  for start in range(0,12000,100):
   end=start+100;assert np.all(tr['threshold'][start:end]==cut)
   clipped=np.clip(tr['raw'][start:end],1e-10,1-1e-10);odds=(clipped/(1-clipped))*((1-cut)/cut);assert np.allclose(odds/(1+odds),tr['p'][start:end],atol=1e-14,rtol=0)
   while cursor<len(events) and events[cursor]['t']<end:
    e=events[cursor]
    if e['held']:held_state[e['event']]=e['label']
    cursor+=1
   if row['policy']!='frozen_detector' and len(held_state)>=20:
    ix=np.asarray(sorted(held_state)[-128:],int);yy=np.asarray([held_state[j] for j in ix]);candidates=[]
    for candidate in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
     hh=tr['raw'][ix]>=candidate;candidates.append((int((4*((yy==1)&~hh)+((yy==0)&hh)).sum()),abs(candidate-.5),candidate))
    cut=min(candidates)[-1]
   controller_blocks+=1
  if row['policy']=='new_selected':
   assert np.array_equal(np.where(tr['eligible'],np.maximum(tr['p1'],tr['p2']),tr['p1']),tr['raw'])
   if row['seed'] not in base_labels:base_labels[row['seed']]=np.load(D4/f"inputs/history_{row['seed']}.npz")['y_base']
   known={};cursor=0;fitted=updates=recoveries=0
   for f in log['fits']:
    while cursor<len(events) and events[cursor]['t']<=f['t']:
     e=events[cursor]
     if not e['held']:known[e['event']]=e['label']
     cursor+=1
    ids=np.asarray(f['verified_ids'],int);source=np.asarray(f['source_ids'],int);yy=np.asarray([known[j] for j in ids],int);assert ids.tolist()==sorted(known)[-1024:];assert all(not req[j]['held'] for j in ids)
    if f['role']=='ML2':assert yy.tolist()==f['labels'];weight=np.r_[np.ones(len(source)),np.full(len(ids),4.)]
    else:
     assert yy.tolist()==f['verified_labels'] and f['pseudo_ids']==[];weight=np.r_[np.ones(len(source)),4*np.where(yy==1,2.,1.)];fitted+=f['training_rows'];updates+=1;recoveries+=int(f['recover'])
    combined=np.r_[base_labels[row['seed']][source],yy];assert digest(combined,weight)==f['weight_hash'];assert len(combined)==f['training_rows'];hgb_fit_checks+=1;hgb_label_checks+=len(ids)
   assert fitted==row['training_rows'] and updates==row['updates'] and recoveries==row['recoveries']
  if row['policy']=='arf_native_matched':
   folder=D4/'results/confirmation' if row['dataset'] in ['dos_main','bot_main'] else ROOT/'results';refstem=f"{row['dataset']}_{row['seed']}_{row['scenario']}_new_selected";reference=json.loads(gzip.decompress((folder/(refstem+'_ledger.json.gz')).read_bytes()));assert event_hash(reference)==event_hash(log);assert reference['initial_source_rows']==log['initial_source_rows'];matched+=1
   current={};used={};cursor=0
   for f in log['fits']:
    while cursor<len(events) and events[cursor]['t']<=f['t']:
     e=events[cursor]
     if not e['held']:current[e['event']]=e['label']
     cursor+=1
    changed=any(current[j]!=y for j,y in used.items());assert changed==f['reset']
    if changed:used={}
    for j,y in zip(f['trained_ids'],f['labels']):
     assert j in current and y==current[j] and not req[j]['held'];assert j not in used;used[j]=y;label_checks+=1
    assert digest(np.asarray(list(current),int),np.asarray(list(current.values()),int))==f['current_label_hash'];fit_checks+=1
   for start in range(0,12000,100):assert np.all(tr['version1'][start:start+100]==sum(f['t']<start for f in log['fits']))
   assert row['recoveries']==sum(f['reset'] for f in log['fits'])
  if 'test_timestamps' in tr:assert np.all(np.diff(tr['test_timestamps'])>=0)
  reports.append(dict(run_id=row['run_id'],passed=True))
 oldkeys=set()
 for p in (D4/'data/external').glob('*_11.npz'):
  with np.load(p) as a:oldkeys.update(map(int,a['test_feature_key']))
 inputs={}
 for ds in ['dos16_new','ddos20_new']:
  a=dict(np.load(ROOT/f'data/{ds}_11.npz'));keys=set(map(int,a['test_feature_key']));assert len(keys)==12000 and not keys&oldkeys
  assert a['test'].shape==(12000,82) and np.isfinite(a['test']).all();inputs[ds]=keys
 cross=len(inputs['dos16_new']&inputs['ddos20_new'])
 report=dict(passed=True,system_evaluations_audited=len(reports),native_ARF_feedback_schedules_matched=matched,ARF_fit_events_checked=fit_checks,ARF_training_label_uses_checked=label_checks,ARF_audit_training_uses=0,HGB_NN_fit_events_checked=hgb_fit_checks,HGB_NN_training_label_uses_checked=hgb_label_checks,independent_controller_blocks_checked=controller_blocks,arithmetic_uses_independent_D4_auditor=True,max_metric_recompute_error=worst,new_capture_overlap_with_old_sampled_2018=0,new_capture_cross_overlap_fingerprints=cross,D4_original_lock_files_unchanged=len(original['files']),D5_lock_verified=True,scope='All completed extension runs audited; shared routing evaluations are correlated. Cross-capture repeated predictor fingerprints, if present, are reported rather than claimed independent.',runs=reports)
 (ROOT/'checks/all_run_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
