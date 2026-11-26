from pathlib import Path
import json,gzip,time,hashlib,pickle,sys
import numpy as np
from sklearn.metrics import average_precision_score,f1_score
from threadpoolctl import threadpool_limits
from engine import Config,Session,initialize,estimator,balanced,digest
ROOT=Path(__file__).resolve().parent;SEEDS=[11,23,37,51,79];SCENARIOS=['clean1','delay80','feedback_corrected','checker_fault']

def metrics(y,p):
 y=np.asarray(y);p=np.clip(np.asarray(p),0,1);h=p>=.5;tp=int(((y==1)&h).sum());fn=int(((y==1)&~h).sum());tn=int(((y==0)&~h).sum());fp=int(((y==0)&h).sum());recall=tp/(tp+fn) if tp+fn else float('nan');fpr=fp/(fp+tn) if fp+tn else float('nan')
 return dict(n=len(y),tp=tp,fn=fn,tn=tn,fp=fp,balanced_accuracy=(recall+1-fpr)/2,recall=recall,fnr=1-recall,fpr=fpr,macro_f1=float(f1_score(y,h,average='macro',labels=[0,1],zero_division=0)),AP=float(average_precision_score(y,p)) if (y==1).any() else float('nan'),brier=float(np.mean((y-p)**2)),security_loss=float((4*fn+fp)/len(y)))
def load(ds,seed):
 if ds in ['nsl','unsw']:
  a=dict(np.load(ROOT/f'inputs/{ds}_{seed}.npz'));order=np.random.default_rng(1111).permutation(len(a['test']));ix=np.sort(order[:4000]) if ds=='nsl' else a['stream_main'];a['test']=a['test'][ix];a['y_test']=a['y_test'][ix]
  if 'test_family' in a:a['test_family']=a['test_family'][ix]
  a['test_source_index']=ix;return a
 a=dict(np.load(ROOT/f'inputs/history_{seed}.npz'));path=ROOT/f'inputs/{ds}_{seed}.npz' if ds in ['wednesday','friday'] else ROOT/f'data/external/{ds}_{seed}.npz';a.update(dict(np.load(path)));return a
def name(cfg):return f'{cfg.kind}_{cfg.route}_P{int(cfg.pseudo)}_{cfg.query}_w{cfg.positive_weight:g}'
def save(out,r,tr,ledger):
 stem=r['run_id'].replace(':','_');np.savez_compressed(out/(stem+'.npz'),**tr)
 with gzip.open(out/(stem+'_ledger.json.gz'),'wt',encoding='utf-8') as f:json.dump(ledger,f)
def simulate(a,seed,scenario,cfg,init=None,keep_session=False):
 tick=time.perf_counter();core=Session({k:a[k] for k in ['base','meta','cal','y_base','y_meta','y_cal']},seed,cfg,init);n=len(a['test']);rng=np.random.default_rng(seed+995);wrong=rng.random(n)<.1 if scenario=='feedback_corrected' else np.zeros(n,bool);faultdraw=np.random.default_rng(seed+777).random(n)<.25;delivery={};correction={};blocks=[];sourcecheck=[];query=np.zeros(n,bool);held=np.zeros(n,bool);arrival=np.full(n,n+1000,int);fault=np.zeros(n,bool)
 initialsource=core.source_predictions();initial_y=initialsource['y'];initial1=metrics(initial_y,initialsource['p1']);initial2=metrics(initial_y,initialsource['p2'])
 for start in range(0,n,100):
  end=min(start+100,n);mask=(a['y_test'][start:end]==1)&faultdraw[start:end] if scenario=='checker_fault' else np.zeros(end-start,bool);fault[start:end]=mask
  def modify(p):p[mask]=.001;return p
  trace=core.predict_block(a['test'][start:end],modify if scenario=='checker_fault' else None);blocks.append(trace)
  for pos,i in enumerate(range(start,end)):
   issued={k:v[pos] for k,v in trace.items()};core.observe(a['test'][i],issued)
   for j in delivery.get(i,[]):core.receive(dict(event=j,t=i,label=int(a['y_test'][j])^int(wrong[j]),held=bool(held[j]),type='feedback',revision=0,origin='simulator'))
   for j in correction.get(i,[]):core.receive(dict(event=j,t=i,label=int(a['y_test'][j]),held=bool(held[j]),type='correction',revision=1,origin='simulator'))
  requests=core.request_block(trace)
  for req in requests:
   j=req['event'];query[j]=True;held[j]=req['held'];at=req['request_t']+(1 if scenario=='clean1' else 80);arrival[j]=at;delivery.setdefault(at,[]).append(j)
   if wrong[j]:correction.setdefault(at+400,[]).append(j)
  core.update()
  if end%1000==0 or end==n:
   source=core.source_predictions();sourcecheck.append(dict(t=end-1,labels_requested=len(core.requests),training_arrived=len(core.verified),audit_arrived=len(core.audit),source1_BA=metrics(source['y'],source['p1'])['balanced_accuracy'],source2_BA=metrics(source['y'],source['p2'])['balanced_accuracy'],source_BA=metrics(source['y'],source['p'])['balanced_accuracy']))
 trace={k:np.concatenate([block[k] for block in blocks],axis=0) for k in blocks[0]};source=core.source_predictions();r=metrics(a['y_test'],trace['p'])
 for role,key in [('ML1','p1'),('ML2','p2')]:r.update({role+'_'+m:v for m,v in metrics(a['y_test'],trace[key]).items() if m!='n'})
 r.update(seed=seed,scenario=scenario,policy=name(cfg),config=json.dumps(cfg.__dict__,sort_keys=True),queried=int(query.sum()),held_requested=int(held.sum()),arrived_train=len(core.verified),arrived_audit=len(core.audit),human_origin_received=sum(e.get('origin')=='human' for e in core.events),feedback_origin='simulator',updates=core.updates,recoveries=core.recoveries,training_rows=core.train_rows,fit_s=core.fit_s,inspector_s=core.inspector_s,controller_s=core.controller_s,online_s=time.perf_counter()-tick,source_BA=metrics(source['y'],source['p'])['balanced_accuracy'],source1_BA=metrics(source['y'],source['p1'])['balanced_accuracy'],source2_BA=metrics(source['y'],source['p2'])['balanced_accuracy'],initial_source1_BA=initial1['balanced_accuracy'],initial_source2_BA=initial2['balanced_accuracy'],prediction_sha256=digest(trace['p']),source_prediction_sha256=digest(source['p']))
 trace.update(y=a['y_test'],query=query,held=held,arrival=arrival,feedback_wrong=wrong,checker_fault=fault,source_y=source['y'],source_p=source['p'],source_p1=source['p1'],source_p2=source['p2'],source_indices=source['indices'])
 for key in ['test_family','test_raw_rows','test_timestamps','test_attempted','test_feature_key','test_source_index']:
  if key in a:trace[key]=a[key]
 ledger=dict(config=cfg.__dict__,requests=list(core.requests.values()),events=core.events,fits=core.fitlog,progress=core.progress,source_checkpoints=sourcecheck,initial_source_rows=core.init['source'].tolist(),initial_inspector_rows=core.init['prototypes'].tolist(),source_replay_rows=core.init['old'].tolist(),source_initial1=initial1,source_initial2=initial2,scope='Prediction before feedback. Gold exists only in simulator/evaluator. No real human labels claimed.')
 return (r,trace,ledger,core) if keep_session else (r,trace,ledger)

def oracle(a,seed,init):
 """100%perfect past labels, separate nondeployable upper reference."""
 source=init['source'];model=__import__('copy').deepcopy(init['m1']);n=len(a['test']);p=np.zeros(n);fits=[];t0=time.perf_counter();rows=0
 for start in range(0,n,500):
  end=min(start+500,n);p[start:end]=model.predict_proba(a['test'][start:end])[:,1];ids=np.arange(end);xx=np.vstack([a['base'][source],a['test'][ids]]);yy=np.r_[a['y_base'][source],a['y_test'][ids]];weights=np.r_[np.ones(len(source)),np.full(len(ids),4.)];model=estimator('hgb',seed).fit(xx,yy,sample_weight=weights);fits.append(dict(t=end-1,observed_training_ids=ids.tolist(),weight_hash=digest(yy,weights),training_rows=len(yy)));rows+=len(yy)
 cal=init['cal'];sp=model.predict_proba(a['cal'][cal])[:,1];r=metrics(a['y_test'],p);r.update(seed=seed,scenario='clean1',policy='oracle_100pct',feedback_origin='perfect_past_gold_upperbound',queried=n,held_requested=0,arrived_train=n,arrived_audit=0,human_origin_received=0,updates=len(fits),training_rows=rows,online_s=time.perf_counter()-t0,source_BA=metrics(a['y_cal'][cal],sp)['balanced_accuracy'],source1_BA=metrics(a['y_cal'][cal],sp)['balanced_accuracy'],prediction_sha256=digest(p),source_prediction_sha256=digest(sp));tr=dict(y=a['y_test'],p=p,source_y=a['y_cal'][cal],source_p=sp,source_indices=cal)
 for key in ['test_family','test_raw_rows','test_timestamps']:
  if key in a:tr[key]=a[key]
 return r,tr,dict(fits=fits,source_ids=source.tolist(),scope='Oracle100percentperfectpastlabels; no futurelabels; no matchedlabelbudgetclaim')
