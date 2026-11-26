"""Independent saved-output arithmetic, feedback availability and fit-input audit.
Does not call either learner or the experiment metric function.
"""
from pathlib import Path
import json,gzip,hashlib,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(*v):return hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in v)).hexdigest()
def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();h=p>=.5;tp=int(((y==1)&h).sum());fn=int(((y==1)&~h).sum());fp=int(((y==0)&h).sum());tn=int(((y==0)&~h).sum());tpr=tp/(tp+fn) if tp+fn else float('nan');fpr=fp/(fp+tn) if fp+tn else float('nan');order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(p)-1];cum=np.cumsum(ys)[ends];increments=np.diff(np.r_[0,cum]);ap=float(np.sum(increments*cum/(ends+1))/y.sum()) if y.sum() else float('nan')
 return dict(n=len(y),tp=tp,fn=fn,tn=tn,fp=fp,balanced_accuracy=(tpr+1-fpr)/2,recall=tpr,fnr=1-tpr,fpr=fpr,macro_f1=.5*(2*tp/max(1,2*tp+fp+fn)+2*tn/max(1,2*tn+fp+fn)),AP=ap,brier=float(np.mean((y-p)**2)),security_loss=(4*fn+fp)/len(y))
def arrays(ds,seed,phase):
 if ds in ['nsl','unsw']:
  a=dict(np.load(ROOT/f'inputs/{ds}_{seed}.npz'));ix=np.sort(np.random.default_rng(1111).permutation(len(a['test']))[:4000]) if ds=='nsl' else a['stream_main'];a['test']=a['test'][ix];a['y_test']=a['y_test'][ix];return a
 a=dict(np.load(ROOT/f'inputs/history_{seed}.npz'));p=ROOT/f'inputs/{ds}_{seed}.npz' if ds in ['wednesday','friday'] else ROOT/f'data/external/{ds}_{seed}.npz';a.update(dict(np.load(p)))
 if phase=='confirmation':ret=dict(np.load(ROOT/f'inputs/retention_{seed}.npz'));a['cal']=ret['cal'];a['y_cal']=ret['y_cal']
 return a
def compare(row,y,p,prefix=''):
 count=0;maxerr=0.
 for k,v in metrics(y,p).items():
  field=prefix+k
  if field not in row or pd.isna(row[field]):assert np.isnan(v) or field not in row,(field,v);continue
  error=abs(float(v)-float(row[field]));maxerr=max(maxerr,error);assert error<1e-10,(field,v,row[field]);count+=1
 return count,maxerr
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phases',nargs='*',default=['screen','check','confirmation']);ap.add_argument('--partial',action='store_true',help='Audit completed case rows while a stage runs; save a separate partial receipt');args=ap.parse_args();total=checks=0;maxerr=0.;contracts={};cache={};training=[]
 for phase in args.phases:
  lockfile=ROOT/f'protocol/{phase}_lock.json'
  if lockfile.exists():
   for e in json.loads(lockfile.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256'];checks+=1
  directory=ROOT/f'results/{phase}';df=pd.concat([pd.read_csv(p) for p in directory.glob('case_*.csv')],ignore_index=True) if args.partial else pd.read_csv(directory/'runs.csv');assert df.run_id.is_unique
  for _,r in df.iterrows():
   stem=r.run_id.replace(':','_');z=dict(np.load(directory/(stem+'.npz')))
   with gzip.open(directory/(stem+'_ledger.json.gz'),'rt',encoding='utf-8') as f:l=json.load(f)
   c,error=compare(r,z['y'],z['p']);checks+=c;maxerr=max(maxerr,error);total+=1;assert digest(z['p'])==r.prediction_sha256 and digest(z['source_p'])==r.source_prediction_sha256;assert abs(metrics(z['source_y'],z['source_p'])['balanced_accuracy']-r.source_BA)<1e-10
   key=(phase,r.dataset,int(r.seed))
   if key not in cache:cache[key]=arrays(r.dataset,int(r.seed),phase)
   a=cache[key];assert np.array_equal(a['y_test'],z['y']);cal=z.get('source_indices',z.get('source_cal_indices'));assert np.array_equal(a['y_cal'][cal],z['source_y'])
   if r.policy=='oracle_100pct':
    source=np.asarray(l['source_ids'],int);kind=l['kind'];pos=l['positive_weight'];rows=0
    for f in l['fits']:
     ids=np.asarray(f['observed_training_ids'],int);assert np.array_equal(ids,np.arange(f['t']+1));yy=np.r_[a['y_base'][source],z['y'][ids]];w=np.r_[np.ones(len(source)),4*np.where(z['y'][ids]==1,pos,1.)];assert digest(yy,w)==f['weight_hash'] and f['training_rows']==len(yy);rows+=len(yy);checks+=3
    assert rows==r.training_rows;continue
   if r.policy=='old_rvea_full':
    # Primary arithmetic and legacy timing assertions; separate legacy-fit pass below.
    events=l['events'];n=len(z['y']);query=z['query'];held=z['held'];arrival=z['arrival'];wrong=z['feedback_wrong']
    for role,keyp in [('ML1','p1'),('ML2','p2')]:c,error=compare(r,z['y'],z[keyp],role+'_');checks+=c;maxerr=max(maxerr,error)
    # Independent reconstruction of the predecessor's arrived-audit score rule.
    audit={};position=0;cut=.5;replayed=np.zeros(n)
    for start in range(0,n,100):
     end=min(start+100,n);raw=np.clip(z['raw'][start:end],1e-10,1-1e-10);odds=raw/(1-raw)*(1-cut)/cut;replayed[start:end]=odds/(1+odds)
     while position<len(events) and events[position]['t']<end:
      e=events[position]
      if e['held']:audit[e['event']]=e['label']
      position+=1
     if end%100 or len(audit)<20:continue
     ids=np.asarray(sorted(audit)[-128:],int);yy=np.asarray([audit[j] for j in ids]);choices=[]
     for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
      h=z['raw'][ids]>=threshold;loss=int((4*((yy==1)&~h)+((yy==0)&h)).sum());choices.append((loss,abs(threshold-.5),threshold))
     cut=min(choices)[-1];checks+=1
    assert np.allclose(replayed,z['p'],atol=1e-14,rtol=0) and cut==r.final_threshold
    raw=np.clip(z['source_raw'],1e-10,1-1e-10);odds=raw/(1-raw)*(1-cut)/cut;assert np.allclose(odds/(1+odds),z['source_p'],atol=1e-14,rtol=0)
    for e in events:
     j=e['event'];expected=int(z['y'][j])^int(wrong[j]) if e['type']=='feedback' else int(z['y'][j]);assert e['label']==expected and query[j] and bool(held[j])==e['held'];assert e['t']==arrival[j]+(400 if e['type']=='correction' else 0);checks+=3
    known={};position=0;source=np.asarray(l['source_replay_rows'],int);rows=0
    for f in l['decisions']:
     while position<len(events) and events[position]['t']<=f['t']:
      e=events[position]
      if not e['held']:known[e['event']]=e['label']
      position+=1
     if not f['fit']:continue
     ids=np.asarray(f['verified_training_ids'],int);pi=np.asarray(f['pseudo_training_ids'],int);assert all(j in known and not held[j] for j in ids);assert all(j<=f['t'] and not held[j] and j not in known for j in pi);yy=np.r_[a['y_base'][source],np.asarray([known[j] for j in ids]),(z['p2'][pi]>=.5).astype(int)];pw=np.full(len(pi),.25)
     if pw.sum()>.25*len(ids):pw*=.25*len(ids)/pw.sum()
     w=np.r_[np.ones(len(source)),np.full(len(ids),4.),pw];assert digest(ids,yy,w)==f['training_label_hash'];assert abs(pw.sum()-f['pseudo_mass'])<1e-8 and len(yy)==f['training_rows'];rows+=len(yy);checks+=4
    assert rows==r.training_rows;continue
   cfg=json.loads(r.config);q=z['query'];held=z['held'];arrival=z['arrival'];wrong=z['feedback_wrong'];n=len(z['y']);requests={e['event']:e for e in l['requests']};events=l['events'];assert set(requests)==set(np.flatnonzero(q));assert int(r.queried)==q.sum() and int(r.held_requested)==held.sum();assert not (held&~q).any()
   for start in range(0,n,100):
    end=min(start+100,n);k=min(end-start,max(1,int(round(.05*(end-start)))));ix=np.flatnonzero(q[start:end])+start;assert len(ix)==k and held[start:end].sum()==1
    for j in ix:assert requests[j]['request_t']==end and requests[j]['held']==bool(held[j]);assert arrival[j]==end+(1 if r.scenario=='clean1' else 80);checks+=2
   contractkey=(phase,r.dataset,int(r.seed),r.scenario,cfg['query']);contract=digest(q,held,arrival,wrong,z['p2'])
   if contractkey in contracts:assert contracts[contractkey]==contract,('Matchedquery/inspectorcontract',stem)
   contracts[contractkey]=contract
   for role,keyp in [('ML1','p1'),('ML2','p2')]:c,error=compare(r,z['y'],z[keyp],role+'_');checks+=c;maxerr=max(maxerr,error)
   for e in events:
    j=e['event'];assert j<e['t']<n and q[j] and e['held']==bool(held[j]) and e['origin']=='simulator';expected=int(z['y'][j])^int(wrong[j]) if e['type']=='feedback' else int(z['y'][j]);assert e['label']==expected and e['t']==arrival[j]+(400 if e['type']=='correction' else 0);checks+=3
   assert len(events)==sum(q&(arrival<n))+sum(q&wrong&(arrival+400<n));known={};audit={};position=0;fittedrows=updates=recoveries=0;used_pseudo=[];goodmass=totalmass=0.
   for f in l['fits']:
    while position<len(events) and events[position]['t']<=f['t']:
     e=events[position];(audit if e['held'] else known)[e['event']]=e['label'];position+=1
    ids=np.asarray(f['verified_ids'],int);source=np.asarray(f['source_ids'],int);labels=np.asarray([known[j] for j in ids],int);assert all(q[j] and not held[j] and arrival[j]<=f['t'] for j in ids)
    if f['role']=='ML2':assert ids.tolist()==sorted(known)[-1024:] and f['labels']==labels.tolist();yy=np.r_[a['y_base'][source],labels];w=np.r_[np.ones(len(source)),np.full(len(ids),4.)]
    else:
     assert f['verified_labels']==labels.tolist();pi=np.asarray(f['pseudo_ids'],int);pw=np.asarray(f['pseudo_weights'],float);assert not len(pi) or cfg['pseudo'];assert all(j<=f['t'] and j not in known and not held[j] and f['t']+1-j<=1000 for j in pi);pl=(z['p2'][pi]>=.5).astype(int);assert pl.tolist()==f['pseudo_labels'];assert np.all((z['p1'][pi]>=.5)==pl);assert np.all(np.maximum(z['p1'][pi],1-z['p1'][pi])>=.8) and np.all(np.maximum(z['p2'][pi],1-z['p2'][pi])>=.95);assert pw.sum()<=.25*len(ids)+1e-8 and np.all(pw<=.25+1e-10);yy=np.r_[a['y_base'][source],labels,pl];w=np.r_[np.ones(len(source)),4*np.where(labels==1,cfg['positive_weight'],1.),pw];fittedrows+=len(yy);updates+=1;recoveries+=int(f['recover']);used_pseudo.extend(pi.tolist());totalmass+=float(pw.sum());goodmass+=float(pw[pl==z['y'][pi]].sum())
    assert digest(yy,w)==f['weight_hash'] and len(yy)==f['training_rows'],('fitinputhash',stem,f['t']);checks+=6
   assert fittedrows==r.training_rows and updates==r.updates and recoveries==r.recoveries
   # Independently reconstruct every controller state from available held labels.
   audit={};position=0;cut=.5;alpha=1.
   for start in range(0,n,100):
    end=min(start+100,n);assert np.all(z['threshold'][start:end]==cut) and np.all(z['alpha'][start:end]==alpha)
    while position<len(events) and events[position]['t']<end:
     e=events[position]
     if e['held']:audit[e['event']]=e['label']
     position+=1
    if cfg['route']=='frozen' or len(audit)<20:continue
    ids=np.asarray(sorted(audit)[-128:],int);y=np.asarray([audit[j] for j in ids]);choices=[]
    for aa in ([0.,.25,.5,.75,1.] if cfg['route']=='blend' else [1.]):
     raw=np.where(z['eligible'][ids],(1-aa)*z['p1'][ids]+aa*z['p2'][ids],z['p1'][ids]) if cfg['route']=='blend' else z['raw'][ids]
     for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
      h=raw>=threshold;loss=int((4*((y==1)&~h)+((y==0)&h)).sum());choices.append((loss,abs(threshold-.5),aa,threshold))
    _,_,alpha,cut=min(choices);checks+=1
   if cfg['route'] in ['verified','frozen']:raw=z['p1']
   elif cfg['route']=='max':raw=np.maximum(z['p1'],z['p2'])
   elif cfg['route']=='gate':raw=np.where(z['eligible'],np.maximum(z['p1'],z['p2']),z['p1'])
   else:raw=np.where(z['eligible'],(1-z['alpha'])*z['p1']+z['alpha']*z['p2'],z['p1'])
   assert np.allclose(raw,z['raw'],atol=1e-14,rtol=0);pp=np.clip(raw,1e-10,1-1e-10);odds=pp/(1-pp)*(1-z['threshold'])/z['threshold'];assert np.allclose(odds/(1+odds),z['p'],atol=1e-14,rtol=0);training.append(dict(phase=phase,run_id=r.run_id,policy=r.policy,dataset=r.dataset,seed=r.seed,scenario=r.scenario,unique_pseudo_fitted=len(set(used_pseudo)),pseudo_fit_weighted_precision=goodmass/totalmass if totalmass else None,total_pseudo_objective_mass=totalmass))
   if total%100==0:print('AUDIT',total,phase,flush=True)
  print('AUDITED',phase,len(df),flush=True)
 (ROOT/'checks').mkdir(exist_ok=True);(ROOT/'analysis').mkdir(exist_ok=True);pd.DataFrame(training).to_csv(ROOT/('analysis/actual_pseudo_training_partial.csv' if args.partial else 'analysis/actual_pseudo_training.csv'),index=False);result=dict(passed=True,streams=total,grouped_checks=checks,max_metric_error=maxerr,phases=args.phases,partial=args.partial,scope='Independent arithmetic, label availability, matchedquery/ML2trajectories, classifierlabel/weighthashes, held exclusion, pseudo mass and score-controller reconstruction; not universalcorrectness or populationvalidityproof');(ROOT/('checks/partial_run_audit.json' if args.partial else 'checks/all_run_audit.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
