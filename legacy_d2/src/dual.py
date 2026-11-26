"""DV-CAR experimental integration: version/context evidence, bounded mass,
and correction-triggered reconstruction. Existing sklearn learners, not new trees.
"""
from pathlib import Path
import sys,json,time,hashlib,math,copy
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import rankdata
from memory_verifier import MemoryVerifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'inputs/d1/src'))
from model import metrics

SEEDS=[11,23,37,51,79]
SCENARIOS=['clean1','delay80','feedback_corrected','checker_fault']
SPECS={f'dual_E{e}_B{b}_R{r}':dict(E=e,B=b,R=r) for e in [0,1] for b in [0,1] for r in [0,1]}
BASELINES=['frozen_ml1','verified_only','direct_ml2','simple_average']

def hgb(seed):return HistGradientBoostingClassifier(max_iter=64,max_leaf_nodes=15,max_bins=64,l2_regularization=1.,early_stopping=False,random_state=seed)
def verifier(seed,kind='extra'):
 if kind=='knn':return MemoryVerifier()
 if kind=='hgb':return hgb(seed)
 return ExtraTreesClassifier(n_estimators=64,max_depth=16,min_samples_leaf=3,n_jobs=1,random_state=seed)
def balanced(ids,labels,cap,seed):
 ids=np.asarray(ids,int)
 if not len(ids):return ids
 yy=np.asarray(labels,int);rng=np.random.default_rng(seed);parts=[]
 for c in [0,1]:
  v=ids[yy==c];parts.append(rng.choice(v,min(len(v),cap//2),replace=False))
 return np.concatenate(parts).astype(int)
def initial(a,seed,teacher_kind='extra'):
 rng=np.random.default_rng(seed+101);ix=rng.choice(len(a['base']),min(8192,len(a['base'])),replace=False)
 t=time.perf_counter();m1=hgb(seed).fit(a['base'][ix],a['y_base'][ix]);s1=time.perf_counter()-t
 tix=balanced(ix,a['y_base'][ix],1024,seed+27) if teacher_kind=='knn' else ix
 t=time.perf_counter();m2=verifier(seed,teacher_kind).fit(a['base'][tix],a['y_base'][tix]);s2=time.perf_counter()-t
 ctx=MiniBatchKMeans(n_clusters=8,n_init=3,batch_size=512,random_state=seed).fit(a['base'][ix])
 meta=balanced(np.arange(len(a['meta'])),a['y_meta'],1024,seed+13);cal=balanced(np.arange(len(a['cal'])),a['y_cal'],1024,seed+17)
 old=balanced(np.arange(len(a['base'])),a['y_base'],1024,seed+19)
 radius=float(np.quantile(m2.index.kneighbors(a['meta'][meta],n_neighbors=1)[0][:,0],.95)) if teacher_kind=='knn' else 0.
 return dict(m1=m1,m2=m2,fit1_s=s1,fit2_s=s2,context=ctx,meta=meta,cal=cal,old=old,initial_rows=ix,teacher_rows=tix,teacher_kind=teacher_kind,local_radius=radius)

def schedule(n,seed,scenario,query_mode='random',planner=None):
 rng=np.random.default_rng(seed+441);query=np.zeros(n,bool);held=np.zeros(n,bool);arrivals=np.full(n,n+1000,int)
 for start in range(0,n,100):
  end=min(start+100,n);ids=np.arange(start,end);k=min(5,len(ids));chosen=rng.choice(ids,k,replace=False);audit=int(chosen[0])
  train=chosen[1:]
  if query_mode=='strategic' and k==5:
   random_train=int(chosen[1]);available=[j for j in ids if j not in [audit,random_train]];rank=sorted(available,key=lambda j:(-planner['score'][j],j));pick=[];groups_seen=set()
   for j in rank:
    g=int(planner['groups'][j])
    if g not in groups_seen:pick.append(j);groups_seen.add(g)
    if len(pick)==3:break
   if len(pick)<3:pick += [j for j in rank if j not in pick][:3-len(pick)]
   train=np.asarray([random_train,*pick])
  selection=np.r_[audit,train].astype(int);query[selection]=True;held[audit]=True;arrivals[selection]=end+(1 if scenario=='clean1' else 80)
 # Fault assignment is independent of query/audit assignment and of gold values.
 wrong=rng.random(n)<.1 if scenario=='feedback_corrected' else np.zeros(n,bool)
 checker_draw=rng.random(n)<.25;delivery={};correction={}
 for j in np.flatnonzero(query):
  delivery.setdefault(int(arrivals[j]),[]).append(int(j))
  if wrong[j]:correction.setdefault(int(arrivals[j]+400),[]).append(int(j))
 return dict(query=query,held=held,arrival=arrivals,wrong=wrong,checker_draw=checker_draw,delivery=delivery,correction=correction,query_mode=query_mode)

def teacher_trace(a,stream,seed,scenario,init,query_mode='random'):
 """A shared ML2 trajectory depends only on fixed verified-label schedule.
 Cached once per case; its recorded compute cost is charged to every dual run.
 """
 trace_start=time.perf_counter();planner_start=time.perf_counter();x=a['test'][stream];gold=a['y_test'][stream];n=len(x);planner=None
 if query_mode=='strategic':
  g=init['context'].predict(x);dist=np.linalg.norm(x-init['context'].cluster_centers_[g],axis=1);pa=init['m1'].predict_proba(x)[:,1];pb=init['m2'].predict_proba(x)[:,1];raw=[dist,np.abs(pa-pb),1-2*np.abs(.5-.5*(pa+pb))];score=np.zeros(n)
  for start in range(0,n,100):
   end=min(start+100,n)
   for weight,v in zip([.45,.35,.20],raw):
    rank=(rankdata(v[start:end],method='average')-1)/max(1,end-start-1);score[start:end]+=weight*rank
  planner=dict(groups=g,score=score)
 planner_s=time.perf_counter()-planner_start;sc=schedule(n,seed,scenario,query_mode,planner);model=copy.deepcopy(init['m2']);version=0;known={};audit={};p=np.zeros(n);ver=np.zeros(n,int);fault=np.zeros(n,bool);local_counts=np.zeros((n,2),int);local_near=np.full((n,2),np.inf);cost=0.;fits=0;train_rows=0;refresh={};models={0:copy.deepcopy(model)};meta=a['meta'][init['meta']];meta_groups=init['context'].predict(meta);priors={};prior_s=0.;fit_log=[]
 for i in range(0,n,100):
  if version not in priors:
   tick=time.perf_counter();priors[version]=trust_prior(model.predict_proba(meta)[:,1],a['y_meta'][init['meta']],meta_groups);prior_s+=time.perf_counter()-tick
  end=min(i+100,n);tick=time.perf_counter();p[i:end]=model.predict_proba(x[i:end])[:,1];cost+=time.perf_counter()-tick;ver[i:end]=version
  if init['teacher_kind']=='knn':local_counts[i:end],local_near[i:end],_=model.evidence(x[i:end])
  if scenario=='checker_fault':
   # Evaluator-controlled false-benign fault; learner receives only modified p.
   mask=(gold[i:end]==1)&sc['checker_draw'][i:end];fault[i:end]=mask;p[i:end][mask]=.001
  for t in range(i,end):
   for j in sc['delivery'].get(t,[]):
    yy=int(gold[j])^int(sc['wrong'][j]);(audit if sc['held'][j] else known)[j]=yy
   for j in sc['correction'].get(t,[]):(audit if sc['held'][j] else known)[j]=int(gold[j])
  if end%(100 if init['teacher_kind']=='knn' else 2000)==0 and known and end<n:
   ids=np.asarray(sorted(known)[-512:],int);ti=init['teacher_rows'];xx=np.vstack([a['base'][ti],x[ids]]);yy=np.r_[a['y_base'][ti],[known[j] for j in ids]];kind=init['teacher_kind'];ww=np.r_[np.ones(len(ti)),np.full(len(ids),16. if kind=='hgb' else 4.)]
   tick=time.perf_counter();model=verifier(seed,kind).fit(xx,yy,sample_weight=ww);cost+=time.perf_counter()-tick;fits+=1;train_rows+=len(yy);version+=1;models[version]=copy.deepcopy(model);fit_log.append(dict(t=end-1,version=version,source_rows=ti.tolist(),training_events=ids.tolist(),labels=[int(known[j]) for j in ids],label_weight_sha256=hashlib.sha256(yy.tobytes()+ww.tobytes()).hexdigest()))
  if end<n:refresh[end]=version
 return dict(p=p,version=ver,fault=fault,local_counts=local_counts,local_near=local_near,priors=priors,cost_s=time.perf_counter()-trace_start,planner_s=planner_s,prior_s=prior_s,fits=fits,train_rows=train_rows,models=models,schedule=sc,fit_log=fit_log)

def trust_prior(prob,y,groups):
 out=np.zeros((8,2,2),float);h=prob>=.5
 for g in range(8):
  for c in [0,1]:
   ix=(groups==g)&(h==c);n=int(ix.sum());scale=min(n,16)/max(n,1);correct=float((h[ix]==y[ix]).sum())
   out[g,c]=[correct*scale,n*scale]
 return out
def lower(score):
 good,n=score
 if n<8:return 0.
 phat=good/n;z=1.2815515655446004
 return float((phat+z*z/(2*n)-z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n)))/(1+z*z/n))

def run(a,stream,probe,seed,scenario,policy,init,teacher):
 sp=SPECS.get(policy,dict(E=0,B=0,R=0));isdual=policy in SPECS
 x=a['test'][stream];gold=a['y_test'][stream];n=len(x);sc=teacher['schedule'];student=copy.deepcopy(init['m1']);meta=a['meta'][init['meta']];my=a['y_meta'][init['meta']];source=a['cal'][init['cal']];sy=a['y_cal'][init['cal']]
 groups=init['context'].predict(x);mg=init['context'].predict(meta)
 verified={};audit={};pseudo={};quarantine=set();used=set();records=[];pred=np.zeros(n);p1all=np.zeros(n);p2=teacher['p'];vp=teacher['version'];pending_recovery=False;last_teacher_version=-1;student_version=0
 trust1=trust_prior(student.predict_proba(meta)[:,1],my,mg);base_trust1=trust1.copy();trust2=None;ledger=[];update_count=recovery_count=0;train_rows=0;train_s=0.;admitted=correct_admitted=0;revoked=0;override=0
 initial_source=float(metrics(sy,student.predict_proba(source)[:,1])['balanced_accuracy']);tick_total=time.perf_counter()
 for start in range(0,n,100):
  end=min(start+100,n);version=int(vp[start])
  if version!=last_teacher_version:
   trust2=teacher['priors'][version].copy();last_teacher_version=version
  # Refresh current-student audit only from arrived, held labels.
  trust1=base_trust1.copy()
  if audit:
   ids=np.asarray(sorted(audit)[-128:],int);hh=student.predict_proba(x[ids])[:,1]>=.5
   for j,h in zip(ids,hh):trust1[groups[j],int(h)]+=[float(h==audit[j]),1.]
  ps=student.predict_proba(x[start:end])[:,1];p1all[start:end]=ps
  for i in range(start,end):
   p1=float(ps[i-start]);pteacher=float(p2[i]);c=int(pteacher>=.5);g=int(groups[i]);confidence=max(pteacher,1-pteacher);l2=lower(trust2[g,c]);l1=lower(trust1[g,int(p1>=.5)])
   fresh=(teacher['local_counts'][i,c]>=3 and teacher['local_counts'][i,1-c]==0 and teacher['local_near'][i,c]<=init['local_radius'])
   conflict=teacher['local_counts'][i,1-c]>=2
   supported=((l2>=.75 or fresh) and not conflict and (version,g,c) not in quarantine)
   confident=confidence>=.9
   # E governs class/context evidence for BOTH inference correction and admission.
   route=isdual and confident and ((supported and (fresh or l2>=l1+.02)) if sp['E'] else True) and ((p1>=.5)!=(pteacher>=.5))
   if policy=='direct_ml2':pred[i]=pteacher
   elif policy=='simple_average':pred[i]=.5*(p1+pteacher)
   else:pred[i]=pteacher if route else p1
   override+=int(route)
   # Role assignment is fixed before labels; held audit examples never pseudo-train.
   if isdual and confident and (supported if sp['E'] else True):
    pseudo[i]=dict(label=c,version=version,group=g,confidence=confidence);admitted+=1;correct_admitted+=int(c==gold[i])
   for j in sc['delivery'].get(i,[]):
    assert j<i and sc['arrival'][j]<=i;yy=int(gold[j])^int(sc['wrong'][j])
    if sc['held'][j]:
     audit[j]=yy
     if int(vp[j])==version:trust2[groups[j],int(p2[j]>=.5)]+=[float(int(p2[j]>=.5)==yy),1.]
    else:
     verified[j]=yy
     if j in pseudo:
      old=pseudo.pop(j)
      if old['label']!=yy:
       if sp['R']:
        cohort=(old['version'],old['group'],old['label']);quarantine.add(cohort);kill=[k for k,v in pseudo.items() if (v['version'],v['group'],v['label'])==cohort]
        for k in kill:pseudo.pop(k);revoked+=1
        revoked+=1;pending_recovery |= j in used or bool(set(kill)&used)
    records.append(dict(t=i,event=j,type='feedback',label=yy,held=bool(sc['held'][j])))
   for j in sc['correction'].get(i,[]):
    target=audit if sc['held'][j] else verified;old=target.get(j);target[j]=int(gold[j])
    if sc['held'][j] and int(vp[j])==version and old is not None:
     h=int(p2[j]>=.5);trust2[groups[j],h,0]+=float(h==int(gold[j]))-float(h==old)
    if old!=target[j] and j in used and sp['R']:pending_recovery=True
    records.append(dict(t=i,event=j,type='correction',old_label=old,label=int(gold[j]),held=bool(sc['held'][j])))
  # Persistent data/label ledger; scheduled refit, not sklearn partial_fit.
  pseudo={j:v for j,v in pseudo.items() if end-j<=1000 and j not in verified and not(sc['query'][j] and sc['held'][j])}
  regular=end%500==0 or end==n;recover=bool(pending_recovery and sp['R'])
  dofit=policy!='frozen_ml1' and policy!='direct_ml2' and bool(verified) and (regular or recover)
  steps=0;mass=0.;pseudo_ids=[];datahash=None
  if dofit:
   ids=np.asarray(sorted(verified)[-512:],int);labs=np.asarray([verified[j] for j in ids]);vi=balanced(ids,labs,256,seed+end)
   old=init['old'];xx=[a['base'][old],x[vi]];yy=[a['y_base'][old],np.asarray([verified[j] for j in vi])];weights=[np.ones(len(old)),np.full(len(vi),4.)]
   if isdual and pseudo:
    pi=np.asarray(sorted(pseudo)[-512:],int);pw=np.full(len(pi),.25 if sp['B'] else 1.)
    if sp['B']:
     cap=.2/(1-.2)*len(vi)
     if pw.sum()>cap:pw*=cap/max(pw.sum(),1e-12)
    if pw.sum()>0:
     pseudo_ids=pi.tolist();mass=float(pw.sum());xx.append(x[pi]);yy.append(np.asarray([pseudo[j]['label'] for j in pi]));weights.append(pw)
   xx=np.vstack(xx);yy=np.concatenate(yy);ww=np.concatenate(weights);assert all(sc['query'][j] and not sc['held'][j] and sc['arrival'][j]<end for j in vi)
   assert all(not(sc['query'][j] and sc['held'][j]) for j in pseudo_ids)
   if sp['B']:assert mass<=.25*len(vi)+1e-9
   datahash=hashlib.sha256(np.asarray(vi).tobytes()+yy.tobytes()+ww.tobytes()).hexdigest();tick=time.perf_counter();student=hgb(seed).fit(xx,yy,sample_weight=ww);train_s+=time.perf_counter()-tick;train_rows+=len(yy);steps=len(yy)*64;update_count+=1;student_version+=1;used=set(vi.tolist()+pseudo_ids);recovery_count+=int(recover);pending_recovery=False
   base_trust1=trust_prior(student.predict_proba(meta)[:,1],my,mg)
  ledger.append(dict(t=end-1,student_version=student_version,verifier_version=version,fit=bool(dofit),recovery=recover and dofit,training_rows=steps//64,tree_iteration_rows=steps,verified_current=len(verified),audit_current=len(audit),pseudo_current=len(pseudo),pseudo_mass=mass,pseudo_training_ids=pseudo_ids,verified_training_ids=[] if not dofit else vi.tolist(),training_label_hash=datahash,quarantined_cohorts=len(quarantine)))
 elapsed=time.perf_counter()-tick_total
 # Gold appears only in evaluation and fault/feedback simulator, not admission.
 trust1=base_trust1.copy()
 if audit:
  final_audit=np.asarray(sorted(audit)[-128:],int);final_h=student.predict_proba(x[final_audit])[:,1]>=.5
  for j,h in zip(final_audit,final_h):trust1[groups[j],int(h)]+=[float(h==audit[j]),1.]
 source1=student.predict_proba(source)[:,1];source2=teacher['models'][int(vp[-1])].predict_proba(source)[:,1];sourcepred=source1.copy()
 if policy=='direct_ml2':sourcepred=source2
 elif policy=='simple_average':sourcepred=.5*(source1+source2)
 elif isdual:
  sg=init['context'].predict(source)
  if init['teacher_kind']=='knn':scount,snear,_=teacher['models'][int(vp[-1])].evidence(source)
  else:scount=np.zeros((len(source),2),int);snear=np.full((len(source),2),np.inf)
  for j,(s1,s2,g) in enumerate(zip(source1,source2,sg)):
   c=int(s2>=.5);l2=lower(trust2[g,c]);l1=lower(trust1[g,int(s1>=.5)]);fresh=scount[j,c]>=3 and scount[j,1-c]==0 and snear[j,c]<=init['local_radius'];conflict=scount[j,1-c]>=2;supported=(l2>=.75 or fresh) and not conflict and (int(vp[-1]),int(g),c) not in quarantine
   if max(s2,1-s2)>=.9 and ((supported and (fresh or l2>=l1+.02)) if sp['E'] else True) and ((s1>=.5)!=(s2>=.5)):sourcepred[j]=s2
 r=metrics(gold,pred);r.update(policy=policy,seed=seed,scenario=scenario,source_BA=metrics(sy,sourcepred)['balanced_accuracy'],source_student_BA=metrics(sy,source1)['balanced_accuracy'],initial_source_BA=initial_source,updates=update_count,recoveries=recovery_count,training_rows=train_rows,student_train_s=train_s,online_loop_s=elapsed,verifier_compute_s=teacher['cost_s'] if policy!='frozen_ml1' and policy!='verified_only' else teacher['planner_s'] if policy=='verified_only' and sc['query_mode']=='strategic' else 0.,verifier_refit_rows=teacher['train_rows'] if policy!='frozen_ml1' and policy!='verified_only' else 0,admitted=admitted,admission_precision=correct_admitted/admitted if admitted else None,revoked=revoked,overrides=override,queried=int(sc['query'].sum()),arrived_train=len(verified),arrived_audit=len(audit),prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),faults=int(teacher['fault'].sum()))
 r['source_prediction_sha256']=hashlib.sha256(np.asarray(sourcepred).tobytes()).hexdigest()
 return r,ledger,records,dict(y=gold,p=pred,p1=p1all,p2=p2,groups=groups,verifier_version=vp,query=sc['query'],held=sc['held'],arrival=sc['arrival'],feedback_wrong=sc['wrong'],checker_fault=teacher['fault'],local_counts=teacher['local_counts'],local_near=teacher['local_near'],source_index=np.asarray(stream),source_cal_indices=init['cal'],source_y=sy,source_p=sourcepred)
