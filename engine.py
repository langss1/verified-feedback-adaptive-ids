"""Two-role adaptive IDS. This module never accepts stream gold labels.
Only historical labels and validated arrived Feedback records enter learning.
"""
from dataclasses import dataclass,asdict
import copy,hashlib,time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def digest(*values):return hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in values)).hexdigest()
def balanced(y,cap,seed):
 rng=np.random.default_rng(seed);return np.concatenate([rng.choice(np.flatnonzero(y==c),min(int((y==c).sum()),cap//2),replace=False) for c in [0,1]]).astype(int)
def estimator(kind,seed):
 if kind=='hgb':return HistGradientBoostingClassifier(max_iter=64,max_leaf_nodes=15,max_bins=64,l2_regularization=1,early_stopping=False,random_state=seed)
 return ExtraTreesClassifier(n_estimators=64,max_depth=18,min_samples_leaf=2,n_jobs=1,random_state=seed)
class Inspector:
 def fit(self,x,y,weight):
  self.x=np.asarray(x).copy();self.y=np.asarray(y,int).copy();self.weight=np.asarray(weight,float).copy();self.keys=np.asarray([int.from_bytes(hashlib.blake2b(row.tobytes(),digest_size=8).digest(),'little') for row in self.x],np.uint64);self.index=NearestNeighbors(n_neighbors=min(7,len(y)),algorithm='brute',metric='manhattan',n_jobs=1).fit(x);return self
 def inspect(self,x):
  dist,ix=self.index.kneighbors(x);weight=self.weight[ix]/np.maximum(dist,1e-6);p=(weight*self.y[ix]).sum(axis=1)/weight.sum(axis=1);counts=np.zeros((len(x),2),int);near=np.full((len(x),2),np.inf)
  for i in range(len(x)):
   for c in [0,1]:
    keep=(self.weight[ix[i]]>1)&(self.y[ix[i]]==c)
    if keep.any():counts[i,c]=len(np.unique(self.keys[ix[i][keep]]));near[i,c]=float(dist[i][keep].min())
  return p,counts,near,dist[:,0]
@dataclass(frozen=True)
class Config:
 kind:str='hgb'
 route:str='gate' # verified,gate,blend,max,frozen
 pseudo:bool=False
 query:str='mixed'
 interval:int=200
 positive_weight:float=2.
 full_history:bool=False
 legacy_memory:bool=False

def initialize(a,seed,kind):
 rng=np.random.default_rng(seed+101);source=rng.choice(len(a['base']),min(8192,len(a['base'])),replace=False);t=time.perf_counter();m1=estimator(kind,seed).fit(a['base'][source],a['y_base'][source]);initial_s=time.perf_counter()-t
 positions=balanced(a['y_base'][source],1024,seed+27);prototypes=source[positions];m2=Inspector().fit(a['base'][prototypes],a['y_base'][prototypes],np.ones(len(prototypes)));ctx=MiniBatchKMeans(n_clusters=8,n_init=3,batch_size=512,random_state=seed).fit(a['base'][source]);meta=balanced(a['y_meta'],1024,seed+13);cal=balanced(a['y_cal'],1024,seed+17);radius=float(np.quantile(m2.index.kneighbors(a['meta'][meta],n_neighbors=1)[0][:,0],.95));old=balanced(a['y_base'],1024,seed+19)
 return dict(m1=m1,m2=m2,context=ctx,source=source,prototypes=prototypes,old=old,meta=meta,cal=cal,radius=radius,initial_s=initial_s)

class Session:
 def __init__(self,history,seed,config,initial=None):
  self.a=history;self.seed=seed;self.config=config;self.init=initial or initialize(history,seed,config.kind);self.m1=copy.deepcopy(self.init['m1']);self.m2=copy.deepcopy(self.init['m2']);self.previous1=self.previous2=None;self.time=-1;self.rows=[];self.requests={};self.verified={};self.audit={};self.provisional={};self.used=set();self.pending=False;self.events=[];self.fitlog=[];self.progress=[];self.threshold=.5;self.alpha=1.;self.v1=self.v2=0;self.updates=self.recoveries=0;self.train_rows=0;self.fit_s=0.;self.inspector_s=0.;self.controller_s=0.;self.last_inspector_ids=();self.p1_issued=[];self.p2_issued=[];self.route_issued=[];self.issued=[];self.raw_issued=[];self.block_sizes=[]
 def x(self,ids):return np.asarray([self.rows[j] for j in ids])
 def raw(self,p1,p2,eligible,alpha=None):
  cfg=self.config
  if cfg.route in ['verified','frozen']:return p1.copy()
  if cfg.route=='max':return np.maximum(p1,p2)
  if cfg.route=='blend':return np.where(eligible,(1-(self.alpha if alpha is None else alpha))*p1+(self.alpha if alpha is None else alpha)*p2,p1)
  return np.where(eligible,np.maximum(p1,p2),p1)
 def predict_block(self,x,proposal_transform=None):
  x=np.asarray(x);p1=self.m1.predict_proba(x)[:,1];tick=time.perf_counter();p2,counts,near,distance=self.m2.inspect(x);self.inspector_s+=time.perf_counter()-tick
  if proposal_transform is not None:p2=proposal_transform(p2.copy())
  eligible=(p2>=.8)&(counts[:,1]>=2)&(counts[:,0]==0)&(near[:,1]<=self.init['radius']);raw=self.raw(p1,p2,eligible);eps=np.clip(raw,1e-10,1-1e-10);odds=(eps/(1-eps))*((1-self.threshold)/self.threshold);p=odds/(1+odds)
  old1=self.previous1.predict_proba(x)[:,1] if self.previous1 is not None else p1.copy();old2=self.previous2.inspect(x)[0] if self.previous2 is not None else p2.copy()
  if self.previous2 is not None and proposal_transform is not None:old2=proposal_transform(old2)
  self.previous1=self.previous2=None
  return dict(p=p,raw=raw,p1=p1,p2=p2,eligible=eligible,counts=counts,near=near,distance=distance,p1_previous=old1,p2_previous=old2,version1=np.full(len(x),self.v1),version2=np.full(len(x),self.v2),threshold=np.full(len(x),self.threshold),alpha=np.full(len(x),self.alpha))
 def observe(self,x,issued):
  self.time+=1;i=self.time;self.rows.append(np.asarray(x).copy());self.p1_issued.append(float(issued['p1']));self.p2_issued.append(float(issued['p2']));self.route_issued.append(bool(issued['eligible']));self.issued.append(float(issued['p']));self.raw_issued.append(float(issued['raw']));cfg=self.config
  c=int(issued['p2']>=.5);counts=issued['counts'];near=issued['near'];agree=(issued['p1']>=.5)==c
  if cfg.pseudo and agree and max(issued['p1'],1-issued['p1'])>=.8 and max(issued['p2'],1-issued['p2'])>=.95 and counts[c]>=2 and counts[1-c]==0 and near[c]<=self.init['radius']:self.provisional[i]=dict(label=c,version=self.v2)
  return i
 def receive(self,event):
  e=dict(event);j=int(e['event']);t=int(e['t']);label=int(e['label']);assert t==self.time and j<t and label in [0,1];assert j in self.requests and t>=self.requests[j]['request_t'];held=self.requests[j]['held'];assert bool(e['held'])==held;target=self.audit if held else self.verified
  if e['type']=='correction':assert j in target and e['revision']>=1
  else:assert e['type']=='feedback' and j not in target
  old=target.get(j);prior=self.provisional.get(j,{}).get('label',old);target[j]=label
  if j in self.provisional:self.provisional.pop(j)
  if j in self.used and prior!=label:self.pending=True
  e.update(held=held);self.events.append(e)
 def request_block(self,trace):
  end=self.time+1;start=end-len(trace['p']);ids=np.arange(start,end);k=min(len(ids),max(1,int(round(.05*len(ids)))))
  rng=np.random.default_rng(self.seed+441+start*101);uniform=rng.choice(ids,k,replace=False);audit=int(uniform[0]);train=list(map(int,uniform[1:]))
  if self.config.query=='mixed':
   train=train[:2];groups=self.init['context'].predict(self.x(ids));seen={int(groups[j-start]) for j in train};available=[int(j) for j in ids if j not in [audit,*train]]
   uncertainty=1-2*np.abs(trace['p2']-.5)
   for scores in [uncertainty,trace['distance']][:(k-1-len(train))]:
    ranked=sorted(available,key=lambda j:(int(groups[j-start]) in seen,-float(scores[j-start]),j));chosen=ranked[0];train.append(chosen);available.remove(chosen);seen.add(int(groups[chosen-start]))
  for j in [audit,*train]:
   reason='uniform_audit' if j==audit else 'uniform_training' if j in uniform[1:3] or self.config.query=='random' else 'inspector_uncertainty_or_novelty'
   self.requests[j]=dict(event=j,request_t=end,held=j==audit,reason=reason);self.provisional.pop(j,None) if j==audit else None
  assert len(train)==k-1 and len(set([audit,*train]))==k
  return [self.requests[j] for j in [audit,*train]]
 def update_controller(self):
  if self.config.route=='frozen' or len(self.audit)<20:return
  tick=time.perf_counter();ids=np.asarray(sorted(self.audit)[-128:],int);y=np.asarray([self.audit[j] for j in ids]);p1=np.asarray(self.p1_issued)[ids];p2=np.asarray(self.p2_issued)[ids];eligible=np.asarray(self.route_issued)[ids];choices=[]
  for alpha in ([0.,.25,.5,.75,1.] if self.config.route=='blend' else [1.]):
   # Other policies tune threshold on their originally issued raw score.
   raw=self.raw(p1,p2,eligible,alpha) if self.config.route=='blend' else np.asarray(self.raw_issued)[ids]
   for cut in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    h=raw>=cut;loss=int((4*((y==1)&~h)+((y==0)&h)).sum());choices.append((loss,abs(cut-.5),alpha,cut))
  _,_,self.alpha,self.threshold=min(choices);self.controller_s+=time.perf_counter()-tick
 def update(self):
  end=self.time+1;cfg=self.config;self.provisional={j:v for j,v in self.provisional.items() if end-j<=1000 and j not in self.verified and not self.requests.get(j,{}).get('held',False)};ids=np.asarray(sorted(self.verified)[-1024:],int);labels=np.asarray([self.verified[j] for j in ids],int);labelhash=digest(ids,labels);inspector_fit=False
  if len(ids) and labelhash!=getattr(self,'last_inspector_hash',None):
   source=self.init['prototypes'];xx=np.vstack([self.a['base'][source],self.x(ids)]);yy=np.r_[self.a['y_base'][source],labels];weight=np.r_[np.ones(len(source)),np.full(len(ids),4.)];self.previous2=self.m2;tick=time.perf_counter();self.m2=Inspector().fit(xx,yy,weight);self.inspector_s+=time.perf_counter()-tick;self.v2+=1;self.last_inspector_hash=labelhash;inspector_fit=True
   self.fitlog.append(dict(role='ML2',t=self.time,version=self.v2,source_ids=source.tolist(),verified_ids=ids.tolist(),labels=labels.tolist(),weight_hash=digest(yy,weight),training_rows=len(yy)))
  dofit=cfg.route!='frozen' and len(ids)>0 and (end%cfg.interval==0 or self.pending);recover=bool(self.pending and dofit)
  if dofit:
   if cfg.legacy_memory:
    last=np.asarray(sorted(self.verified)[-512:],int);positions=balanced(np.asarray([self.verified[j] for j in last]),256,self.seed+end);ids=last[positions];labels=np.asarray([self.verified[j] for j in ids],int)
   source=self.init['source'] if cfg.full_history else self.init['old'];xx=[self.a['base'][source],self.x(ids)];yy=[self.a['y_base'][source],labels];vw=np.full(len(ids),4.)*np.where(labels==1,cfg.positive_weight,1.);weights=[np.ones(len(source)),vw];pi=np.asarray(sorted(self.provisional)[-256:],int);pw=np.zeros(0)
   if cfg.pseudo and len(pi):
    pw=np.full(len(pi),.25);cap=.25*len(ids)
    if pw.sum()>cap:pw*=cap/pw.sum()
    xx.append(self.x(pi));yy.append(np.asarray([self.provisional[j]['label'] for j in pi],int));weights.append(pw)
   else:pi=np.asarray([],int)
   xx=np.vstack(xx);yy=np.concatenate(yy);ww=np.concatenate(weights);assert all(not self.requests[j]['held'] for j in ids);assert all(not self.requests.get(j,{}).get('held',False) for j in pi);assert pw.sum()<=.25*len(ids)+1e-9;self.previous1=self.m1;tick=time.perf_counter();self.m1=estimator(cfg.kind,self.seed).fit(xx,yy,sample_weight=ww);seconds=time.perf_counter()-tick;self.fit_s+=seconds;self.v1+=1;self.updates+=1;self.recoveries+=int(recover);self.train_rows+=len(yy);self.used=set(ids.tolist()+pi.tolist());self.pending=False
   self.fitlog.append(dict(role='ML1',t=self.time,version=self.v1,source_ids=source.tolist(),verified_ids=ids.tolist(),verified_labels=labels.tolist(),pseudo_ids=pi.tolist(),pseudo_labels=[self.provisional[j]['label'] for j in pi],pseudo_weights=pw.tolist(),weight_hash=digest(yy,ww),training_rows=len(yy),recover=recover,fit_s=seconds))
  self.update_controller();self.progress.append(dict(t=self.time,labels_requested=len(self.requests),training_arrived=len(self.verified),audit_arrived=len(self.audit),corrections=sum(e['type']=='correction' for e in self.events),version1=self.v1,version2=self.v2,fit=bool(dofit),recover=recover,training_rows=self.train_rows,threshold=self.threshold,alpha=self.alpha))
 def source_predictions(self):
  ix=self.init['cal'];x=self.a['cal'][ix];p1=self.m1.predict_proba(x)[:,1];p2,counts,near,_=self.m2.inspect(x);eligible=(p2>=.8)&(counts[:,1]>=2)&(counts[:,0]==0)&(near[:,1]<=self.init['radius']);raw=self.raw(p1,p2,eligible);pp=np.clip(raw,1e-10,1-1e-10);odds=pp/(1-pp)*(1-self.threshold)/self.threshold;return dict(y=self.a['y_cal'][ix],p1=p1,p2=p2,p=odds/(1+odds),indices=ix)
