"""Read-only D4 routing reconstruction and label-only native ARF adapter."""
from pathlib import Path
import sys,copy,pickle,time,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;D4=ROOT.parent/'adaptive_feedback_d4_20261126'
sys.path[:0]=[str(ROOT/'vendor'),str(D4)]
from river import forest
from engine import digest
from experiment import metrics

def shifted(raw,cut):
 p=np.clip(np.asarray(raw,float),1e-10,1-1e-10);odds=(p/(1-p))*((1-cut)/cut);return odds/(1+odds)

def tune(audit,issued):
 ids=np.asarray(sorted(audit)[-128:],int);y=np.asarray([audit[j] for j in ids]);raw=np.asarray(issued)[ids]
 return min((int((4*((y==1)&(raw<c))+((y==0)&(raw>=c))).sum()),abs(c-.5),1.,c) for c in [.1,.2,.3,.4,.5,.6,.7,.8,.9])[-1]

def event_hash(log):
 return hashlib.sha256(json.dumps(dict(requests=log['requests'],events=log['events']),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def derive_route(tr,log,route,frozen_raw=None,frozen_source=None):
 assert route in ['verified','max','frozen'];n=len(tr['p']);cut=.5;prob=np.zeros(n);cuts=np.zeros(n);audit={};events=sorted(log['events'],key=lambda e:e['t']);cursor=0
 raw=np.asarray(frozen_raw if route=='frozen' else tr['p1'] if route=='verified' else np.maximum(tr['p1'],tr['p2']));source_raw=np.asarray(frozen_source if route=='frozen' else tr['source_p1'] if route=='verified' else np.maximum(tr['source_p1'],tr['source_p2']))
 for start in range(0,n,100):
  end=min(start+100,n);prob[start:end]=shifted(raw[start:end],cut);cuts[start:end]=cut
  while cursor<len(events) and events[cursor]['t']<end:
   e=events[cursor];assert e['t']>=start
   if e['held']:audit[e['event']]=e['label']
   cursor+=1
  if route!='frozen' and len(audit)>=20:cut=tune(audit,raw[:end])
 out={k:v.copy() for k,v in tr.items()};out.update(p=prob,raw=raw,threshold=cuts,source_p=shifted(source_raw,cut))
 if route=='frozen':out.update(p1=raw,source_p1=source_raw,version1=np.zeros(n,int))
 return out,cut

def new_arf(seed):return forest.ARFClassifier(seed=seed)
def dictionaries(x):return [dict(enumerate(map(float,row))) for row in x]
def probabilities(model,rows):
 p=np.asarray([model.predict_proba_one(x).get(1,0.) for x in rows],float);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();return p

class ARFAdapter:
 """Consumes only arrived label events and features; never target truth arrays."""
 def __init__(self,initial_blob,rows):
  self.initial_blob=initial_blob;self.model=pickle.loads(initial_blob);self.rows=rows;self.verified={};self.audit={};self.first_order=[];self.used={};self.cut=.5;self.raw=[];self.fits=[];self.resets=0;self.training_work=0;self.fit_s=0.;self.version=0
 def receive_and_update(self,events,end):
  tick=time.perf_counter()
  for e in events:
   j=int(e['event']);label=int(e['label']);assert j<e['t']<end and label in [0,1]
   target=self.audit if e['held'] else self.verified
   if e['type']=='feedback':assert j not in target
   else:assert e['type']=='correction' and j in target
   if not e['held'] and j not in self.verified:self.first_order.append(j)
   target[j]=label
  changed=any(self.verified[j]!=v for j,v in self.used.items())
  ids=self.first_order if changed else [j for j in self.first_order if j not in self.used]
  if changed:self.model=pickle.loads(self.initial_blob);self.used={};self.resets+=1
  for j in ids:self.model.learn_one(self.rows[j],self.verified[j]);self.used[j]=self.verified[j]
  if ids:
   self.version+=1;self.training_work+=len(ids)
   self.fits.append(dict(t=end-1,version=self.version,reset=changed,trained_ids=list(ids),labels=[self.verified[j] for j in ids],current_label_hash=digest(np.asarray(self.first_order,int),np.asarray([self.verified[j] for j in self.first_order],int))))
  if len(self.audit)>=20:self.cut=tune(self.audit,self.raw)
  self.fit_s+=time.perf_counter()-tick

def run_arf(x,initial_blob,events,source_x):
 """Model function has no y_test, evaluation truth is handled by the caller."""
 rows=dictionaries(x);source_rows=dictionaries(source_x);core=ARFAdapter(initial_blob,rows);initial_source=probabilities(core.model,source_rows);n=len(x);p=np.zeros(n);raw=np.zeros(n);cuts=np.zeros(n);versions=np.zeros(n,int);ordered=sorted(events,key=lambda e:e['t']);cursor=0;progress=[];checkpoints=[];starttime=time.perf_counter()
 for start in range(0,n,100):
  end=min(start+100,n);raw[start:end]=probabilities(core.model,rows[start:end]);p[start:end]=shifted(raw[start:end],core.cut);cuts[start:end]=core.cut;versions[start:end]=core.version;core.raw.extend(raw[start:end].tolist());arrived=[]
  while cursor<len(ordered) and ordered[cursor]['t']<end:
   e=ordered[cursor];assert e['t']>=start;arrived.append(e);cursor+=1
  core.receive_and_update(arrived,end)
  progress.append(dict(t=end-1,training_arrived=len(core.verified),audit_arrived=len(core.audit),threshold=core.cut,version=core.version,resets=core.resets,learned_events=core.training_work))
  if end%1000==0 or end==n:
   sp=probabilities(core.model,source_rows);checkpoints.append(dict(t=end-1,raw=sp.tolist(),p=shifted(sp,core.cut).tolist()))
 source_raw=probabilities(core.model,source_rows)
 return dict(p=p,raw=raw,threshold=cuts,version1=versions,source_p=shifted(source_raw,core.cut),source_p1=source_raw,initial_source_p1=initial_source),dict(fits=core.fits,progress=progress,source_checkpoints=checkpoints,resets=core.resets,training_work=core.training_work,fit_s=core.fit_s,online_s=time.perf_counter()-starttime,drifts_in_final_state=core.model.n_drifts_detected(),warnings_in_final_state=core.model.n_warnings_detected(),scope='Native ARF, shared acquisition; one event/label; reset/replay on changed learned labels. No audit training or target gold input.')
