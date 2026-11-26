"""100%perfectpastlabels at the SAME200-event regular cadence as D4.
Descriptive full-supervision reference, never a mathematicallyoptimal upperbound.
"""
import copy,time
import numpy as np
from engine import estimator,digest
from experiment import metrics
def run(a,seed,kind,positive_weight,init):
 source=init['source'];model=copy.deepcopy(init['m1']);n=len(a['test']);p=np.zeros(n);fits=[];t0=time.perf_counter();rows=0
 for start in range(0,n,200):
  end=min(start+200,n);p[start:end]=model.predict_proba(a['test'][start:end])[:,1];ids=np.arange(end);xx=np.vstack([a['base'][source],a['test'][ids]]);yy=np.r_[a['y_base'][source],a['y_test'][ids]];w=np.r_[np.ones(len(source)),4*np.where(a['y_test'][ids]==1,positive_weight,1.)];model=estimator(kind,seed).fit(xx,yy,sample_weight=w);fits.append(dict(t=end-1,observed_training_ids=ids.tolist(),weight_hash=digest(yy,w),training_rows=len(yy)));rows+=len(yy)
 cal=init['cal'];sp=model.predict_proba(a['cal'][cal])[:,1];r=metrics(a['y_test'],p);r.update(seed=seed,scenario='clean1',policy='oracle_100pct',kind=kind,positive_weight=positive_weight,feedback_origin='perfect_past_gold_reference',queried=n,held_requested=0,arrived_train=n,arrived_audit=0,human_origin_received=0,updates=len(fits),training_rows=rows,online_s=time.perf_counter()-t0,source_BA=metrics(a['y_cal'][cal],sp)['balanced_accuracy'],source1_BA=metrics(a['y_cal'][cal],sp)['balanced_accuracy'],initial_source1_BA=metrics(a['y_cal'][cal],init['m1'].predict_proba(a['cal'][cal])[:,1])['balanced_accuracy'],prediction_sha256=digest(p),source_prediction_sha256=digest(sp));tr=dict(y=a['y_test'],p=p,source_y=a['y_cal'][cal],source_p=sp,source_indices=cal)
 for key in ['test_family','test_raw_rows','test_timestamps']:
  if key in a:tr[key]=a[key]
 return r,tr,dict(fits=fits,source_ids=source.tolist(),kind=kind,positive_weight=positive_weight,scope='100percentperfectpastlabels,200eventrefits,sameestimatorfamily; differentlabelbudget; nofuturelabels; notmathematicallyoptimalreference')
