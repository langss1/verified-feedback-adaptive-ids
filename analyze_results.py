from pathlib import Path
import json,gzip
import numpy as np
import pandas as pd
from scipy.stats import t
from audit_results import metrics
ROOT=Path(__file__).resolve().parent
FIELDS=['balanced_accuracy','fnr','fpr','attack_precision','attack_F1','AP','brier','security_loss','source_BA','ML1_balanced_accuracy','ML2_balanced_accuracy','ML1_attack_precision','ML2_attack_precision','training_rows','fit_s','online_s','queried','arrived_train','arrived_audit']
def interval(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if not len(v):return dict(mean=None,lo=None,hi=None,n=0)
 m=float(v.mean());width=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else float('nan');return dict(mean=m,lo=m-width,hi=m+width,n=len(v))
def main():
 lock=json.loads((ROOT/'protocol/confirmation_lock.json').read_text());df=pd.read_csv(ROOT/'results/confirmation/runs.csv');assert len(df)==lock['expected_runs'];out=ROOT/'analysis';out.mkdir(exist_ok=True)
 for prefix in ['','ML1_','ML2_']:
  df[prefix+'attack_precision']=df[prefix+'tp']/(df[prefix+'tp']+df[prefix+'fp']);df[prefix+'attack_F1']=2*df[prefix+'tp']/(2*df[prefix+'tp']+df[prefix+'fp']+df[prefix+'fn'])
 df.to_csv(out/'confirmation_all_runs.csv',index=False);seed=df.groupby(['dataset','seed','policy'])[FIELDS].mean().reset_index();seed.to_csv(out/'confirmation_seed_means.csv',index=False);summary=[];paired=[]
 for (ds,sc,policy),g in df.groupby(['dataset','scenario','policy']):
  for m in FIELDS:
   if m in g and g[m].notna().any():summary.append(dict(dataset=ds,scenario=sc,policy=policy,metric=m,**interval(g[m])))
 pd.DataFrame(summary).to_csv(out/'confirmation_summary.csv',index=False)
 for ds in lock['cohorts']:
  for sc in ['ALL','CLEAN',*df[df.dataset==ds].scenario.unique()]:
   z=seed[seed.dataset==ds] if sc=='ALL' else df[(df.dataset==ds)&df.scenario.isin(['clean1','delay80'])].groupby(['seed','policy'])[FIELDS].mean().reset_index() if sc=='CLEAN' else df[(df.dataset==ds)&(df.scenario==sc)]
   candidate=z[z.policy=='new_selected'].set_index('seed')
   if not len(candidate):continue
   for comparator,g in z.groupby('policy'):
    g=g.set_index('seed');shared=candidate.index.intersection(g.index)
    for m in FIELDS:paired.append(dict(dataset=ds,scenario=sc,policy='new_selected',comparator=comparator,metric=m,**interval(candidate.loc[shared,m]-g.loc[shared,m])))
 paired=pd.DataFrame(paired).drop_duplicates(['dataset','scenario','policy','comparator','metric']);paired.to_csv(out/'confirmation_paired.csv',index=False);verdict=[]
 for ds in lock['primary']:
  def pick(comp,m):return paired[(paired.dataset==ds)&(paired.scenario=='ALL')&(paired.comparator==comp)&(paired.metric==m)].iloc[0][['mean','lo','hi','n']].to_dict()
  z=df[(df.dataset==ds)&(df.policy=='new_selected')];gain=pick(lock['roles']['frozen_detector'],'balanced_accuracy');increment=pick(lock['roles']['matched_verified'],'balanced_accuracy');ret=interval(z.groupby('seed').apply(lambda x:float((x.source_BA-x.initial_source1_BA).mean())));fpr=float(z.fpr.mean());sourcegap=pick(lock['roles']['matched_verified'],'source_BA');fpgap=pick(lock['roles']['matched_verified'],'fpr');verdict.append(dict(dataset=ds,BA_gain_vs_frozen=gain,BA_gain_vs_matched_verified=increment,retention_change_vs_initial=ret,FPR=fpr,source_gap_vs_matched=sourcegap,FPR_gap_vs_matched=fpgap,adaptation_passed=bool(gain['mean']>=.02 and ret['mean']>=-.02 and fpr<=.02),increment_screen_passed=bool(increment['mean']>=.01 and sourcegap['mean']>=-.02 and fpgap['mean']<=.01),increment_CI_positive=bool(increment['lo']>0)))
 families=[];curves=[];inspection=[];opportunity=[];updateeffects=[];sourcecurves=[]
 for _,r in df.iterrows():
  directory=ROOT/'results/confirmation';stem=r.run_id.replace(':','_');z=dict(np.load(directory/(stem+'.npz')));y=z['y'];p=z['p'];pred=p>=.5
  with gzip.open(directory/(stem+'_ledger.json.gz'),'rt',encoding='utf-8') as f:l=json.load(f)
  if 'test_family' in z:
   for family in np.unique(z['test_family']):
    ix=(z['test_family']==family)&(y==1)
    if ix.any():families.append(dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,policy=r.policy,family=family,support=int(ix.sum()),recall=float(pred[ix].mean()),TP=int(pred[ix].sum())))
  if r.policy=='oracle_100pct':continue
  for end in range(1000,len(y)+1000,1000):
   end=min(end,len(y));start=max(0,end-1000);entry=dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,policy=r.policy,end=end,queried=int(z['query'][:end].sum()),training_received=int((z['query']&~z['held']&(z['arrival']<end)).sum()),audit_received=int((z['held']&(z['arrival']<end)).sum()),attack_support=int(y[:end].sum()),block_attacks=int(y[start:end].sum()))
   for role,arr in [('system',p),('ML1',z['p1']),('ML2',z['p2'])]:
    entry.update({role+'_'+m:v for m,v in metrics(y[:end],arr[:end]).items() if m in ['balanced_accuracy','recall','fpr','AP','security_loss']});entry.update({role+'_block_'+m:v for m,v in metrics(y[start:end],arr[start:end]).items() if m in ['balanced_accuracy','recall','fpr']})
   curves.append(entry)
  if r.policy=='old_rvea_full':continue
  h1=z['p1']>=.5;h2=z['p2']>=.5;raw=z['raw']>=.5;disagree=h1!=h2;changed=raw!=h1;inspection.append(dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,policy=r.policy,disagreements=int(disagree.sum()),inspector_correct_on_disagreement=float(np.mean(h2[disagree]==y[disagree])) if disagree.any() else None,raw_corrections=int(changed.sum()),raw_correction_precision=float(np.mean(raw[changed]==y[changed])) if changed.any() else None,rescued_attacks_raw=int(((y==1)&~h1&raw).sum()),added_false_alarms_raw=int(((y==0)&~h1&raw).sum()),lost_attacks_raw=int(((y==1)&h1&~raw).sum()),threshold_rescues=int(((y==1)&~raw&pred).sum()),threshold_losses=int(((y==1)&raw&~pred).sum())))
  for row in l['source_checkpoints']:sourcecurves.append(dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,policy=r.policy,**row))
  for fit in l['fits']:
   start=fit['t']+1;end=min(start+100,len(y));role=fit['role'];key='p1' if role=='ML1' else 'p2'
   if end<=start:continue
   new=metrics(y[start:end],z[key][start:end]);old=metrics(y[start:end],z[key+'_previous'][start:end]);updateeffects.append(dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,policy=r.policy,role=role,fit_t=fit['t'],next_block_n=end-start,next_block_attacks=int(y[start:end].sum()),TP_change=new['tp']-old['tp'],FP_change=new['fp']-old['fp'],loss_change=new['security_loss']-old['security_loss'],BA_change=new['balanced_accuracy']-old['balanced_accuracy']))
  if r.policy!='new_selected' or 'test_family' not in z:continue
  for family in np.unique(z['test_family'][y==1]):
   mask=(z['test_family']==family)&(y==1);correct=[e for e in l['events'] if not e['held'] and e['label']==1 and mask[e['event']]];first=min(correct,key=lambda e:e['t']) if correct else None;first_t=first['t'] if first else None;row=dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,family=family,support=int(mask.sum()),overall_recall=float(pred[mask].mean()),first_correct_training_label_t=first_t)
   for role in ['ML1','ML2']:
    fits=[f for f in l['fits'] if f['role']==role and first is not None and f['t']>=first_t and first['event'] in f['verified_ids']];ft=min((f['t'] for f in fits),default=None);later=mask&(np.arange(len(y))>ft) if ft is not None else np.zeros(len(y),bool);row[role+'_first_fit_t']=ft;row[role+'_remaining_attack_support']=int(later.sum());row[role+'_later_system_recall']=float(pred[later].mean()) if later.any() else None
   opportunity.append(row)
 for filename,rows in [('family_scores',families),('learning_curves',curves),('inspector_quality',inspection),('label_opportunity',opportunity),('next_block_update_effect',updateeffects),('source_retention_curves',sourcecurves)]:pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
 families=pd.DataFrame(families);rare=[]
 for (ds,family),g in families[families.policy=='new_selected'].groupby(['dataset','family']):rare.append(dict(dataset=ds,family=family,unique_flow_support=int(g.support.iloc[0]),mean_recall=float(g.recall.mean()),aspiration_evaluable=bool(g.support.iloc[0]>=20),recall70_passed=bool(g.recall.mean()>=.7) if g.support.iloc[0]>=20 else None))
 result=dict(selected_config=lock['config'],roles=lock['roles'],trained_runs=len(df),predictions=int(df.n.sum()),real_human_events_received=int(df.human_origin_received.sum()),primary_verdict=verdict,adaptation_passed_both=all(z['adaptation_passed'] for z in verdict),increment_passed_both=all(z['increment_screen_passed'] for z in verdict),increment_CI_positive_both=all(z['increment_CI_positive'] for z in verdict),families=rare,scope='Frozenexternal evaluation. web_mainONEattackdescriptiveonly. web_challengecasecontrolnotnaturalprevalence.5seedsnot5independentsites. Realhumanoriginseparate; nogroundtruthfeedbackclaimedfrominspector.')
 (out/'verdict.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print(seed.groupby(['dataset','policy'])[['balanced_accuracy','fnr','fpr','source_BA','ML1_balanced_accuracy','ML2_balanced_accuracy']].mean().round(4).to_string())
if __name__=='__main__':main()
