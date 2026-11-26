from pathlib import Path
import json,gzip,itertools
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import numpy as np
import pandas as pd
from scipy.stats import t
from adapter import ROOT,D4
from experiment import metrics
POLICIES=['new_selected','matched_verified','simple_max','frozen_detector','arf_native_matched']
METRICS=['balanced_accuracy','recall','fpr','AP','macro_f1','security_loss','source_BA','source1_BA','initial_source1_BA','queried','arrived_train','arrived_audit','updates','training_rows']
def main():
 out=ROOT/'analysis';out.mkdir(exist_ok=True);new=pd.read_csv(ROOT/'results/runs.csv');old=pd.read_csv(D4/'analysis/confirmation_all_runs.csv');old=old[old.dataset.isin(['dos_main','bot_main'])&old.policy.isin(POLICIES[:-1])].copy();old['evidence']='original_D4_confirmation';data=pd.concat([old,new],ignore_index=True);assert len(data)==400 and data.run_id.is_unique
 data['source_change_pp']=100*(data.source_BA-data.initial_source1_BA);data['BA_pp']=100*data.balanced_accuracy;data['recall_pct']=100*data.recall;data['FPR_pct']=100*data.fpr;data.to_csv(out/'all_system_evaluations.csv',index=False)
 seedmeans=data.groupby(['dataset','policy','seed'])[METRICS+['source_change_pp']].mean().reset_index();seedmeans.to_csv(out/'seed_means.csv',index=False);summary=seedmeans.groupby(['dataset','policy'])[METRICS+['source_change_pp']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(out/'summary.csv',index=False)
 paired=[]
 for ds in sorted(data.dataset.unique()):
  for comparator in POLICIES[1:]:
   a=seedmeans[(seedmeans.dataset==ds)&(seedmeans.policy=='new_selected')].set_index('seed');b=seedmeans[(seedmeans.dataset==ds)&(seedmeans.policy==comparator)].set_index('seed');assert a.index.equals(b.index) and len(a)==5
   record=dict(dataset=ds,comparator=comparator,n_seeds=5)
   for metric in ['balanced_accuracy','recall','fpr','AP','source_BA','security_loss']:
    scale=1 if metric=='AP' else 100;diff=(a[metric]-b[metric])*scale;mean=diff.mean();margin=t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5);record.update({metric+'_difference':mean,metric+'_ci_low':mean-margin,metric+'_ci_high':mean+margin})
   paired.append(record)
 paired=pd.DataFrame(paired);paired.to_csv(out/'paired_capture_comparisons.csv',index=False)
 bycondition=[]
 for (ds,scenario),group in data.groupby(['dataset','scenario']):
  gate=group[group.policy=='new_selected'].set_index('seed')
  for comparator in POLICIES[1:]:
   base=group[group.policy==comparator].set_index('seed');diff=100*(gate.balanced_accuracy-base.balanced_accuracy)
   bycondition.append(dict(dataset=ds,scenario=scenario,comparator=comparator,BA_gap_pp=diff.mean(),seeds_positive=int((diff>0).sum()),mean_FPR_gap_pp=100*(gate.fpr-base.fpr).mean(),mean_loss_gap= (gate.security_loss-base.security_loss).mean()))
 pd.DataFrame(bycondition).to_csv(out/'condition_comparisons_descriptive.csv',index=False)
 gates=[];families=[];raw_metrics=[];sensitivity=[]
 k1=set(map(int,np.load(ROOT/'data/dos16_new_11.npz')['test_feature_key']));k2=set(map(int,np.load(ROOT/'data/ddos20_new_11.npz')['test_feature_key']));cross=k1&k2
 for ds,seed,scenario in itertools.product(sorted(data.dataset.unique()),[11,23,37,51,79],['clean1','delay80','feedback_corrected','checker_fault']):
  folder=D4/'results/confirmation' if ds in ['dos_main','bot_main'] else ROOT/'results';prefix=f'{ds}_{seed}_{scenario}_';tr=dict(np.load(folder/(prefix+'new_selected.npz')));h=tr['raw']>=.5;h1=tr['p1']>=.5;changed=h&~h1;mask=tr['raw']>=tr['threshold'];raw_h1=tr['p1']>=tr['threshold'];changed_cut=mask&~raw_h1;y=tr['y'];gates.append(dict(dataset=ds,seed=seed,scenario=scenario,eligible=int(tr['eligible'].sum()),raw_flips=int(changed.sum()),raw_rescues=int((changed&(y==1)).sum()),raw_extra_FP=int((changed&(y==0)).sum()),issued_cut_flips=int(changed_cut.sum()),issued_cut_rescues=int((changed_cut&(y==1)).sum()),issued_cut_extra_FP=int((changed_cut&(y==0)).sum()),arrived_positive_training=int(sum(e['label']==1 and not e['held'] and e['type']=='feedback' for e in json.loads(gzip.decompress((folder/(prefix+'new_selected_ledger.json.gz')).read_bytes()))['events']))))
  for policy in POLICIES:
   pf=ROOT/'results' if policy=='arf_native_matched' else folder;pt=dict(np.load(pf/(prefix+policy+'.npz')));hh=pt['p']>=.5
   raw_metrics.append(dict(dataset=ds,seed=seed,scenario=scenario,policy=policy,**metrics(pt['y'],pt.get('raw',pt['p']))))
   if ds in ['dos16_new','ddos20_new']:
    keep=np.asarray([int(k) not in cross for k in pt['test_feature_key']]);m=metrics(pt['y'][keep],pt['p'][keep]);full=metrics(pt['y'],pt['p']);sensitivity.append(dict(dataset=ds,seed=seed,scenario=scenario,policy=policy,removed=int((~keep).sum()),BA_change_pp=100*(m['balanced_accuracy']-full['balanced_accuracy']),**m))
   for fam in sorted(set(pt['test_family'][pt['y']==1])):
    support=(pt['test_family']==fam)&(pt['y']==1);families.append(dict(dataset=ds,seed=seed,scenario=scenario,policy=policy,family=str(fam),attacks=int(support.sum()),recall=float(hh[support].mean())))
 pd.DataFrame(gates).to_csv(out/'gate_coverage.csv',index=False);pd.DataFrame(families).to_csv(out/'family_scores.csv',index=False)
 pd.DataFrame(raw_metrics).to_csv(out/'raw_score_metrics.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'overlap_exclusion_sensitivity_posthoc.csv',index=False)
 budgets=[]
 for row in data.to_dict('records'):
  for cap in [.001,.005,.01,.02]:budgets.append({k:row[k] for k in ['dataset','seed','scenario','policy','balanced_accuracy','fpr']}|dict(FPR_cap=cap,within_budget=bool(row['fpr']<=cap)))
 pd.DataFrame(budgets).to_csv(out/'FPR_budgets_descriptive.csv',index=False)
 verdict={}
 for ds in ['dos16_new','ddos20_new']:
  gate=seedmeans[(seedmeans.dataset==ds)&(seedmeans.policy=='new_selected')];v=paired[(paired.dataset==ds)&(paired.comparator=='matched_verified')].iloc[0];f=paired[(paired.dataset==ds)&(paired.comparator=='frozen_detector')].iloc[0];s=paired[(paired.dataset==ds)&(paired.comparator=='simple_max')].iloc[0];a=paired[(paired.dataset==ds)&(paired.comparator=='arf_native_matched')].iloc[0]
  verdict[ds]=dict(gate_vs_verified_BA_pp=float(v.balanced_accuracy_difference),gate_vs_verified_CI=[float(v.balanced_accuracy_ci_low),float(v.balanced_accuracy_ci_high)],gate_vs_max_BA_pp=float(s.balanced_accuracy_difference),gate_vs_ARF_BA_pp=float(a.balanced_accuracy_difference),gate_increment_target_met=bool(v.balanced_accuracy_difference>=1 and v.balanced_accuracy_ci_low>0 and v.fpr_difference<=1 and v.source_BA_difference>=-2),adaptation_criteria_met=bool(f.balanced_accuracy_difference>=2 and gate.source_change_pp.mean()>=-2 and gate.fpr.mean()<=.02),adequate_support=json.loads((ROOT/f'data/{ds}_preparation.json').read_text())['adequate_primary_support'])
 verdict['consistent_gate_increment_on_both_new_captures']=all(verdict[d]['gate_increment_target_met'] and verdict[d]['adequate_support'] for d in ['dos16_new','ddos20_new']);verdict['scope']='Capture-level paired intervals over five algorithmic seeds, four conditions averaged within seed. Same benchmark, not independent organizations; baseline weighting/update/memory differ explicitly. No target-threshold or model retuning.'
 (out/'verdict.json').write_text(json.dumps(verdict,indent=2));print(summary[['dataset','policy','balanced_accuracy_mean','recall_mean','fpr_mean','source_change_pp_mean']].to_string(index=False));print(json.dumps(verdict,indent=2))
if __name__=='__main__':main()
