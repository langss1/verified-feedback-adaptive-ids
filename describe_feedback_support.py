"""Exploratory label-support description; does not change predictions or criteria."""
from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
def main():
 df=pd.read_csv(ROOT/'results/confirmation/runs.csv');assert len(df)==390;rows=[]
 for _,r in df[df.policy!='oracle_100pct'].iterrows():
  stem=r.run_id.replace(':','_');z=np.load(ROOT/f'results/confirmation/{stem}.npz');y=z['y']
  with gzip.open(ROOT/f'results/confirmation/{stem}_ledger.json.gz','rt',encoding='utf-8') as f:events=json.load(f)['events']
  train={};audit={}
  for e in events:(audit if e['held'] else train)[int(e['event'])]=int(e['label'])
  entry=dict(dataset=r.dataset,seed=r.seed,scenario=r.scenario,policy=r.policy,feedback_origin='simulator',train_arrived=len(train),audit_arrived=len(audit),correction_events=sum(e['type']=='correction' for e in events),scope='Post-lock exploratory description of saved feedback support, not real-human accuracy')
  for role,d in [('train',train),('audit',audit)]:entry.update({role+'_true_attacks':sum(int(y[j]) for j in d),role+'_reported_positive':sum(d.values()),role+'_correct_positive':sum(y[j]==1 and label==1 for j,label in d.items()),role+'_currently_wrong':sum(y[j]!=label for j,label in d.items())})
  rows.append(entry)
 out=pd.DataFrame(rows);out.to_csv(ROOT/'analysis/feedback_class_support.csv',index=False);selected=out[(out.policy=='new_selected')&out.dataset.isin(['dos_main','bot_main'])];summary=[]
 for ds,g in selected.groupby('dataset'):summary.append(dict(dataset=ds,runs=len(g),mean_correct_positive_training=float(g.train_correct_positive.mean()),mean_correct_positive_audit=float(g.audit_correct_positive.mean()),runs_without_correct_positive_audit=int((g.audit_correct_positive==0).sum()),scope='Exploratory budget diagnostic; repeated conditions/seeds on the same flow cohort, not a causal effect estimate'))
 (ROOT/'analysis/feedback_support_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
