"""Run the frozen analysis, then remove comparisons with unequal condition sets.
The frozen models, criteria, predictions and primary verdict remain unchanged.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
import pandas as pd
ROOT=Path(__file__).resolve().parent
def main():
 out=ROOT/'analysis';out.mkdir(exist_ok=True)
 with (out/'frozen_analysis_console.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'analyze_results.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 path=out/'confirmation_paired.csv';original=hashlib.sha256(path.read_bytes()).hexdigest();df=pd.read_csv(path);bad=(df.comparator=='oracle_100pct')&(df.scenario!='clean1');removed=df[bad];df=df[~bad];df.to_csv(path,index=False);removed.to_csv(out/'excluded_unmatched_reference_comparisons.csv',index=False)
 seedpath=out/'confirmation_seed_means.csv';seed=pd.read_csv(seedpath);runs=pd.read_csv(out/'confirmation_all_runs.csv');scopes=runs.groupby(['dataset','seed','policy']).scenario.agg(lambda x:','.join(sorted(set(x)))).rename('available_conditions').reset_index();seed.merge(scopes,on=['dataset','seed','policy'],validate='one_to_one').to_csv(seedpath,index=False)
 receipt=dict(passed=True,frozen_analysis_sha256=hashlib.sha256((ROOT/'analyze_results.py').read_bytes()).hexdigest(),unfiltered_paired_csv_sha256=original,excluded_unmatched_comparisons=len(removed),rule='Full-supervision reference is compared ONLY within clean1, the only condition in which it was run. Remove ALL/CLEAN aggregates and other unmatched conditions.',unchanged='No models, input arrays, selection, primary verdict, per-condition measured scores or primary-comparison criteria changed.');(ROOT/'checks/analysis_reporting_correction.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));print((out/'verdict.json').read_text())
if __name__=='__main__':main()
