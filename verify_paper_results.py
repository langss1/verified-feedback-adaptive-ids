"""Independently recompute paper Table II, paired intervals and issued-score mapping.

Reads prepared saved traces, never trains a model or overwrites experiment results.
Run after restoring both D4 and D5 public release artifacts; see reviewer guide.
"""
from pathlib import Path
import argparse,json,csv,re,importlib.util,itertools
import numpy as np
import pandas as pd
from scipy.stats import t as student

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--d4',type=Path,default=ROOT)
    parser.add_argument('--d5',type=Path,default=ROOT.parent/'adaptive_baseline_d5_20261126')
    parser.add_argument('--paper',type=Path,default=ROOT/'manuscripts/r11/main.tex')
    parser.add_argument('--out',type=Path,default=ROOT/'checks/paper_audit')
    args=parser.parse_args();D4=args.d4.resolve();D5=args.d5.resolve()
    DOC=args.out.resolve();DOC.mkdir(parents=True,exist_ok=True)
    SEEDS=[11,23,37,51,79];SCENARIOS=['clean1','delay80','feedback_corrected','checker_fault']
    spec=importlib.util.spec_from_file_location('independent_auditor',D4/'audit_results.py');aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)
    datasets=['dos_main','bot_main','dos16_new','ddos20_new']
    policies=['new_selected','matched_verified','simple_max','frozen_detector','arf_native_matched']
    cache={};raw={};max_score_equation_error=0.
    for ds,seed,scenario,policy in itertools.product(datasets,SEEDS,SCENARIOS,policies):
        folder=D4/'results/confirmation' if ds in datasets[:2] and policy!='arf_native_matched' else D5/'results'
        with np.load(folder/f'{ds}_{seed}_{scenario}_{policy}.npz') as z:
            u=np.clip(z['raw'],1e-10,1-1e-10);tau=z['threshold']
            score=u*(1-tau)/(u*(1-tau)+(1-u)*tau)
            error=float(np.max(np.abs(score-z['p'])));assert error<1e-14
            max_score_equation_error=max(max_score_equation_error,error)
            assert np.array_equal(score>=.5,z['p']>=.5)
            m=aud.metrics(z['y'],z['p']);m['source_BA']=aud.metrics(z['source_y'],z['source_p'])['balanced_accuracy']
            cache[(ds,seed,scenario,policy)]=m
            if policy=='new_selected':
                for role in ['ML1','ML2']:raw[(ds,seed,scenario,role)]=aud.metrics(z['y'],z['p1' if role=='ML1' else 'p2'])

    tex=args.paper.read_text(encoding='utf-8')
    section=tex.split(r'\label{tab:results}')[1].split(r'\end{table*}')[0]
    expected=re.findall(r'&\s*([0-9]+\.[0-9]+)',section);assert len(expected)==70
    rows=['ML1','ML2',*policies];fields=['balanced_accuracy','recall','fpr','AP'];cells=[];index=0
    for method in rows:
        for ds in datasets:
            for field in (fields if ds in datasets[:2] else ['balanced_accuracy']):
                source=raw if method in ['ML1','ML2'] else cache
                v=float(np.mean([np.mean([source[(ds,seed,sc,method)][field] for sc in SCENARIOS]) for seed in SEEDS]))
                scaled=v if field=='AP' else 100*v;formatted=f'{scaled:.3f}' if field=='AP' else f'{scaled:.2f}'
                assert formatted==expected[index],(method,ds,field,formatted,expected[index])
                cells.append(dict(table='II',method=method,dataset=ds,metric=field,reported=expected[index],recomputed=scaled,rounding_match=True));index+=1

    published=pd.read_csv(D5/'analysis/paired_capture_comparisons.csv');pairs=[]
    for row in published.to_dict('records'):
        ds=row['dataset'];comp=row['comparator']
        for field in ['balanced_accuracy','recall','fpr','AP','source_BA','security_loss']:
            scale=1 if field=='AP' else 100
            values=np.array([scale*np.mean([cache[(ds,s,sc,'new_selected')][field]-cache[(ds,s,sc,comp)][field] for sc in SCENARIOS]) for s in SEEDS])
            mean=float(values.mean());margin=float(student.ppf(.975,4)*values.std(ddof=1)/np.sqrt(5))
            for suffix,value in [('difference',mean),('ci_low',mean-margin),('ci_high',mean+margin)]:
                actual=row[field+'_'+suffix];assert abs(actual-value)<1e-9,(ds,comp,field,suffix,actual,value)
            pairs.append(dict(dataset=ds,comparator=comp,metric=field,mean=mean,ci_low=mean-margin,ci_high=mean+margin))

    with (DOC/'table_ii_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    with (DOC/'paired_interval_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    report=dict(passed=True,saved_system_evaluations_recomputed=len(cache),table_II_cells_verified=len(cells),paired_metric_intervals_verified=len(pairs),interval_means_and_endpoints_verified=3*len(pairs),issued_score_equation_max_error=max_score_equation_error,scope='400 correlated system evaluations include 160 reused D4 and 240 additional D5 evaluations. No new training or universal correctness claim.')
    (DOC/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
