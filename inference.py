"""Predict on corrected flow-feature CSVs using the supplied illustrative snapshot."""
from pathlib import Path
import argparse,json,pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from export_transformers import transform
def predict(snapshot,x):
 p1=snapshot['detector'].predict_proba(x)[:,1];p2,counts,near,_=snapshot['inspector'].inspect(x);eligible=(p2>=.8)&(counts[:,1]>=2)&(counts[:,0]==0)&(near[:,1]<=snapshot['radius']);raw=np.where(eligible,np.maximum(p1,p2),p1);q=np.clip(raw,1e-10,1-1e-10);tau=snapshot['threshold'];odds=q/(1-q)*(1-tau)/tau;score=odds/(1+odds)
 return pd.DataFrame(dict(ML1_attack_score=p1,ML2_attack_score=p2,verified_gate_eligible=eligible,raw_combined_score=raw,decision_score=score,ML1_label=(p1>=.5).astype(int),ML2_label=(p2>=.5).astype(int),system_label=(score>=.5).astype(int)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--transformer',required=True);ap.add_argument('--input-csv',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();params=json.loads(Path(args.transformer).read_text());raw=pd.read_csv(args.input_csv,low_memory=False);raw.columns=raw.columns.str.strip();missing=set(params['feature_columns'])-set(raw.columns);assert not missing,f'Missing predictor columns: {sorted(missing)}';xx=raw[params['feature_columns']].apply(pd.to_numeric,errors='coerce').to_numpy(float);x=transform(xx,params)
 with Path(args.snapshot).open('rb') as f:snapshot=pickle.load(f)
 assert snapshot['seed']==params['seed'] and snapshot['config']['route']=='gate' and not snapshot['config']['pseudo'];assert x.shape[1]==snapshot['feature_count']
 with threadpool_limits(limits=1):pred=predict(snapshot,x)
 pred.to_csv(args.output,index=False);print(json.dumps(dict(rows=len(pred),output=str(Path(args.output).resolve()),snapshot_origin=snapshot['feedback_origin'],scope='Fixed final benchmark snapshot; no online update or accuracy claim from unlabeled inputs'),indent=2))
if __name__=='__main__':main()
