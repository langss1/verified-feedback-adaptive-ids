"""Export source-only transforms already used in confirmation; never refit on target."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent
def transform(x,params):
 x=np.asarray(x,dtype=float);x=np.where(np.isfinite(x),x,np.asarray(params['median']));x=np.sign(x)*np.log1p(np.abs(x));return np.clip((x-np.asarray(params['mean']))/np.asarray(params['scale']),-10,10)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-archive',default=str(ROOT.parent/'dual_verifier_d2_20261125/data/CICIDS2017_improved.zip'));args=ap.parse_args();path=Path(args.source_archive);expected=json.loads((ROOT/'data/source_archive_recheck.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()=='97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883';features=json.loads((ROOT/'data/external/preparation.json').read_text())['feature_columns']
 with zipfile.ZipFile(path) as z:
  with z.open('tuesday.csv') as f:raw=pd.read_csv(f,usecols=lambda c:c.strip() in features,low_memory=False)
 raw.columns=raw.columns.str.strip();x=raw[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float);out=ROOT/'deployment';out.mkdir(exist_ok=True);checks=[]
 for seed in [11,23,37,51,79]:
  rows=json.loads((ROOT/f'provenance/history_rows_{seed}.json').read_text())['base'];imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x.iloc[rows]);xx=imp.transform(x.iloc[rows]);xx=np.sign(xx)*np.log1p(np.abs(xx));sc=StandardScaler().fit(xx);params=dict(seed=seed,feature_columns=features,median=imp.statistics_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),clip=[-10,10],scope='Source BASE-only fit, exported after confirmation lock with exact historical parity; not used to change frozen experiments');actual=transform(x.iloc[rows],params);ref=np.load(ROOT/f'inputs/history_{seed}.npz')['base'];error=float(np.max(np.abs(actual-ref)));assert np.allclose(actual,ref,atol=1e-12,rtol=0);target=out/f'transform_seed_{seed}.json';target.write_text(json.dumps(params,indent=2,allow_nan=False));checks.append(dict(seed=seed,base_rows=len(rows),maximum_absolute_error=error,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
  if seed==11:x.iloc[rows[:10]].to_csv(out/'example_raw_features.csv',index=False,float_format='%.17g');np.savez_compressed(out/'example_transformed_reference.npz',X=ref[:10])
 result=dict(passed=True,source_archive_sha256='97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883',checks=checks,scope='No classifier or target-label fit; parameter export of the source transform already used');(ROOT/'checks/exported_transformers.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
