"""Replay frozen prepared-input experiments in a new directory, never in place."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['dos_main','bot_main','web_challenge','web_main'],default='bot_main');ap.add_argument('--seed',type=int,choices=[11,23,37,51,79],default=11);ap.add_argument('--scenario',choices=['clean1','delay80','feedback_corrected','checker_fault'],default='delay80');ap.add_argument('--all',action='store_true');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
 lock=json.loads((ROOT/'protocol/confirmation_lock.json').read_text())
 for e in lock['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');dst=ROOT/'replays'/stamp;dst.mkdir(parents=True,exist_ok=False)
 paths={e['path'] for e in lock['files']}|{p.name for p in ROOT.glob('*.py')}|{'protocol/confirmation_lock.json','environment.json','requirements.txt'}
 paths.update(p.name for p in [ROOT/'README.md',ROOT/'README.txt'] if p.exists())
 for rel in sorted(paths):
  source=ROOT/rel;target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 (dst/'checks').mkdir(exist_ok=True);(dst/'analysis').mkdir(exist_ok=True)
 if args.all:
  (dst/'results').mkdir(exist_ok=True)
  subprocess.run([sys.executable,'run_confirmation.py','--workers',str(args.workers)],cwd=dst,check=True)
  subprocess.run([sys.executable,'audit_results.py','confirmation'],cwd=dst,check=True)
  print('Fresh full confirmation:',dst);return
 archive=ROOT/'results/confirmation/runs.csv'
 if not archive.exists():archive=ROOT/f'results/confirmation/case_{args.dataset}_{args.seed}.csv'
 assert archive.exists(),'The requested original run must finish before hash comparison'
 code=f'DATASET={args.dataset!r}\nSEED={args.seed}\nSCENARIO={args.scenario!r}\n'+'''from pathlib import Path
import gzip,json,pickle,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from engine import Config,initialize
from experiment import load,simulate,save
root=Path('.');cfg=Config(**json.loads((root/'protocol/confirmation_lock.json').read_text())['config'])
with threadpool_limits(limits=1):
 a=load(DATASET,SEED);ret=dict(np.load(root/f'inputs/retention_{SEED}.npz'));a['cal']=ret['cal'];a['y_cal']=ret['y_cal'];init=initialize(a,SEED,cfg.kind);r,tr,ledger,core=simulate(a,SEED,SCENARIO,cfg,init,keep_session=True)
 r.update(dataset=DATASET,policy='new_selected',run_id=f'{DATASET}:{SEED}:{SCENARIO}:new_selected');out=root/'results/confirmation';out.mkdir(parents=True);save(out,r,tr,ledger);pd.DataFrame([r]).to_csv(out/'runs.csv',index=False)
 (root/'replay_result.json').write_text(json.dumps(r,indent=2))
 snapshot=dict(detector=core.m1,inspector=core.m2,threshold=core.threshold,radius=core.init['radius'],config=cfg.__dict__,seed=SEED,feature_count=a['test'].shape[1],feedback_origin='simulator',dataset=DATASET,scenario=SCENARIO,observed=len(a['test']),ML1_version=core.v1,ML2_version=core.v2,source_prediction_sha256=r['source_prediction_sha256'],scope='Illustrative final benchmark snapshot from source and arrived simulated labels, not real-human or deployment validation')
 with (root/'final_snapshot.pkl').open('wb') as f:pickle.dump(snapshot,f,protocol=5)
'''
 (dst/'fresh_one.py').write_text(code)
 subprocess.run([sys.executable,'fresh_one.py'],cwd=dst,check=True)
 import csv
 rid=f'{args.dataset}:{args.seed}:{args.scenario}:new_selected';expected=next(r for r in csv.DictReader(archive.open()) if r['run_id']==rid);actual=json.loads((dst/'replay_result.json').read_text());fields=['prediction_sha256','source_prediction_sha256'];passed=all(expected[k]==actual[k] for k in fields);result=dict(passed=passed,run_id=rid,workspace=str(dst),checked={k:dict(expected=expected[k],actual=actual[k]) for k in fields},scope='Fresh copied prepared-input workspace and refit, one frozen configuration; timing excluded; not all seeds/cases replayed')
 (dst/'fresh_replay_check.json').write_text(json.dumps(result,indent=2));(ROOT/'checks/fresh_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert passed
 models=ROOT/'model_examples';models.mkdir(exist_ok=True);target=models/f'{args.dataset}_{args.seed}_{args.scenario}_simulated_snapshot.pkl';shutil.copy2(dst/'final_snapshot.pkl',target);result['exported_snapshot']=str(target);result['snapshot_sha256']=sha(target);(ROOT/'checks/fresh_replay.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
