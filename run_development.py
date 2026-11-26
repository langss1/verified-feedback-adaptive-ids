from pathlib import Path
import argparse,concurrent.futures,json,time,hashlib
from dataclasses import asdict
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from engine import Config,initialize
from experiment import simulate,load,name,save
ROOT=Path(__file__).resolve().parent
def screen_configs():
 return [Config(kind=kind,route=route,pseudo=pseudo) for kind in ['hgb','extra'] for route,pseudo in [('verified',False),('gate',False),('gate',True),('blend',False),('blend',True),('max',False)]]
def case(ds,seed,phase,configs,scenarios):
 out=ROOT/f'results/{phase}';file=out/f'case_{ds}_{seed}.csv';rows=pd.read_csv(file).to_dict('records') if file.exists() else [];done={r['run_id'] for r in rows};start=time.time()
 with threadpool_limits(limits=1):
  a=load(ds,seed);inits={kind:initialize(a,seed,kind) for kind in set(c.kind for c in configs)}
  for scenario in scenarios:
   order=np.random.default_rng(seed+len(scenario)).permutation(len(configs))
   for pos in order:
    cfg=configs[pos];rid=f'{ds}:{seed}:{scenario}:{name(cfg)}'
    if rid in done:continue
    r,tr,ledger=simulate(a,seed,scenario,cfg,inits[cfg.kind]);r.update(dataset=ds,run_id=rid,execution_order=int(pos));save(out,r,tr,ledger);rows.append(r);pd.DataFrame(rows).to_csv(file,index=False);done.add(rid);print('DEV',phase,ds,seed,scenario,name(cfg),'BA',round(r['balanced_accuracy'],4),'seconds',round(time.time()-start),flush=True)
 return len(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['screen','check']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();phase=args.phase
 if phase=='screen':configs=screen_configs();seeds=[11];scenarios=['delay80']
 else:
  shortlist=json.loads((ROOT/'protocol/shortlist.json').read_text());selected=[Config(**c) for c in shortlist['configs']];configs=[]
  for cfg in selected:
   for query in ['random','mixed']:configs.append(Config(**{**asdict(cfg),'query':query}))
  configs=list({name(c):c for c in configs}.values());seeds=[11,23,37];scenarios=['delay80','feedback_corrected','checker_fault']
 out=ROOT/f'results/{phase}';out.mkdir(parents=True,exist_ok=True);lock=dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),phase=phase,configs=[asdict(c) for c in configs],seeds=seeds,scenarios=scenarios,expected_runs=len(configs)*len(seeds)*len(scenarios)*4,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [ROOT/'engine.py',ROOT/'experiment.py',ROOT/'run_development.py',ROOT/'protocol/design.json']]);lockpath=ROOT/f'protocol/{phase}_lock.json'
 if lockpath.exists():
  prior=json.loads(lockpath.read_text());assert all(hashlib.sha256((ROOT/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in prior['files']);assert prior['configs']==lock['configs']
 else:lockpath.write_text(json.dumps(lock,indent=2))
 start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  tasks=[pool.submit(case,ds,s,phase,configs,scenarios) for ds in ['nsl','unsw','wednesday','friday'] for s in seeds]
  for job in concurrent.futures.as_completed(tasks):print('CASE DONE',job.result(),flush=True)
 df=pd.concat([pd.read_csv(p) for p in out.glob('case_*.csv')],ignore_index=True);assert len(df)==lock['expected_runs'] and df.run_id.is_unique;df.to_csv(out/'runs.csv',index=False);(out/'receipt.json').write_text(json.dumps(dict(completed=True,runs=len(df),seconds=time.time()-start,workers=args.workers),indent=2));print('PHASE DONE',phase,len(df))
if __name__=='__main__':main()
