from pathlib import Path
import argparse,sys,json,hashlib,time,gzip,importlib.util,concurrent.futures
from dataclasses import asdict
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from engine import Config,initialize,digest
from experiment import SEEDS,SCENARIOS,simulate,load,save,metrics,name
from oracle_reference import run as oracle
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def definitions(cfg):
 entries=[('new_selected',cfg),('matched_verified',Config(**{**asdict(cfg),'route':'verified','pseudo':False})),('simple_max',Config(**{**asdict(cfg),'route':'max','pseudo':False})),('without_pseudo',Config(**{**asdict(cfg),'pseudo':False})),('frozen_detector',Config(**{**asdict(cfg),'route':'frozen','pseudo':False})),('budget_full_history',Config(**{**asdict(cfg),'route':'verified','pseudo':False,'full_history':True,'interval':500}))];unique=[];seen={};roles={}
 for role,c in entries:
  key=json.dumps(asdict(c),sort_keys=True)
  if key not in seen:seen[key]=role;unique.append((role,asdict(c)))
  roles[role]=seen[key]
 roles['old_rvea_full']='old_rvea_full';return unique,roles
def freeze():
 selected=json.loads((ROOT/'protocol/selected_method.json').read_text());prep=json.loads((ROOT/'data/external/preparation.json').read_text());assert all(k in prep['cohorts'] for k in ['web_main','web_challenge','bot_main','dos_main']);assert prep['cohorts']['bot_main']['adequate_primary_attack_support'] and prep['cohorts']['dos_main']['adequate_primary_attack_support'];cfg=Config(**selected['config']);policies,roles=definitions(cfg)
 model_files=['engine.py','experiment.py','run_confirmation.py','oracle_reference.py','check_information_flow.py','human_workflow.py','select_development.py','run_development.py','analyze_results.py']
 files=[*[ROOT/n for n in model_files],*ROOT.glob('protocol/*.json'),*ROOT.glob('inputs/*.npz'),*ROOT.glob('data/*.json'),*ROOT.glob('data/external/*.npz'),*ROOT.glob('data/external/*.json'),*ROOT.glob('provenance/*.json'),*ROOT.glob('legacy_d2/**/*.py')];files=sorted(p for p in files if p.is_file() and p.name!='confirmation_lock.json');lock=dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),selected=selected['selected'],config=selected['config'],roles=roles,policies=policies,cohorts=['dos_main','bot_main','web_challenge','web_main'],primary=['dos_main','bot_main'],secondary='web_challenge',underpowered_descriptive='web_main',seeds=SEEDS,scenarios=SCENARIOS,expected_runs=15*((len(policies)+1)*4)+15+15,underpowered_scope='Onlydelay80/newselected/matchedverified/frozendetector; noattack-performanceclaimfromoneattack',oracle_scope='100percentperfectpastlabels,200eventrefits,onlyclean1,notweb_main,notmathematicallyoptimalbound',fullhistory_scope='Same5percentlabelbudget; sourceall8192,currentallarrived; conventional500cadence+correctionupdates; matchedverified200isprimarycausalreference',files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files],scope='No2018predictivescorebeforethislock; firsttwoCIC2017daysnowdevelopment; unusedhistoricalS3probeonlyevaluatedafterlock; plotting/PDFlayout excludedfrommodel-sourcefreeze')
 file=ROOT/'protocol/confirmation_lock.json';assert not file.exists(),'Do not replace a confirmation lock';file.write_text(json.dumps(lock,indent=2));print(json.dumps({k:v for k,v in lock.items() if k!='files'},indent=2))
def legacy_import():
 sys.path.insert(0,str(ROOT/'legacy_d2/src'))
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 return module('d4_legacy_dual',ROOT/'legacy_d2/src/dual.py'),module('d4_legacy_threshold',ROOT/'legacy_d2/calibration_control.py')
def old_run(a,seed,scenario,legacy,calibrator,init):
 stream=np.arange(len(a['test']));teacher=legacy.teacher_trace(a,stream,seed,scenario,init,'strategic');r,decisions,events,tr=legacy.run(a,stream,np.asarray([],int),seed,scenario,'dual_E1_B1_R1',init,teacher);prob,threshold=calibrator.replay(tr['p'],events);source=calibrator.shifted(tr['source_p'],threshold);tr.update(raw=tr['p'].copy(),source_raw=tr['source_p'].copy(),p=prob,source_p=source);r['legacy_raw_AP']=r.pop('pr_auc');r.update(metrics(tr['y'],prob));r.update(policy='old_rvea_full',source_BA=metrics(tr['source_y'],source)['balanced_accuracy'],source1_BA=r['source_student_BA'],initial_source1_BA=r['initial_source_BA'],prediction_sha256=digest(prob),source_prediction_sha256=digest(source),feedback_origin='simulator',held_requested=int(tr['held'].sum()),human_origin_received=0,online_s=r['online_loop_s']+teacher['cost_s'],fit_s=r['student_train_s'],final_threshold=threshold)
 for role,key in [('ML1','p1'),('ML2','p2')]:r.update({role+'_'+m:v for m,v in metrics(tr['y'],tr[key]).items() if m!='n'})
 for key in ['test_family','test_raw_rows','test_timestamps','test_attempted']:
  if key in a:tr[key]=a[key]
 ledger=dict(decisions=decisions,events=events,source_replay_rows=init['old'].tolist(),teacher_fit_log=teacher['fit_log'],source_prototypes=init['teacher_rows'].tolist(),scope='UnmodifiedD2full111plusoriginalfixedthreshold; lastpartialblockrequestsmin5 ratherthanrounded5percent,actualbudgetreported; primary12kstreamsareexactlymatched5percentrequests')
 return r,tr,ledger
def case(ds,seed,lock):
 out=ROOT/'results/confirmation';file=out/f'case_{ds}_{seed}.csv';rows=pd.read_csv(file).to_dict('records') if file.exists() else [];done={r['run_id'] for r in rows};start=time.time();cfg=Config(**lock['config'])
 with threadpool_limits(limits=1):
  a=load(ds,seed);ret=dict(np.load(ROOT/f'inputs/retention_{seed}.npz'));a['cal']=ret['cal'];a['y_cal']=ret['y_cal'];init=initialize(a,seed,cfg.kind);legacy,calibrator=legacy_import();oldinit=legacy.initial(a,seed,'knn')
  for scenario in (['delay80'] if ds=='web_main' else SCENARIOS):
   entries=[*lock['policies'],('old_rvea_full',None)];order=np.random.default_rng(seed+len(scenario)).permutation(len(entries))
   if ds=='web_main':entries=[e for e in entries if e[0] in ['new_selected',lock['roles']['matched_verified'],lock['roles']['frozen_detector']]];order=np.arange(len(entries))
   for pos in order:
    alias,c=entries[pos];rid=f'{ds}:{seed}:{scenario}:{alias}'
    if rid in done:continue
    if c is None:r,tr,ledger=old_run(a,seed,scenario,legacy,calibrator,oldinit)
    else:r,tr,ledger=simulate(a,seed,scenario,Config(**c),init)
    r.update(dataset=ds,policy=alias,run_id=rid,execution_order=int(pos));save(out,r,tr,ledger);rows.append(r);done.add(rid);pd.DataFrame(rows).to_csv(file,index=False);print('CONFIRM',ds,seed,scenario,alias,'BA',round(r['balanced_accuracy'],4),'seconds',round(time.time()-start),flush=True)
  if ds!='web_main':
   rid=f'{ds}:{seed}:clean1:oracle_100pct'
   if rid not in done:
    r,tr,ledger=oracle(a,seed,cfg.kind,cfg.positive_weight,init);r.update(dataset=ds,run_id=rid);save(out,r,tr,ledger);rows.append(r);pd.DataFrame(rows).to_csv(file,index=False);print('ORACLE',ds,seed,'BA',round(r['balanced_accuracy'],4),flush=True)
 return len(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
 if args.freeze:freeze();return
 lock=json.loads((ROOT/'protocol/confirmation_lock.json').read_text())
 for e in lock['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
 out=ROOT/'results/confirmation';out.mkdir(parents=True,exist_ok=True);start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  jobs=[pool.submit(case,ds,seed,lock) for ds in lock['cohorts'] for seed in SEEDS]
  for job in concurrent.futures.as_completed(jobs):print('CONFIRM CASE DONE',job.result(),flush=True)
 df=pd.concat([pd.read_csv(p) for p in out.glob('case_*.csv')],ignore_index=True);assert len(df)==lock['expected_runs'] and df.run_id.is_unique;df.to_csv(out/'runs.csv',index=False);(out/'receipt.json').write_text(json.dumps(dict(completed=True,runs=len(df),workers=args.workers,seconds=time.time()-start),indent=2));print('CONFIRM COMPLETE',len(df))
if __name__=='__main__':main()
