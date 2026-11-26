from pathlib import Path
import sys,json,pickle,time
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import numpy as np
from threadpoolctl import threadpool_limits
from adapter import ROOT,D4,new_arf,dictionaries,run_arf,event_hash
from engine import initialize,Config,digest
from experiment import simulate
from run_extension import history
def main():
 start=time.time();ds='dos16_new';seed=11;scenario='feedback_corrected';a=history(seed);a.update(dict(np.load(ROOT/f'data/{ds}_{seed}.npz')))
 cfg=Config(**json.loads((D4/'protocol/selected_method.json').read_text())['config']);initial=initialize(a,seed,'hgb');r,tr,ledger=simulate(a,seed,scenario,cfg,initial)
 reference=dict(np.load(ROOT/f'results/{ds}_{seed}_{scenario}_new_selected.npz'));assert np.array_equal(reference['p'],tr['p']);assert np.array_equal(reference['source_p'],tr['source_p'])
 model=new_arf(seed)
 for x,y in zip(dictionaries(a['base'][initial['source']]),a['y_base'][initial['source']]):model.learn_one(x,int(y))
 other,detail=run_arf(a['test'],pickle.dumps(model),ledger['events'],a['cal'][initial['cal']]);ref=dict(np.load(ROOT/f'results/{ds}_{seed}_{scenario}_arf_native_matched.npz'));assert np.array_equal(ref['p'],other['p']);assert np.array_equal(ref['source_p'],other['source_p'])
 report=dict(passed=True,dataset=ds,seed=seed,scenario=scenario,initial_models_rebuilt_from_source=True,GDT_target_sha256=digest(tr['p']),GDT_source_sha256=digest(tr['source_p']),ARF_target_sha256=digest(other['p']),ARF_source_sha256=digest(other['source_p']),feedback_schedule_sha256=event_hash(ledger),ARF_correction_resets=detail['resets'],seconds=time.time()-start,scope='One corrected-feedback new-capture case replayed in a separate process; exact target/source prediction agreement, not a universal correctness proof.')
 (ROOT/'checks/fresh_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
