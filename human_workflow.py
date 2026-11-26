"""Resumable local human feedback workflow. Exported queues contain NO gold labels.
Real accuracy requires independent human audit labels; synthetic selftest is labelled.
"""
from pathlib import Path
import argparse,csv,json,pickle,datetime
import numpy as np
from threadpoolctl import threadpool_limits
from engine import Config,Session
from experiment import load,metrics

def create(dataset,seed):
 root=Path(__file__).resolve().parent;selected=root/'protocol/selected_method.json'
 if dataset not in ['nsl','unsw','wednesday','friday'] and not (root/'protocol/confirmation_lock.json').exists():raise ValueError('Externalcapture evaluation remainsreserved untilconfirmationlock')
 cfg=Config(**json.loads(selected.read_text())['config']) if selected.exists() else Config(pseudo=True)
 a=load(dataset,seed);core=Session({k:a[k] for k in ['base','meta','cal','y_base','y_meta','y_cal']},seed,cfg);metadata={k:a[k] for k in ['test_raw_rows','test_timestamps','test_source_index'] if k in a}
 space=(dataset if dataset in ['nsl','unsw'] else 'cic2017_history')+f'_seed{seed}'
 return dict(engine=core,X=a['test'],metadata=metadata,position=0,dataset=dataset,seed=seed,feature_space=space,streams=[dict(dataset=dataset,start=0,end=len(a['test']))],incoming=[],submissions=[])
def append_dataset(state,dataset):
 root=Path(__file__).resolve().parent
 if dataset not in ['nsl','unsw','wednesday','friday'] and not (root/'protocol/confirmation_lock.json').exists():raise ValueError('Externalcapture remainsreserved untilconfirmationlock')
 space=(dataset if dataset in ['nsl','unsw'] else 'cic2017_history')+f"_seed{state['seed']}";assert space==state['feature_space'],'Keep exactly the same fittedfeaturetransformation';assert dataset not in [s['dataset'] for s in state['streams']],'Do not append the same benchmarkcapturetwice';a=load(dataset,state['seed']);assert a['test'].shape[1]==state['X'].shape[1];offset=len(state['X']);state['X']=np.vstack([state['X'],a['test']]);state['streams'].append(dict(dataset=dataset,start=offset,end=len(state['X'])))
 for key in list(state['metadata']):
  if key in a:state['metadata'][key]=np.r_[state['metadata'][key],a[key]]
  else:state['metadata'].pop(key)
 return state
def advance(state,until):
 core=state['engine'];x=state['X'];target=min(until,len(x));assert target>=state['position']
 while state['position']<target:
  start=state['position'];end=min((start//100+1)*100,target,len(x));assert end%100==0 or end==len(x),'Advance to a100event boundary or the stream end';tr=core.predict_block(x[start:end])
  for pos,i in enumerate(range(start,end)):
   core.observe(x[i],{k:v[pos] for k,v in tr.items()});arrived=[e for e in state['incoming'] if e['t']==i]
   for e in arrived:core.receive(e)
   state['incoming']=[e for e in state['incoming'] if e['t']!=i]
  core.request_block(tr);core.update();state['position']=end
 return state
def submit(state,rows,origin='human'):
 core=state['engine'];seen={(e['event'],e['revision']) for e in [*state['incoming'],*core.events]};accepted=[]
 for row in rows:
  if str(row.get('human_label','')).strip()=='':continue
  j=int(row['request_id']);label=int(row['human_label']);revision=int(row.get('revision') or 0);t=int(row.get('available_event') or (state['position']+1));assert j in core.requests and label in [0,1] and revision>=0;assert t>=state['position'] and t>=core.requests[j]['request_t']+1 and (j,revision) not in seen
  if revision>0:
   previous=[e for e in [*state['incoming'],*core.events] if e['event']==j and e['revision']==revision-1];assert len(previous)==1 and previous[0]['t']<t,'Correction must follow previousrevisionavailability'
  else:assert not any(e['event']==j for e in [*state['incoming'],*core.events]),'Duplicate initialreview'
  event=dict(event=j,t=t,label=label,revision=revision,type='feedback' if revision==0 else 'correction',held=core.requests[j]['held'],origin=origin,reviewer=str(row.get('reviewer','')),imported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());accepted.append(event);seen.add((j,revision));state['incoming'].append(event)
 state['submissions'].extend(accepted);return len(accepted)
def export_queue(state,path):
 core=state['engine'];reviewed={e['event'] for e in [*core.events,*state['incoming']]};keys=['request_id','event_index','capture_dataset','source_raw_row','capture_time_ns','role','query_reason','ML1_attack_score','ML2_attack_score','combined_attack_score','human_label','revision','available_event','reviewer','transformed_features_json']
 with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for j,req in sorted(core.requests.items()):
   if j in reviewed:continue
   meta=state['metadata'];raw=meta.get('test_raw_rows',meta.get('test_source_index',[]));capture=meta.get('test_timestamps',[])
   capture_name=next(s['dataset'] for s in state['streams'] if s['start']<=j<s['end']);writer.writerow(dict(request_id=j,event_index=j,capture_dataset=capture_name,source_raw_row=int(raw[j]) if len(raw) else '',capture_time_ns=int(capture[j]) if len(capture) else '',role='held_audit' if req['held'] else 'training',query_reason=req['reason'],ML1_attack_score=core.p1_issued[j],ML2_attack_score=core.p2_issued[j],combined_attack_score=core.issued[j],human_label='',revision=0,available_event=state['position']+1,reviewer='',transformed_features_json=json.dumps(core.rows[j].tolist())))
def status(state):
 core=state['engine'];human=[e for e in core.events if e.get('origin')=='human'];auditids=sorted(core.audit);result=dict(dataset=state['dataset'],captures=[s['dataset'] for s in state['streams']],total_available_rows=len(state['X']),observed=state['position'],requests=len(core.requests),training_received=len(core.verified),held_audit_received=len(core.audit),real_human_events_received=len(human),synthetic_events_received=sum(e.get('origin')=='synthetic_selftest' for e in core.events),pending_feedback=len(state['incoming']),ML1_version=core.v1,ML2_version=core.v2,ML1_retrains=core.updates,corrections_received=sum(e['type']=='correction' for e in core.events),performance_scope='Only independentheldreviews, not trainingresubstitution; no datasetgold is stored in this session')
 if auditids:
  y=np.asarray([core.audit[j] for j in auditids]);result['held_review_n']=len(y)
  if len(y)>=20 and min(int((y==c).sum()) for c in [0,1])>=2:result['held_review_metrics']={key:metrics(y,np.asarray(p)[auditids]) for key,p in [('ML1',core.p1_issued),('ML2',core.p2_issued),('system',core.issued)]}
  else:result['held_review_metrics']=None;result['held_review_note']='Insufficient independentreview support for performance: require20reviews and>=2perclass'
 return result
def save_state(state,path):
 with Path(path).open('wb') as f:pickle.dump(state,f,pickle.HIGHEST_PROTOCOL)
def read_state(path):
 with Path(path).open('rb') as f:return pickle.load(f)
def selftest(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);state=create('nsl',11);advance(state,100);prefix=np.asarray(state['engine'].issued).copy();export_queue(state,output/'example_blank_queue.csv');reqs=sorted(state['engine'].requests);rows=[dict(request_id=j,human_label=i%2,revision=0,available_event=101,reviewer='SYNTHETIC_SELFTEST') for i,j in enumerate(reqs)];assert submit(state,rows,origin='synthetic_selftest')==5;advance(state,200);train=next(j for j in reqs if not state['engine'].requests[j]['held']);old=state['engine'].verified[train];submit(state,[dict(request_id=train,human_label=1-old,revision=1,available_event=201,reviewer='SYNTHETIC_SELFTEST')],origin='synthetic_selftest');advance(state,300);assert np.array_equal(prefix,state['engine'].issued[:100]);save_state(state,output/'synthetic_session.pkl');restored=read_state(output/'synthetic_session.pkl');assert restored['engine'].verified[train]==1-old and restored['engine'].v1>0 and restored['engine'].v2>0;result=status(restored);assert result['real_human_events_received']==0;result['passed']=True;(output/'selftest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
 new=sub.add_parser('new');new.add_argument('--dataset',default='nsl');new.add_argument('--seed',type=int,default=11);new.add_argument('--session',required=True);new.add_argument('--until',type=int,default=100)
 go=sub.add_parser('advance');go.add_argument('--session',required=True);go.add_argument('--until',type=int,required=True)
 feed=sub.add_parser('submit');feed.add_argument('--session',required=True);feed.add_argument('--csv',required=True)
 export=sub.add_parser('export');export.add_argument('--session',required=True);export.add_argument('--csv',required=True)
 stat=sub.add_parser('status');stat.add_argument('--session',required=True)
 add=sub.add_parser('append');add.add_argument('--session',required=True);add.add_argument('--dataset',required=True)
 test=sub.add_parser('selftest');test.add_argument('--output',default='checks/human_workflow')
 args=ap.parse_args()
 with threadpool_limits(limits=1):
  if args.command=='selftest':selftest(args.output);return
  if args.command=='new':
   if Path(args.session).exists():raise SystemExit('Choose a new session filename')
   state=create(args.dataset,args.seed);advance(state,args.until)
  else:state=read_state(args.session)
  if args.command=='advance':advance(state,args.until)
  elif args.command=='submit':
   with Path(args.csv).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
   print('accepted',submit(state,rows))
  elif args.command=='export':export_queue(state,args.csv)
  elif args.command=='append':append_dataset(state,args.dataset)
  if args.command in ['new','advance','submit','append']:save_state(state,args.session)
  print(json.dumps(status(state),indent=2))
if __name__=='__main__':main()
