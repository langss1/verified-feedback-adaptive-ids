"""Prepare predeclared unseen capture members without fitting/scoring a classifier."""
from pathlib import Path
import sys,io,json,hashlib,zipfile,time,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;D4=ROOT.parent/'adaptive_feedback_d4_20261126';OLD=ROOT.parent/'dual_verifier_d2_20261125'
sys.path.insert(0,str(D4))
from remote_zip import RemoteZipIO
from prepare_external import HashReader,priority

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args()
 plan=json.loads((ROOT/'protocol/design.json').read_text());entry=next(e for e in plan['new_captures'] if e['dataset']==args.dataset);out=ROOT/'data';start=time.time()
 transforms={seed:json.loads((D4/f'deployment/transform_seed_{seed}.json').read_text()) for seed in plan['seeds']};features=transforms[11]['feature_columns']
 with zipfile.ZipFile(OLD/'data/CICIDS2017_improved.zip') as z:
  with z.open('tuesday.csv') as f:hist=pd.read_csv(f,usecols=lambda c:c.strip() in features,low_memory=False)
 hist.columns=hist.columns.str.strip();hx=hist[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float)
 exclude=set(map(int,pd.util.hash_pandas_object(hx,index=False).to_numpy()));del hist,hx
 for p in (D4/'data/external').glob('*_11.npz'):
  with np.load(p) as a:exclude.update(map(int,a['test_feature_key']))
 print('PREP START',entry,'excluded fingerprints',len(exclude),flush=True)
 inventory=json.loads((D4/'data/remote_inventory.json').read_text());remote=RemoteZipIO();assert remote.etag==inventory['etag'] and remote.length==inventory['archive_bytes']
 seen={};conflicts=set();buf=None;count=duplicates=overlap=badtime=0;families={};lasttick=0
 with zipfile.ZipFile(remote) as archive:
  info=archive.getinfo(entry['member'])
  with archive.open(info) as raw:
   reader=HashReader(raw)
   for chunk in pd.read_csv(io.BufferedReader(reader,1024*1024),chunksize=75000,low_memory=False):
    chunk.columns=chunk.columns.str.strip();rawrows=np.arange(count,count+len(chunk));count+=len(chunk)
    xx=chunk[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float)
    keys=pd.util.hash_pandas_object(xx,index=False).to_numpy();labels=chunk.Label.astype(str);yy=((labels!='BENIGN')&(chunk['Attempted Category']==-1)).to_numpy(np.int8);times=pd.to_datetime(chunk.Timestamp,format='mixed',errors='coerce');valid=times.notna().to_numpy();badtime+=int((~valid).sum());keep=np.zeros(len(chunk),bool)
    for j,(key,y) in enumerate(zip(keys,yy)):
     k=int(key)
     if k in exclude:overlap+=1;continue
     if k in seen:
      duplicates+=1
      if seen[k]!=int(y):conflicts.add(k)
      continue
     seen[k]=int(y);keep[j]=valid[j]
    ix=np.flatnonzero(keep);part=xx.iloc[ix].copy();part['_key']=keys[ix];part['_priority']=priority(keys[ix]);part['_y']=yy[ix];part['_family']=labels.iloc[ix].to_numpy();part['_raw']=rawrows[ix];part['_time']=times.iloc[ix].to_numpy().astype('datetime64[ns]').astype('int64');part['_attempted']=chunk['Attempted Category'].iloc[ix].to_numpy(int)
    for fam,n in part['_family'].value_counts().items():families[fam]=families.get(fam,0)+int(n)
    buf=(part if buf is None else pd.concat([buf,part],ignore_index=True)).nsmallest(36000,'_priority').copy()
    if count-lasttick>=500000:lasttick=count;print('PREP',args.dataset,'rows',count,'unique',len(seen),'rangeMB',round(remote.transferred/1e6),'seconds',round(time.time()-start),flush=True)
   assert reader.bytes==info.file_size
 d=buf[~buf._key.isin(conflicts)].nsmallest(12000,'_priority').sort_values(['_time','_raw'],kind='stable');assert len(d)==12000 and d._key.is_unique and not any(int(k) in exclude for k in d._key)
 xx=d[features].to_numpy();shared=dict(y_test=d._y.to_numpy(int),test_family=d._family.to_numpy(str),test_raw_rows=d._raw.to_numpy(int),test_timestamps=d._time.to_numpy(np.int64),test_attempted=d._attempted.to_numpy(int),test_feature_key=d._key.to_numpy(np.uint64));seedfiles={}
 for seed,t in transforms.items():
  filled=np.where(np.isnan(xx),np.asarray(t['median']),xx);transformed=np.sign(filled)*np.log1p(np.abs(filled));transformed=np.clip((transformed-np.asarray(t['mean']))/np.asarray(t['scale']),*t['clip']);p=out/f'{args.dataset}_{seed}.npz';np.savez_compressed(p,test=transformed,**shared);seedfiles[str(seed)]=dict(path=p.name,sha256=sha(p))
 report=dict(dataset=args.dataset,member=entry['member'],model_scores_computed=False,raw_rows=count,unique_fingerprints=len(seen),duplicates=duplicates,excluded_history_or_old_target=overlap,conflicting_binary_fingerprints=len(conflicts),invalid_timestamp=badtime,source_archive_url=remote.url,archive_etag=remote.etag,archive_bytes=remote.length,member_crc32=f'{info.CRC:08x}',streamed_csv_sha256=reader.digest.hexdigest(),csv_bytes=reader.bytes,https_bytes_transferred=remote.transferred,requests=remote.requests,seconds=time.time()-start,records=len(d),binary_counts={str(k):int(v) for k,v in d._y.value_counts().items()},families={str(k):int(v) for k,v in d._family.value_counts().items()},unique_raw_family_counts=families,times=[str(pd.to_datetime(d._time.min())),str(pd.to_datetime(d._time.max()))],sample_type='Label-blind deterministic fingerprint sample, chronological order',adequate_primary_support=bool((d._y==1).sum()>=30 and (d._y==0).sum()>=30),feature_columns=features,seeds=seedfiles)
 (out/f'{args.dataset}_preparation.json').write_text(json.dumps(report,indent=2));print('PREP DONE',json.dumps({k:v for k,v in report.items() if k not in ['feature_columns','seeds','unique_raw_family_counts']}),flush=True)
if __name__=='__main__':main()
