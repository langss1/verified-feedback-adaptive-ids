"""Stream two pinned corrected2018 members, without whole-archive download.
No classifier is fitted/scored on external records here.
"""
from pathlib import Path
import io,json,hashlib,zipfile,time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from remote_zip import RemoteZipIO
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'dual_verifier_d2_20261125';SALT=np.uint64(0x9e3779b97f4a7c15)
class HashReader(io.RawIOBase):
 def __init__(self,f):self.f=f;self.digest=hashlib.sha256();self.bytes=0
 def readable(self):return True
 def readinto(self,b):
  data=self.f.read(len(b));b[:len(data)]=data;self.digest.update(data);self.bytes+=len(data);return len(data)
def priority(keys):
 v=np.asarray(keys,np.uint64)^SALT;v=(v^(v>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);v=(v^(v>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);return v^(v>>np.uint64(31))
def shrink(old,new):
 if old is None:both=new
 else:both=pd.concat([old,new],ignore_index=True)
 return both.nsmallest(24000,'_priority').copy()
def main():
 start=time.time();out=ROOT/'data/external';out.mkdir(exist_ok=True);meta=json.loads((OLD/'data/confirmation/preparation.json').read_text());features=meta['features'];report=dict(preparation_only=True,model_scores_computed=False,cohorts={},dedup='Float64 numeric predictor fingerprint; keepfirstarchive row. Conflicting binary-label fingerprints removed after entire member scan; oversampled labelblind reservoirs permit replacement. Final chronological ordering. History overlap excluded.',streaming={})
 with zipfile.ZipFile(OLD/'data/CICIDS2017_improved.zip') as z:
  with z.open('tuesday.csv') as f:hist=pd.read_csv(f,low_memory=False)
 hist.columns=hist.columns.str.strip();hx=hist[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float);history_keys=set(pd.util.hash_pandas_object(hx,index=False).to_numpy());transforms={}
 for seed in [11,23,37,51,79]:
  ids=json.loads((OLD/f'data/confirmation/wednesday_{seed}_rows.json').read_text())['base'];imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(hx.iloc[ids]);a=imp.transform(hx.iloc[ids]);a=np.sign(a)*np.log1p(np.abs(a));sc=StandardScaler().fit(a);expected=dict(np.load(ROOT/f'inputs/history_{seed}.npz'));assert np.allclose(np.clip(sc.transform(a),-10,10),expected['base'],atol=1e-12,rtol=0);transforms[seed]=(imp,sc)
 del hist,hx
 remote=RemoteZipIO();inventory=json.loads((ROOT/'data/remote_inventory.json').read_text());assert remote.etag==inventory['etag'] and remote.length==inventory['archive_bytes']
 with zipfile.ZipFile(remote) as archive:
  for member,prefix in [('Friday-23-02-2018.csv','web'),('Friday-02-03-2018.csv','bot')]:
   seen={};conflicts=set();mainbuf=benignbuf=None;rarebuf=[];count=duplicate=overlap=badtime=0;lasttick=0;fullfamilies={};info=archive.getinfo(member)
   with archive.open(info) as raw:
    reader=HashReader(raw)
    for chunk in pd.read_csv(io.BufferedReader(reader,1024*1024),chunksize=75000,low_memory=False):
     chunk.columns=chunk.columns.str.strip();rawrows=np.arange(count,count+len(chunk));count+=len(chunk);xx=chunk[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float);keys=pd.util.hash_pandas_object(xx,index=False).to_numpy();labels=chunk.Label.astype(str);yy=((labels!='BENIGN')&(chunk['Attempted Category']==-1)).to_numpy(np.int8);times=pd.to_datetime(chunk.Timestamp,format='mixed',errors='coerce');valid=times.notna().to_numpy();badtime+=int((~valid).sum());keep=np.zeros(len(chunk),bool)
     for j,(key,y) in enumerate(zip(keys,yy)):
      k=int(key)
      if k in history_keys:overlap+=1;continue
      if k in seen:
       duplicate+=1
       if seen[k]!=int(y):conflicts.add(k)
       continue
      seen[k]=int(y);keep[j]=valid[j]
     ix=np.flatnonzero(keep);part=xx.iloc[ix].copy();part['_key']=keys[ix];part['_priority']=priority(keys[ix]);part['_y']=yy[ix];part['_family']=labels.iloc[ix].to_numpy();part['_raw']=rawrows[ix];part['_time']=times.iloc[ix].to_numpy().astype('datetime64[ns]').astype('int64');part['_attempted']=chunk['Attempted Category'].iloc[ix].to_numpy(int)
     for family,n in part['_family'].value_counts().items():fullfamilies[family]=fullfamilies.get(family,0)+int(n)
     mainbuf=shrink(mainbuf,part)
     if prefix=='web':
      benignbuf=shrink(benignbuf,part[part._y==0]);rare=part[(part._y==1)&part._family.str.contains('web',case=False)]
      if len(rare):rarebuf.append(rare)
     if count-lasttick>=500000:lasttick=count;print('PREP',member,'rawrows',count,'unique',len(seen),'rangeMB',round(remote.transferred/1e6),'seconds',round(time.time()-start),flush=True)
   assert reader.bytes==info.file_size
   def clean(d,k=None):
    d=d[~d._key.isin(conflicts)]
    if k is not None:d=d.nsmallest(k,'_priority')
    return d.sort_values(['_time','_raw'],kind='stable')
   cohorts={prefix+'_main':clean(mainbuf,12000)}
   if prefix=='web':cohorts['web_challenge']=pd.concat([clean(benignbuf,12000),clean(pd.concat(rarebuf,ignore_index=True))],ignore_index=True).sort_values(['_time','_raw'],kind='stable') if rarebuf else clean(benignbuf,12000)
   report['streaming'][member]=dict(raw_rows=count,unique_fingerprints=len(seen),duplicates=duplicate,history_overlap=overlap,conflicting_binary_fingerprints=len(conflicts),invalid_timestamp=badtime,crc32=f'{info.CRC:08x}',streamed_csv_sha256=reader.digest.hexdigest(),csv_bytes=reader.bytes,compressed_member_bytes=info.compress_size,unique_raw_family_counts=fullfamilies)
   for name,d in cohorts.items():
    xx=d[features].to_numpy();shared=dict(y_test=d._y.to_numpy(int),test_family=d._family.to_numpy(str),test_raw_rows=d._raw.to_numpy(int),test_timestamps=d._time.to_numpy(np.int64),test_attempted=d._attempted.to_numpy(int),test_feature_key=d._key.to_numpy(np.uint64));counts=dict(pd.Series(shared['y_test']).value_counts().items());report['cohorts'][name]=dict(records=len(d),binary_counts={str(k):int(v) for k,v in counts.items()},families=d.groupby(['_family','_y']).size().to_dict() if False else d._family.value_counts().to_dict(),sample_type='Case-control webattack census plus12000benign; NOT naturalprevalence' if name.endswith('challenge') else 'Label-blind12000distinctfeature sample',times=[str(pd.to_datetime(d._time.min())),str(pd.to_datetime(d._time.max()))],adequate_primary_attack_support=bool((d._y==1).sum()>=30),seeds={})
    assert len(np.unique(d._key))==len(d)
    for seed,(imp,sc) in transforms.items():
     transformed=imp.transform(xx);transformed=np.sign(transformed)*np.log1p(np.abs(transformed));transformed=np.clip(sc.transform(transformed),-10,10);p=out/f'{name}_{seed}.npz';np.savez_compressed(p,test=transformed,**shared);report['cohorts'][name]['seeds'][str(seed)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    print('COHORT',name,json.dumps({k:v for k,v in report['cohorts'][name].items() if k!='seeds'}),flush=True)
   (out/'preparation.json').write_text(json.dumps(report,indent=2));del seen,conflicts,mainbuf,benignbuf,rarebuf,cohorts
 report.update(url=remote.url,etag=remote.etag,archive_bytes=remote.length,https_bytes_transferred=remote.transferred,requests=remote.requests,seconds=time.time()-start,feature_columns=features);(out/'preparation.json').write_text(json.dumps(report,indent=2));print('PREPARATION COMPLETE; no external model scores')
if __name__=='__main__':main()
