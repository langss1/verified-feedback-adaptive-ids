"""Read public ZIP ranges with pinned ETag; never downloads the full10.4GB archive."""
import io,urllib.request,zipfile,json,hashlib
from pathlib import Path
URL='https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip'
class RemoteZipIO(io.RawIOBase):
 def __init__(self,url=URL,block=8*1024*1024):
  self.url=url;self.block=block;self.pos=0;self.cache={};self.transferred=0;self.requests=0
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:self.length=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag');self.modified=r.headers.get('Last-Modified')
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
  if self.pos<0:raise ValueError('Negative seek')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.length-self.pos
  if n>64*1024*1024:raise ValueError('Explicit bounded range reads required')
  end=min(self.length,self.pos+n);pieces=[]
  while self.pos<end:
   key=self.pos//self.block
   if key not in self.cache:
    lo=key*self.block;hi=min(self.length,lo+self.block)-1;headers={'Range':f'bytes={lo}-{hi}','If-Match':self.etag}
    for attempt in range(3):
     try:
      with urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=45) as r:
       assert r.status==206 and r.headers.get('Content-Range')==f'bytes {lo}-{hi}/{self.length}';data=r.read(hi-lo+1);assert len(data)==hi-lo+1
      break
     except Exception:
      if attempt==2:raise
    if len(self.cache)>=3:self.cache.pop(next(iter(self.cache)))
    self.cache[key]=data;self.transferred+=len(data);self.requests+=1
   off=self.pos-key*self.block;take=min(end-self.pos,len(self.cache[key])-off);pieces.append(self.cache[key][off:off+take]);self.pos+=take
  return b''.join(pieces)
def inventory():
 r=RemoteZipIO()
 with zipfile.ZipFile(r) as z:files=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size,crc32=f'{i.CRC:08x}',method=i.compress_type) for i in z.infolist()]
 obj=dict(url=URL,archive_bytes=r.length,etag=r.etag,last_modified=r.modified,members=files,index_bytes_transferred=r.transferred)
 root=Path(__file__).resolve().parent;(root/'data').mkdir(exist_ok=True);(root/'data/remote_inventory.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj,indent=2))
if __name__=='__main__':inventory()
