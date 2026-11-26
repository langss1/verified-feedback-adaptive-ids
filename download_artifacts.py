"""Fetch versioned public benchmark artifacts and verify hashes before extraction."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parent
REPOSITORY='langss1/verified-feedback-adaptive-ids'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',action='store_true');ap.add_argument('--from-dir',type=Path);args=ap.parse_args();manifest=json.loads((ROOT/'artifact_manifest.json').read_text());wanted=[a for a in manifest['assets'] if args.results or a['name']!='saved_predictions.zip'];urls={}
 if not args.from_dir:
  req=urllib.request.Request(f'https://api.github.com/repos/{REPOSITORY}/releases/tags/{manifest["release"]}',headers={'User-Agent':'verified-feedback-adaptive-ids'})
  with urllib.request.urlopen(req,timeout=60) as f:release=json.load(f)
  urls={a['name']:a['browser_download_url'] for a in release['assets']}
 cache=ROOT/'.artifact_cache';cache.mkdir(exist_ok=True)
 for a in wanted:
  path=args.from_dir/a['name'] if args.from_dir else cache/a['name']
  if not path.exists():
   assert not args.from_dir,f'Missing local asset {path}'
   part=path.with_suffix(path.suffix+'.part');request=urllib.request.Request(urls[a['name']],headers={'User-Agent':'verified-feedback-adaptive-ids'})
   with urllib.request.urlopen(request,timeout=120) as src,part.open('wb') as dst:
    while True:
     b=src.read(1024*1024)
     if not b:break
     dst.write(b)
   assert part.stat().st_size==a['bytes'] and sha(part)==a['sha256'],a['name'];part.replace(path)
  assert path.stat().st_size==a['bytes'] and sha(path)==a['sha256'],a['name']
  with zipfile.ZipFile(path) as z:
   assert z.testzip() is None
   for member in z.infolist():
    target=(ROOT/member.filename).resolve();assert target.is_relative_to(ROOT) and not Path(member.filename).is_absolute(),member.filename
    if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
    if target.exists():
     assert sha(target)==hashlib.sha256(z.read(member)).hexdigest(),f'Refusing to replace changed file {target}; use a new checkout'
     continue
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(member) as src,target.open('xb') as dst:
     while True:
      b=src.read(1024*1024)
      if not b:break
      dst.write(b)
  print('VERIFIED AND EXTRACTED',a['name'],flush=True)
if __name__=='__main__':main()
