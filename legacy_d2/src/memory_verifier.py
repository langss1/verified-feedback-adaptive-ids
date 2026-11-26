"""Existing distance-weighted nearest-neighbor principle, with sample trust weights."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
class MemoryVerifier:
 def fit(self,x,y,sample_weight=None):
  self.x=np.asarray(x).copy();self.y=np.asarray(y,int).copy();self.weight=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,float).copy();self.index=NearestNeighbors(n_neighbors=min(7,len(y)),algorithm='brute',metric='manhattan',n_jobs=1).fit(x);return self
 def predict_proba(self,x):
  d,ix=self.index.kneighbors(x);w=self.weight[ix]/np.maximum(d,1e-6);p=(w*self.y[ix]).sum(axis=1)/w.sum(axis=1);return np.c_[1-p,p]
 def evidence(self,x):
  d,ix=self.index.kneighbors(x);counts=np.zeros((len(x),2),int);near=np.full((len(x),2),np.inf)
  for i in range(len(x)):
   for c in [0,1]:
    mask=(self.weight[ix[i]]>1)&(self.y[ix[i]]==c)
    if mask.any():counts[i,c]=len(np.unique(self.x[ix[i][mask]],axis=0));near[i,c]=float(d[i][mask].min())
  return counts,near,d[:,0]
