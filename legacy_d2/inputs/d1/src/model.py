import numpy as np
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,f1_score,average_precision_score,brier_score_loss
from scipy.stats import t as student_t

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-35,35)))
def security_loss(y,h,fn_weight=4):
 y=np.asarray(y);h=np.asarray(h);return np.where((y==1)&(h==0),float(fn_weight),np.where((y==0)&(h==1),1.,0.))
class Linear:
 def __init__(self,w,b):self.w=np.asarray(w,float).copy();self.b=float(b)
 def copy(self):return Linear(self.w,self.b)
 def p(self,x):return sigmoid(np.asarray(x)@self.w+self.b)
 def update(self,x,y,weights=None,epochs=4):
  if len(y)==0:return 0
  weight=np.ones(len(y)) if weights is None else np.asarray(weights,float);weight=weight/max(weight.mean(),1e-8)
  for _ in range(epochs):
   e=(self.p(x)-y)*weight;self.w-=.08*(x.T@e/len(y)+.001*self.w);self.b-=.08*e.mean()
  return len(y)*epochs
def metrics(y,p):
 y=np.asarray(y);p=np.clip(p,0,1);h=p>=.5;tn,fp,fn,tp=confusion_matrix(y,h,labels=[0,1]).ravel()
 return dict(n=len(y),balanced_accuracy=float(balanced_accuracy_score(y,h)),macro_f1=float(f1_score(y,h,average='macro',zero_division=0)),security_loss=float(security_loss(y,h).mean()),fnr=float(fn/max(1,fn+tp)),fpr=float(fp/max(1,fp+tn)),pr_auc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn))
def original_guard(old,cand,x,y,margin=.03):
 if len(y)<20:return False
 h0=old.p(x)>=.5;h1=cand.p(x)>=.5
 for c in [0,1]:
  ix=y==c
  if ix.sum()<8:return False
  d=security_loss(y[ix],h1[ix])-security_loss(y[ix],h0[ix])
  if d.mean()+student_t.ppf(.95,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))>margin:return False
 return True
def class_risks(model,x,y):
 h=model.p(x)>=.5;return [float(security_loss(y[y==c],h[y==c]).mean()) for c in [0,1]]
