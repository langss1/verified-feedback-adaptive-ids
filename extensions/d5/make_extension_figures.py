from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'vendor'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
NAMES={'dos_main':'DoS15 (original)','bot_main':'Bot02 (original)','dos16_new':'Hulk16 (new)','ddos20_new':'DDoS20 (new)'}
LABELS={'new_selected':'GDT','matched_verified':'Verified-only','simple_max':'Maximum','frozen_detector':'Frozen','arf_native_matched':'Native ARF'}
def save(fig,path):
 for ext in ['pdf','png','svg']:fig.savefig(str(path)+'.'+ext,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)
def main():
 out=ROOT/'analysis/figures';out.mkdir(exist_ok=True);s=pd.read_csv(ROOT/'analysis/summary.csv');p=pd.read_csv(ROOT/'analysis/paired_capture_comparisons.csv');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'savefig.dpi':220})
 ds=list(NAMES);pol=['new_selected','matched_verified','simple_max','arf_native_matched'];colors=['#162f47','#8c9dac','#aa7548','#4f7b73'];xx=np.arange(4)
 fig,axes=plt.subplots(1,2,figsize=(8.0,3.15))
 for ax,metric,title in [(axes[0],'balanced_accuracy_mean','Balanced accuracy (%)'),(axes[1],'fpr_mean','False-positive rate (%)')]:
  for j,(policy,color) in enumerate(zip(pol,colors)):
   values=[100*float(s[(s.dataset==d)&(s.policy==policy)].iloc[0][metric]) for d in ds];ax.bar(xx+(j-1.5)*.18,values,width=.17,color=color,label=LABELS[policy])
  ax.set_xticks(xx,[n.split(' ')[0] for n in NAMES.values()]);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].set_ylim(0,102);axes[1].set_ylim(bottom=0);axes[0].legend(fontsize=7,ncol=2,loc='lower left');fig.tight_layout();save(fig,out/'capture_performance')
 fig,ax=plt.subplots(figsize=(7.6,3.1));offsets=[-.18,0,.18]
 for j,(policy,color) in enumerate(zip(['matched_verified','simple_max','arf_native_matched'],colors[1:])):
  values=[p[(p.dataset==d)&(p.comparator==policy)].iloc[0] for d in ds];m=np.array([v.balanced_accuracy_difference for v in values]);lo=np.array([v.balanced_accuracy_ci_low for v in values]);hi=np.array([v.balanced_accuracy_ci_high for v in values]);ax.errorbar(m,xx+offsets[j],xerr=np.vstack([m-lo,hi-m]),fmt='o',capsize=3,color=color,label='GDT - '+LABELS[policy],ms=4)
 ax.axvline(0,color='black',lw=.8);ax.axvline(1,color='#777777',lw=.7,ls='--');ax.set_yticks(xx,list(NAMES.values()));ax.invert_yaxis();ax.set_xlabel('Paired BA difference (percentage points), 95% t interval');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=7,loc='best');fig.tight_layout();save(fig,out/'paired_gate_comparisons')
 g=pd.read_csv(ROOT/'analysis/gate_coverage.csv');a=g.groupby('dataset')[['raw_rescues','raw_extra_FP','issued_cut_rescues','issued_cut_extra_FP']].mean().reindex(ds)
 fig,ax=plt.subplots(figsize=(7.6,2.65));ax.bar(xx-.16,a.raw_rescues,width=.3,label='Rescued attacks at raw cut 0.5',color='#162f47');ax.bar(xx+.16,a.raw_extra_FP,width=.3,label='Additional false positives',color='#aa7548');ax.set_xticks(xx,[n.split(' ')[0] for n in NAMES.values()]);ax.set_ylabel('Mean changed decisions / run');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();save(fig,out/'gate_coverage')
 # Compact paper figure isolates the gate attribution instead of compressing
 # its sub-point intervals beside much larger package-versus-ARF differences.
 fig,ax=plt.subplots(figsize=(3.45,1.20));records=[p[(p.dataset==d)&(p.comparator=='matched_verified')].iloc[0] for d in ds];m=np.array([r.balanced_accuracy_difference for r in records]);lo=np.array([r.balanced_accuracy_ci_low for r in records]);hi=np.array([r.balanced_accuracy_ci_high for r in records]);ax.errorbar(m,xx,xerr=np.vstack([m-lo,hi-m]),fmt='o',capsize=2,color='#162f47',ms=3);ax.axvline(0,color='black',lw=.6);ax.axvline(1,color='#777777',ls='--',lw=.6);ax.set_yticks(xx,['DoS15','Bot02','Hulk16*','DDoS20*']);ax.invert_yaxis();ax.set_xlabel('GDT - verified BA (pp); 95% paired interval',fontsize=7);ax.tick_params(labelsize=7);ax.grid(axis='x',alpha=.2);fig.tight_layout(pad=.3);save(fig,out/'paper_gate_increment')
 # Per-role learning curves retain the detector/scorer/retraining evidence.
 import gzip
 D4=ROOT.parent/'adaptive_feedback_d4_20261126';fig,axes=plt.subplots(2,2,figsize=(7.6,4.2),sharex=True,sharey=True)
 for ax,d in zip(axes.flat,ds):
  curves={k:[] for k in ['p1','p2','p']};arrived=[]
  for seed in [11,23,37,51,79]:
   folder=D4/'results/confirmation' if d in ['dos_main','bot_main'] else ROOT/'results';stem=f'{d}_{seed}_delay80_new_selected';tr=dict(np.load(folder/(stem+'.npz')));log=json.loads(gzip.decompress((folder/(stem+'_ledger.json.gz')).read_bytes()));arrived.append([v['training_arrived'] for v in log['progress']])
   for key in curves:curves[key].append([float((tr[key][:end][tr['y'][:end]==1]>=.5).mean()) if (tr['y'][:end]==1).any() else np.nan for end in range(100,12001,100)])
  for key,label,style,color in [('p1','ML1 HGB','-','#162f47'),('p2','ML2 7-NN','--','#aa7548'),('p','GDT',':','#4f7b73')]:
   a=np.asarray(curves[key]);count=np.sum(~np.isnan(a),axis=0);mean=np.divide(np.nansum(a,axis=0),count,out=np.full(a.shape[1],np.nan),where=count>0);ax.plot(np.mean(arrived,axis=0),100*mean,label=label,ls=style,color=color,lw=1.2)
  ax.set_title(NAMES[d],fontsize=9);ax.set_ylim(-2,102);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
 axes[0,0].legend(fontsize=7);axes[1,0].set_xlabel('Arrived training labels');axes[1,1].set_xlabel('Arrived training labels');axes[0,0].set_ylabel('Cumulative recall (%)');axes[1,0].set_ylabel('Cumulative recall (%)');fig.tight_layout();save(fig,out/'all_role_learning_curves')
if __name__=='__main__':main()
