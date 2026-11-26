from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scipy.stats import t
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'analysis/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'figure.dpi':150})
C=['#14777c','#cb7950','#657e9d','#8e617f','#779b57','#999999']
NAMES={'new_selected':'New two-model system','matched_verified':'Matched verified-only','simple_max':'Simple max control','frozen_detector':'Frozen detector','budget_full_history':'Full history / 5% labels','old_rvea_full':'Previous full RVEA','oracle_100pct':'Full supervision / 100%'}
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def ci(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)];m=float(v.mean());e=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.;return m,e
def methods():
 fig,ax=plt.subplots(figsize=(7.5,3.6));ax.set_xlim(0,10.4);ax.set_ylim(0,5.7);ax.axis('off')
 boxes=[(.1,4.3,1.8,.8,'Historical BASE\nfixed transform'),(.1,2.9,1.8,.8,'Target flow features\nno target gold'),(2.6,3.8,2.,.9,'ML1 detector\nold replay + verified\nregular retraining'),(2.6,2.2,2.,.9,'ML2 inspector\nweighted 7-NN\nold + verified memory'),(5.8,3.,1.9,1.2,'Verified-support rule\nheld-audit threshold\nissued decision'),(8.3,3.2,1.9,.8,'Saved prediction\nper-model scores'),(.1,.5,2.8,1.,'Request after 100 observed\n4 training + 1 held audit\nCSV human / simulator'),(4.1,.5,3.,1.,'Arrived-label ledger\ncorrections / versions\nheld audit never trains')]
 for x,y,w,h,label in boxes:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',edgecolor=C[0],facecolor='#eaf3f4'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.6)
 def arrow(a,b,label=None):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,linewidth=1,color='#465968'))
  if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=6.8,color='#465968')
 arrow((1.96,4.7),(2.52,4.5));arrow((1.96,3.6),(2.52,4.0));arrow((1.96,3.0),(2.52,2.7));arrow((4.67,4.25),(5.73,3.9));arrow((4.67,2.65),(5.73,3.35));arrow((7.77,3.6),(8.22,3.6));arrow((1.1,2.8),(1.1,1.6));arrow((2.98,1.),(4.02,1.));ax.text(3.5,1.75,'delay +1 / +80',ha='center',fontsize=6.8,color='#465968');arrow((6.6,1.6),(6.6,2.9),'held audit')
 arrow((2.53,2.5),(2.1,1.58))
 ax.plot([4.2,5.1,5.1,3.6],[1.55,1.85,5.05,5.05],linestyle='--',color='#465968',linewidth=1);arrow((3.6,5.05),(3.6,4.78));arrow((4.3,1.6),(4.3,2.13));ax.text(.1,5.45,'Two classifier roles with explicit feedback and correction history',fontsize=10.5,weight='bold');ax.text(.1,.05,'Previous-model snapshots measure update effects on the NEXT block; they are not extra classifier roles.',fontsize=7.3)
 save(fig,'architecture')
 fig,ax=plt.subplots(figsize=(7.5,1.7));ax.axis('off');labels=['Observe flows 0-99\npredict before labels','Request 5 reviews\nat block end t=100','Feedback at t=180\ndelay80 condition','Refit at t=199\npredict from t=200']
 for i,label in enumerate(labels):
  x=.02+i*.25;ax.add_patch(FancyBboxPatch((x,.38),.205,.42,boxstyle='round,pad=.01',transform=ax.transAxes,edgecolor=C[0],facecolor='#eaf3f4'));ax.text(x+.1025,.59,label,transform=ax.transAxes,ha='center',va='center',fontsize=7.7)
  if i<3:ax.annotate('',xy=(x+.243,.59),xytext=(x+.212,.59),xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','color':'#465968'})
 ax.text(.02,.08,'Protocol schematic. Actual feedback, fits and corrections are saved as event-indexed records.',transform=ax.transAxes,fontsize=8);save(fig,'feedback_timing')
def main():
 methods()
 if not (ROOT/'analysis/verdict.json').exists():return
 verdict=json.loads((ROOT/'analysis/verdict.json').read_text());lock=json.loads((ROOT/'protocol/confirmation_lock.json').read_text());df=pd.read_csv(ROOT/'analysis/confirmation_all_runs.csv');seed=pd.read_csv(ROOT/'analysis/confirmation_seed_means.csv');roles=lock['roles'];ps=list(dict.fromkeys(['new_selected',roles['matched_verified'],roles['simple_max'],'old_rvea_full','budget_full_history',roles['frozen_detector']]))
 fig,axes=plt.subplots(1,2,figsize=(7.5,3.3),sharey=True)
 for ax,ds in zip(axes,['dos_main','bot_main']):
  for j,p in enumerate(ps):
   vals=100*seed[(seed.dataset==ds)&(seed.policy==p)].balanced_accuracy;m,e=ci(vals);ax.barh(j,m,color=C[j%len(C)],height=.65);ax.errorbar(m,j,xerr=e,fmt='none',ecolor='#263743',capsize=2,linewidth=.8)
  if ax==axes[0]:ax.set_yticks(range(len(ps)),[NAMES.get(p,p) for p in ps]);ax.invert_yaxis()
  else:ax.tick_params(axis='y',labelleft=False)
  ax.set_xlim(0,100);ax.set_title('DoS capture' if ds=='dos_main' else 'Botnet capture');ax.set_xlabel('Balanced accuracy (%)');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 fig.suptitle('Mean four conditions within seed; descriptive five-seed intervals',fontsize=9);fig.tight_layout();save(fig,'primary_benchmark')
 curves=pd.read_csv(ROOT/'analysis/learning_curves.csv');fig,axes=plt.subplots(2,2,figsize=(7.5,4.4),sharex=True)
 for k,ds in enumerate(['dos_main','bot_main']):
  z=curves[(curves.dataset==ds)&(curves.scenario=='delay80')&(curves.policy=='new_selected')].groupby('end').mean(numeric_only=True)
  for j,role in enumerate(['ML1','ML2','system']):
   axes[0,k].plot(z.training_received,100*z[role+'_balanced_accuracy'],marker='.',color=C[j],label=role+(' (raw .5)' if role!='system' else ' (decision rule)'));axes[1,k].plot(z.training_received,100*z[role+'_recall'],marker='.',color=C[j])
  f=curves[(curves.dataset==ds)&(curves.scenario=='delay80')&(curves.policy==roles['frozen_detector'])].groupby('end').mean(numeric_only=True);axes[0,k].plot(f.training_received,100*f.system_balanced_accuracy,color=C[5],linestyle='--',label='Frozen detector');axes[0,k].set_title(ds.replace('_main','').title());axes[0,k].set_ylabel('Cumulative BA (%)');axes[1,k].set_ylabel('Cumulative attack recall (%)');axes[1,k].set_xlabel('Arrived training labels');axes[0,k].set_ylim(-2,102);axes[1,k].set_ylim(-2,102)
  for ax in axes[:,k]:ax.grid(alpha=.18)
 axes[0,0].legend(fontsize=7,loc='lower right');fig.tight_layout();save(fig,'learning_progress')
 source=pd.read_csv(ROOT/'analysis/source_retention_curves.csv');fig,axes=plt.subplots(1,2,figsize=(7.4,2.7),sharey=True)
 for ax,ds in zip(axes,['dos_main','bot_main']):
  z=source[(source.dataset==ds)&(source.scenario=='delay80')&(source.policy=='new_selected')].groupby('t').mean(numeric_only=True);base=df[(df.dataset==ds)&(df.scenario=='delay80')&(df.policy=='new_selected')];initial=pd.DataFrame([dict(training_arrived=0,source1_BA=base.initial_source1_BA.mean(),source2_BA=base.initial_source2_BA.mean(),source_BA=base.initial_source1_BA.mean())],index=[-1]);z=pd.concat([initial,z]).sort_index()
  for j,key in enumerate(['source1_BA','source2_BA','source_BA']):ax.plot(z.training_arrived,100*z[key],marker='.',color=C[j],label=['ML1','ML2','System'][j])
  ax.set_title(ds.replace('_main','').title());ax.set_xlabel('Arrived training labels');ax.grid(alpha=.18)
 axes[0].set_ylabel('Unused historical probe BA (%)');axes[0].legend(fontsize=8);fig.tight_layout();save(fig,'retention')
 fam=pd.read_csv(ROOT/'analysis/family_scores.csv');z=fam[(fam.policy=='new_selected')&(fam.dataset!='web_main')];order=z.groupby(['dataset','family']).support.first().sort_values().index;fig,ax=plt.subplots(figsize=(7.4,3.25))
 for j,p in enumerate(['new_selected',roles['matched_verified'],'old_rvea_full']):
  for i,(ds,family) in enumerate(order):
   vals=fam[(fam.dataset==ds)&(fam.family==family)&(fam.policy==p)].groupby('seed').recall.mean();m,e=ci(100*vals);ax.errorbar(m,i+(j-1)*.17,xerr=e,fmt='o',markersize=3.5,capsize=2,color=C[j],label=NAMES[p] if i==0 else None)
 labels=[f'{family} (n={int(z[(z.dataset==ds)&(z.family==family)].support.iloc[0])})' for ds,family in order];ax.set_yticks(range(len(order)),labels);ax.set_xlim(-5,105);ax.set_xlabel('Attack recall (%); unique flow support shown');ax.grid(axis='x',alpha=.18);ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3);fig.tight_layout();save(fig,'family_recall')
 effects=pd.read_csv(ROOT/'analysis/next_block_update_effect.csv');fig,axes=plt.subplots(1,2,figsize=(7.3,2.7),sharey=True)
 for ax,ds in zip(axes,['dos_main','bot_main']):
  z=effects[(effects.dataset==ds)&(effects.policy=='new_selected')].groupby(['seed','role']).loss_change.mean().reset_index()
  for j,role in enumerate(['ML1','ML2']):
   m,e=ci(z[z.role==role].loss_change);ax.bar(j,m,color=C[j]);ax.errorbar(j,m,yerr=e,fmt='none',color='#263743',capsize=3)
  ax.set_xticks([0,1],['ML1 update','ML2 update']);ax.axhline(0,color='#999999',linewidth=.8);ax.set_title(ds.replace('_main','').title());ax.set_ylabel('Next-block loss: updated - prior')
 fig.tight_layout();save(fig,'next_block_effect')
 fig,ax=plt.subplots(figsize=(3.45,2.2));width=.28
 for j,role in enumerate(['ML1','ML2']):
  for i,ds in enumerate(['dos_main','bot_main']):
   z=effects[(effects.dataset==ds)&(effects.policy=='new_selected')&(effects.role==role)].groupby('seed').loss_change.mean();m,e=ci(z);x=i+(j-.5)*width;ax.bar(x,m,width=width,color=C[j],label=role+' update' if i==0 else None);ax.errorbar(x,m,yerr=e,fmt='none',color='#263743',capsize=2,linewidth=.8)
 ax.set_xticks([0,1],['DoS','Botnet']);ax.set_ylabel('Next-block loss change',fontsize=8);ax.axhline(0,color='#999999',linewidth=.8);ax.tick_params(labelsize=8);ax.legend(fontsize=7,loc='best');fig.tight_layout();save(fig,'next_block_effect_column')
 fig,axes=plt.subplots(1,2,figsize=(7.3,2.7),sharey=True)
 for ax,ds in zip(axes,['dos_main','bot_main']):
  z=df[(df.dataset==ds)&(df.scenario=='clean1')];policies=list(dict.fromkeys(['new_selected','budget_full_history','oracle_100pct']))
  for j,p in enumerate(policies):m,e=ci(100*z[z.policy==p].balanced_accuracy);ax.bar(j,m,color=C[j]);ax.errorbar(j,m,yerr=e,fmt='none',ecolor='#263743',capsize=3)
  ax.set_xticks(range(3),['Two ML / 5%','Full history / 5%','Full labels / 100%'],rotation=12,fontsize=8);ax.set_ylim(0,100);ax.set_title(ds.replace('_main','').title());ax.set_ylabel('BA (%) - clean1 condition')
 fig.tight_layout();save(fig,'full_retrain_reference')
 fig,ax=plt.subplots(figsize=(3.45,2.5));policies=['new_selected','budget_full_history','oracle_100pct'];labels=['Selected / 5%','Full history / 5%','Full labels / 100%'];width=.23
 for j,p in enumerate(policies):
  for i,ds in enumerate(['dos_main','bot_main']):
   z=df[(df.dataset==ds)&(df.scenario=='clean1')&(df.policy==p)];m,e=ci(100*z.balanced_accuracy);x=i+(j-1)*width;ax.bar(x,m,width=width,color=C[j],label=labels[j] if i==0 else None);ax.errorbar(x,m,yerr=e,fmt='none',color='#263743',capsize=2,linewidth=.8)
 ax.set_xticks([0,1],['DoS','Botnet']);ax.set_ylabel('BA (%) - clean1',fontsize=8);ax.set_ylim(0,100);ax.tick_params(labelsize=8);ax.legend(fontsize=7,loc='lower right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'full_retrain_reference_column')
if __name__=='__main__':main()
