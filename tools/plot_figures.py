"""Revision figures from frozen results and descriptive reuse of saved group metrics."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
E=R/'results/explanatory_suite_v1_1';A=R/'results/external_expansion_v2_2_analysis_v1'
F=R/'generated_figures';D=F/'source_data';T=F/'tables'
for p in [F,D,T]:p.mkdir(exist_ok=True,parents=True)
used={}
def csv(p):
    used[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def save(fig,name):
    from matplotlib.text import Text
    for item in fig.findobj(match=Text):
        assert item.get_fontfamily()==['Times New Roman'], item.get_fontfamily()
        assert float(item.get_fontsize()).is_integer(), item.get_fontsize()
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',pad_inches=.10,dpi=300)
    plt.close(fig)
def tab(name,cols,rows,align):
    s='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(cols)+r'\\'+'\n\\midrule\n'
    s+='\n'.join(' & '.join(map(str,x))+r'\\' for x in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (T/f'{name}.tex').write_text(s,encoding='utf-8')
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.labelsize':11,'axes.titlesize':12,
    'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'axes.linewidth':.7,'axes.titlepad':12,'xtick.major.size':3,'ytick.major.size':3})
teal='#167F79';ochre='#BC7B3D';ink='#284452';grey='#A6B5BE'
models=['mean_feature','probability_mlp','deepsets','set_transformer']
names=dict(zip(models,['Mean-feature','Probability MLP','Deep Sets','Set Transformer']))
short=dict(zip(models,['MF','PMLP','DS','ST']))
order=[('dinov2',m) for m in models]+[('resnet50',m) for m in models[1:3]]
labels=[('DINOv2 / ' if b=='dinov2' else 'ResNet-50 / ')+short[m] for b,m in order]
gain=csv(R/'figure_source_data/gain_cross_encoder.csv')
fixed=csv(R/'figure_source_data/fixed_five_seed_curves.csv')
paired=csv(E/'paired_tests_holm16.csv');curves=csv(E/'real_mean_curves.csv')
effects=csv(A/'paired_effects.csv')
fig,(a,b)=plt.subplots(1,2,figsize=(8.5,3.5),gridspec_kw={'width_ratios':[1.1,1]})
for i,row in enumerate(gain.itertuples()):
    a.errorbar(100*row.delta,i,xerr=[[100*(row.delta-row.lo)],[100*(row.hi-row.delta)]],fmt='o' if row.backbone=='dinov2' else 's',color=teal if row.backbone=='dinov2' else ochre,capsize=3,ms=5)
a.set_yticks(range(6),labels);a.invert_yaxis();a.set_xlim(0,10)
a.set_xlabel('Accuracy gain (pp)');a.set_title('a  Second-photo gain',loc='left')
for col,mark,lab in [(teal,'o','DINOv2'),(ochre,'s','ResNet-50')]:a.plot([],[],marker=mark,color=col,ls='none',label=lab)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2,fontsize=9)
cols=['#617588',teal,ochre,'#81699A']
for i,m in enumerate(models):
    q=fixed[fixed.model==m].groupby('budget').accuracy.agg(['mean','min','max'])*100
    b.fill_between(q.index,q['min'],q['max'],color=cols[i],alpha=.14)
    b.plot(q.index,q['mean'],marker=['o','s','^','D'][i],color=cols[i],label=short[m],lw=1.4,ms=4)
b.set_ylim(0,80);b.set_yticks([0,20,40,60,80]);b.set_xticks(range(1,6));b.set_xlabel('Acquired photographs');b.set_ylabel('Species-macro accuracy (%)')
b.set_title('b  Fixed five-photo support',loc='left');b.legend(ncol=2,loc='lower right',fontsize=9)
b.text(.04,.89,'108 observations',transform=b.transAxes,fontsize=10,color=ink)
fig.tight_layout(w_pad=1.8);save(fig,'figure_gain')

repeat=csv(R/'figure_source_data/grouping.csv').query("method=='native' and budget==2")
em=repeat.query("condition=='difficulty_matched'").groupby(['backbone','model']).mean(numeric_only=True)
attr=csv(E/'matching_attrition.csv');counts=attr.groupby('n').included.agg(['size','sum'])
fig,(a,b)=plt.subplots(1,2,figsize=(9.2,4.9),gridspec_kw={'width_ratios':[2.5,1]})
labs=[]
for i,(enc,m) in enumerate(order):
    labs.append(labels[i]+f"\nNatural {em.loc[(enc,m),'accuracy_real']*100:.1f}%")
    for cond,dy,col,marker in [('unrestricted_common',-.14,ochre,'o'),('difficulty_matched',.14,teal,'s')]:
        v=repeat.query('backbone==@enc and model==@m and condition==@cond').accuracy_delta.to_numpy()*100
        a.scatter(v,i+dy+np.linspace(-.04,.04,len(v)),color=col,alpha=.4,s=9)
        a.plot([v.min(),v.max()],[i+dy]*2,color=col,lw=1)
        a.scatter(v.mean(),i+dy,color=col,marker=marker,s=37)
        a.text(13.8,i+dy,f'{v.mean():+.2f}',va='center',fontsize=9,color=col)
a.set_yticks(range(6),labs);a.invert_yaxis();a.axvline(0,color=grey,lw=1);a.set_xlim(-.3,16);a.set_xticks([0,4,8,12]);a.set_xlabel('Regrouped minus natural accuracy (pp)')
a.set_title('a  Regrouping contrast',loc='left');a.text(.69,1.005,'Mean',transform=a.transAxes,fontsize=9)
for col,mark,lab in [(ochre,'o','Unrestricted'),(teal,'s','Difficulty matched')]:a.plot([],[],marker=mark,color=col,label=lab)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=9)
pct=counts['sum']/counts['size']*100;b.bar(counts.index,pct,color=teal,width=.55)
for n,v in pct.items():b.text(n,v+3,f'{v:.1f}%\n{int(counts.loc[n,"sum"])}/{int(counts.loc[n,"size"])}',ha='center',fontsize=10)
b.set_ylim(0,100);b.set_xticks([2,3,4,5]);b.set_ylabel('Groups retained (%)');b.set_xlabel('Photos per group');b.set_title('b  Matching retention',loc='left')
fig.tight_layout(w_pad=1.5);save(fig,'figure_grouping')

fig=plt.figure(figsize=(8.6,6.1));gs=fig.add_gridspec(2,2,height_ratios=[1.2,1]);axes=[fig.add_subplot(gs[0,i]) for i in range(2)];bot=fig.add_subplot(gs[1,:])
for ax,metric,title in zip(axes,['joint_wrong_pair','same_wrong_pair'],['a  Jointly wrong','b  Same wrong class']):
    for i,key in enumerate(order):
        q=em.loc[key];x=q[metric+'_real']*100;y=q[metric+'_control']*100
        ax.plot([x,y],[i,i],color=grey,lw=1.5);ax.scatter(x,i,color=ochre,s=28);ax.scatter(y,i,color=teal,marker='s',s=26)
    ax.set_yticks(range(6),labels if ax==axes[0] else []);ax.invert_yaxis();ax.set_xlim(0,70 if metric=='joint_wrong_pair' else 26);ax.set_xlabel('Unconditional rate (%)');ax.set_title(title,loc='left',fontsize=11)
q=em.loc[('dinov2','deepsets')]
for j,suffix in enumerate(['real','control']):
    J=q['joint_wrong_pair_'+suffix];D0=q['any_single_correct_set_wrong_'+suffix];RR=q['all_single_wrong_set_correct_'+suffix]
    v=100*np.array([J-RR,D0,RR,1-J-D0]);yy=np.arange(4)+(j-.5)*.29
    bot.barh(yy,v,height=.26,color=[ochre,teal][j],label=['Natural','Difficulty matched'][j])
    for y,x in zip(yy,v):bot.text(x+.6,y,f'{x:.2f}%',va='center',fontsize=10)
bot.set_yticks(range(4),['Both wrong / set wrong','Any correct / set wrong','Both wrong / set correct','Any correct / set correct']);bot.invert_yaxis();bot.set_xlim(0,86);bot.set_xlabel('Two-photo outcome (%)');bot.set_title('c  Deep Sets outcomes',loc='left');bot.legend(loc='upper right',fontsize=9)
fig.tight_layout(h_pad=1.8,w_pad=2);save(fig,'figure_errors')

fig,(a,b)=plt.subplots(1,2,figsize=(8.7,3.9),gridspec_kw={'width_ratios':[1.35,1]})
sel=[('dinov2','deepsets'),('dinov2','set_transformer'),('resnet50','deepsets')]
for i,(enc,m) in enumerate(sel):
    for k,dy,col,marker in [(2,-.12,teal,'o'),(3,.12,ochre,'s')]:
        q=paired.query("backbone==@enc and model==@m and contrast=='native_minus_pool' and budget==@k").iloc[0]
        a.errorbar(q.delta*100,i+dy,xerr=[[100*(q.delta-q.lower95)],[100*(q.upper95-q.delta)]],fmt=marker,color=col,capsize=3,ms=5)
a.set_yticks(range(3),['DINOv2 / Deep Sets','DINOv2 / Set Transformer','ResNet-50 / Deep Sets']);a.invert_yaxis();a.axvline(0,color=grey);a.set_xlim(-3.3,3.6);a.set_xlabel('Native minus probability pool (pp)');a.set_title('a  Native versus pool',loc='left')
for col,mark,lab in [(teal,'o','Two photos'),(ochre,'s','Three photos')]:a.plot([],[],color=col,marker=mark,label=lab)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,fontsize=9)
q=curves.query("backbone=='dinov2' and model=='deepsets' and budget==2").set_index('method')
for j,method in enumerate(['native','probability_pool']):
    vals=100*q.loc[method,['all_single_wrong_set_correct','any_single_correct_set_wrong']].to_numpy(dtype=float)
    xx=np.arange(2)+(j-.5)*.36;b.bar(xx,vals,width=.32,color=[teal,ochre][j],label=['Native','Singleton-probability pool'][j])
    for x,v in zip(xx,vals):b.text(x,v+.12,f'{v:.3f}',ha='center',fontsize=9)
b.set_xticks([0,1],['Rescue','Destruction']);b.set_ylim(0,10.8);b.set_ylabel('Two-photo outcome (%)');b.set_title('b  DINOv2 Deep Sets',loc='left')
b.text(.5,.95,'Net gain +0.145 pp',transform=b.transAxes,ha='center',fontsize=10,fontweight='bold')
b.legend(loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=9)

fig.tight_layout(w_pad=1.6);save(fig,'figure_pooling')

# Supplementary descriptions: existing policies and natural retained/excluded groups.
pol=effects.query("analysis_branch=='primary' and mode=='full8' and before==1 and after==2")
support=csv(R/'figure_source_data/matched_unmatched_natural_descriptive.csv')
fig,(a,b)=plt.subplots(1,2,figsize=(8.4,3.6))
for j,policy in enumerate(['all','k1']):
    part=pol[pol.training_policy==policy].set_index('model').loc[models]
    a.plot(np.arange(4),part.delta_accuracy*100,marker=['o','s'][j],color=[teal,ochre][j],label=['Full-set training','Single-photo training'][j])
a.set_xticks(range(4),[short[m] for m in models]);a.set_ylim(0,10);a.set_ylabel('One-to-two gain (pp)');a.set_title('a  Training policy',loc='left');a.legend(fontsize=9)
for j,inc in enumerate([True,False]):
    q=support[support.included==inc].groupby(['backbone','model']).accuracy.mean()
    v=[q.loc[key]*100 for key in order];b.plot(range(6),v,marker=['o','s'][j],color=[teal,ochre][j],label=['Matched support','Excluded from matching'][j])
b.set_xticks(range(6),['D/MF','D/PM','D/DS','D/ST','R/PM','R/DS']);b.set_ylim(0,100);b.set_ylabel('Natural two-photo macro accuracy (%)');b.set_title('b  Matching selection',loc='left');b.legend(fontsize=9)
fig.tight_layout();save(fig,'figure_sensitivity')
support.to_csv(D/'matched_unmatched_natural_descriptive.csv',index=False);pol.to_csv(D/'training_policy_sensitivity.csv',index=False)

rows=[]
for r in gain.itertuples():
    q=effects.query("analysis_branch=='primary' and mode=='full8' and training_policy=='all' and before==1 and after==2 and model==@r.model").iloc[0] if r.backbone=='dinov2' else paired.query("backbone=='resnet50' and model==@r.model and contrast=='k1_to_k2'").iloc[0]
    raw=q.observer_signflip_p if r.backbone=='dinov2' else q.p
    rows.append(['DINOv2' if r.backbone=='dinov2' else 'ResNet-50',short[r.model],f'{100*r.k1:.2f}',f'{100*r.k2:.2f}',f'{100*r.delta:+.2f}',f'[{100*r.lo:.2f}, {100*r.hi:.2f}]',f'{raw:.4f}',f'{r.adjusted_p:.4f}','128' if r.backbone=='dinov2' else '16'])
tab('main_gain',['Encoder','Model',r'$k=1$',r'$k=2$','Gain','95\% CI',r'Raw $p$',r'Adj. $p$','Tests'],rows,'llrrrrrrr')
rows=[]
for enc,m in order:
    u=repeat.query("backbone==@enc and model==@m and condition=='unrestricted_common'").accuracy_delta*100
    v=repeat.query("backbone==@enc and model==@m and condition=='difficulty_matched'").accuracy_delta*100
    rows.append(['DINOv2' if enc=='dinov2' else 'ResNet-50',short[m],f'{u.mean():+.2f}',f'{v.mean():+.2f}',f'[{v.min():.2f}, {v.max():.2f}]'])
tab('matched_gains',['Encoder','Model','Unrestricted','Matched',r'\shortstack{Range across 20\\regrouping repeats}'],rows,'llrrr')
rows=[]
for q in paired.itertuples():rows.append(['DINOv2' if q.backbone=='dinov2' else 'ResNet-50',short[q.model],'Native--pool' if q.contrast=='native_minus_pool' else q.contrast.replace('k','').replace('_to_',r'$\to$'),q.budget,f'{q.delta*100:+.3f}',f'[{q.lower95*100:.3f}, {q.upper95*100:.3f}]',f'{q.p_holm16:.4f}'])
tab('holm16',['Encoder','Model','Contrast',r'$k$','Effect (pp)','95\% CI (pp)',r'\shortstack{Holm-adjusted $p$\\16-test family}'],rows,'lllrrrr')
for name,df in [('gains',gain),('fixed_five',fixed),('grouping',repeat),('error_means',em),('pooling',paired),('curves',curves),('attrition',attr)]:df.to_csv(D/f'{name}.csv')
(F/'asset_manifest.json').write_text(json.dumps(dict(inputs=used,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_inference=True,plots=5),indent=2),encoding='utf-8')
print('Five revised quantitative figures and tables complete; support descriptive counts:',support.groupby('included')[['groups','taxa']].first().to_dict())
