from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, json
ROOT=Path(__file__).parents[1]
R=ROOT/'results/metrics'; F=ROOT/'results/figures'; F.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(R/'counterfactual_results.csv')
# success comparison
s=df.groupby(['mode','source_risk']).success.mean().reset_index()
plt.figure(figsize=(8,5));
for mode in s['mode'].unique():
    x=s[s['mode']==mode]; plt.bar([f'{mode}\n{r}' for r in x.source_risk],x.success)
plt.ylabel('Success rate'); plt.ylim(0,1.05); plt.title('Counterfactual Success Rate'); plt.tight_layout(); plt.savefig(F/'cf_success_comparison.png',dpi=180); plt.close()
# distributions
for metric in ['l0','l1','l2']:
    plt.figure(figsize=(8,5));
    for mode in df['mode'].unique(): plt.hist(df[df['mode']==mode][metric],bins=12,alpha=.55,label=mode)
    plt.xlabel(metric.upper()); plt.ylabel('Count'); plt.title(f'{metric.upper()} Distribution'); plt.legend(); plt.tight_layout(); plt.savefig(F/f'{metric}_distribution.png',dpi=180); plt.close()
# changed feature frequency constrained
cg=df[df['mode']=='constrained'].copy(); counts={}
for s in cg.changed_features.fillna(''):
    for f in [x for x in s.split('|') if x]: counts[f]=counts.get(f,0)+1
cf=pd.Series(counts).sort_values(ascending=True)
plt.figure(figsize=(8,6)); cf.plot(kind='barh'); plt.xlabel('Number of counterfactuals changed'); plt.title('Changed Feature Frequency — Constrained'); plt.tight_layout(); plt.savefig(F/'changed_feature_frequency.png',dpi=180); plt.close()
# constraint violations baseline
uv=df[df['mode']=='unconstrained'].copy(); violation=(~uv.valid).mean()
cv=df[df['mode']=='constrained'].copy(); cviolation=(~cv.valid).mean()
plt.figure(figsize=(6,5)); plt.bar(['Unconstrained\nchecked against rules','Constrained'],[violation,cviolation]); plt.ylabel('Violation rate'); plt.ylim(0,1.05); plt.title('Constraint Validation'); plt.tight_layout(); plt.savefig(F/'constraint_violation_comparison.png',dpi=180); plt.close()
# representative CF table
rep=[]
for source in ['Medium','High']:
    g=cg[cg.source_risk==source].iloc[0]
    cols=['profile_id','source_risk','p_low','l0','l1','l2','optimization_steps','changed_features']
    rep.append(g[cols].to_dict())
pd.DataFrame(rep).to_csv(R/'representative_counterfactuals.csv',index=False)
print('figures generated',len(list(F.glob('*.png'))))
