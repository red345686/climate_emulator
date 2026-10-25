"""Standalone paper figures from completed analysis tables; no model fitting."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def make_plots(output):
    output=Path(output)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    controls=rows(output/'aerosol_controls.csv')
    complement=rows(output/'branch_complementarity.csv')
    inner=rows(output/'training_sensitivity.csv')
    outer=rows(output/'outer_sensitivity.csv')
    tasks=list(dict.fromkeys(r['task'] for r in controls))
    labels=[t.removeprefix('coverage_').replace('ssp','').replace('_','+') for t in tasks]
    for target in ('tas','pr'):
        fig,ax=plt.subplots(figsize=(8,4))
        for variant,label in [('full_frozen_blend','Full frozen blend'),('ghg_same_settings_blend','GHG only, shared settings'),
                              ('ghg_training_selected_blend','GHG only, inner-selected')]:
            yy=[float(next(r['nrmse_total'] for r in controls if r['task']==t and r['target']==target and r['variant']==variant)) for t in tasks]
            ax.plot(labels,yy,marker='o',label=label)
        ax.set(xlabel='Future training pool',ylabel='Total NRMSE',title=f'{target}: joint aerosol-feature control')
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/f'{target}_aerosol_control.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        selected=[next(r for r in complement if r['task']==t and r['target']==target) for t in tasks]
        yy=[float(r['spatial_error_cosine']) if r['spatial_error_cosine'] else np.nan for r in selected]
        ax.bar(labels,yy,color=['tab:blue' if r['ridge_branch_active']=='True' else 'gray' for r in selected])
        ax.axhline(0,color='black',lw=.7);ax.set(ylim=(-1.05,1.05),ylabel='Area-weighted spatial error cosine',
            xlabel='Future training pool',title=f'{target}: branch error alignment (gray = inactive ridge)')
        fig.tight_layout();fig.savefig(figures/f'{target}_branch_alignment.pdf');plt.close(fig)
        for features in ('full','ghg'):
            fig,axes=plt.subplots(2,len(tasks),figsize=(14,6),squeeze=False)
            for j,task in enumerate(tasks):
                for i,(source,metric,title) in enumerate([(inner,'selection_objective','Inner objective'),(outer,'nrmse_total','Outer total: exploratory')]):
                    subset=[r for r in source if r['task']==task and r['target']==target and r['features']==features]
                    aa=sorted({float(r['alpha']) for r in subset});ww=sorted({float(r['weight']) for r in subset})
                    matrix=np.full((len(aa),len(ww)),np.nan)
                    for r in subset:matrix[aa.index(float(r['alpha'])),ww.index(float(r['weight']))]=float(r[metric])
                    ax=axes[i,j];im=ax.imshow(matrix,aspect='auto',origin='lower',cmap='viridis')
                    ax.set(xticks=range(len(ww)),xticklabels=[f'{w:g}' for w in ww],yticks=range(len(aa)),
                           yticklabels=[f'{a:g}' for a in aa],xlabel='Ridge weight',ylabel='Alpha',title=labels[j]+'\n'+title)
                    ax.tick_params(axis='x',labelrotation=90,labelsize=6);fig.colorbar(im,ax=ax,shrink=.75)
            fig.suptitle(f'{target}, {features} features: lower is better; color scale varies by panel')
            fig.tight_layout();fig.savefig(figures/f'{target}_{features}_sensitivity.pdf');plt.close(fig)
    print(f'Figures written to {figures}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    make_plots(parser.parse_args().output)
