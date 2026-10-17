"""Export a compact comparison of the validated shared-learner bounds."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from audit_panel import OUT


def main():
    specifications=[('daily_multisku_common_d0_h02','Common demand; one cost per SKU'),('daily_multisku_seller_d0_h02','Seller-specific demand; one cost per SKU'),('daily_multisku_common_d5_h02',r'Common demand; within-SKU cost span $\leq 5$'),('daily_multisku_seller_d5_h02',r'Seller-specific demand; cost span $\leq 5$'),('daily_multisku_seller_family10000_d5_h05',r'Seller-specific; span $\leq 5$; 10,000 sellers')]
    span_priors={r['within_sku_cost_span']:r['prior_cost_interval'] for r in json.loads((OUT/'daily_cost_span_prior.json').read_text())}
    rows=[]
    for stem,label in specifications:
        result=json.loads((OUT/f'{stem}_inference.json').read_text());profile=json.loads((OUT/result['profile']).read_text());rows.append((label,result,profile))
    plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.size':10,'axes.spines.top':True,'axes.spines.right':True})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),sharey=True);blue='#245b83';dark='#12364e';gray='#d2d6da';y=np.arange(len(rows))[::-1]
    for pos,(label,result,profile) in zip(y,rows):
        price=profile['mean_receipt'];point=result['point_cost_outer_interval'];prior=span_priors[profile['cost_delta']]
        bounds=[result['asymptotic_cost_outer_interval'],result['asymptotic_margin_outer_interval']]
        centers=[point,[price-point[1],price-point[0]]];priors=[prior,[price-prior[1],price-prior[0]]]
        for ax,uncertainty,fitted,baseline in zip(axes,bounds,centers,priors):
            ax.plot(baseline,[pos,pos],color=gray,lw=2,zorder=1)
            ax.plot(uncertainty,[pos,pos],color=blue,lw=2,zorder=2)
            ax.plot(uncertainty,[pos,pos],linestyle='none',marker='|',markersize=9,color=blue,zorder=3)
            ax.plot(fitted,[pos,pos],color=dark,lw=6,solid_capstyle='butt',zorder=4)
    axes[0].set_yticks(y,[r[0] for r in rows]);axes[0].set_title('Mean replacement cost');axes[1].set_title('Mean gross margin')
    for ax in axes:
        ax.set_xlim(0,150);ax.set_ylim(-.7,len(rows)-.3);ax.set_xticks([0,30,60,90,120,150]);ax.set_xlabel('Receipt-equivalent dollars');ax.grid(axis='x',color='#eeeeee',lw=.7);ax.tick_params(axis='y',length=0)
    axes[0].axvline(rows[0][2]['mean_receipt'],color='#999999',ls='--',lw=.8)
    fig.suptitle('Upgrade: daily pricing shared across product groups',y=.98,fontsize=14)
    legend=[Line2D([0],[0],color=gray,lw=2,label='Prior support'),Line2D([0],[0],color=blue,lw=2,marker='|',label=r'Asymptotic 95\% outer interval'),Line2D([0],[0],color=dark,lw=6,label='Fitted-demand outer bound')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.61,.095),ncol=3,frameon=False,fontsize=9)
    fig.text(.02,.025,'Conditional on structural demand, the stated cost restrictions, and a shared expected-payoff learner.\nFirst four rows allow selection among 20 sellers; the last allows 10,000. Mean receipt: 144.00.',fontsize=8)
    fig.subplots_adjust(left=.38,right=.985,top=.84,bottom=.25,wspace=.13)
    for extension in ['png','pdf']:fig.savefig(OUT/f'daily_multisku_bounds.{extension}',dpi=190,bbox_inches='tight')
    print(OUT/'daily_multisku_bounds.png')

if __name__=='__main__':main()
