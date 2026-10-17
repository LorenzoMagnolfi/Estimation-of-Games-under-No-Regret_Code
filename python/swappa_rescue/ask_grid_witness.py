"""Full feasible integer-ask oracle audit with actual pre-tax receipts."""
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from coordinate_profit import peer_spillover_bounds
from oracle_witness import maximum_own_profit
from ask_price_grid import recover_receipts,rounded_proposal

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag');ap.add_argument('--cost-file');ap.add_argument('--fit',default='multiproduct_initial_terminal.joblib');args=ap.parse_args()
    s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/args.fit)
    codes,idx=np.unique(s.Code,return_inverse=True);p=s.Price.to_numpy();x=recover_receipts(p);s['receipt']=x;P=350.;qa=fit.predict(s);z=fit.logit(s);n=s.own_n.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy()
    ef=fit.effects();b=(ef['own']+ef.get('own_seller::UPGRADE SOLUTION INC',0)+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100;assert bp>=0;offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
    old=pd.read_csv(OUT/'upgrade_optimized_oracle_costs.csv').set_index('Code').cost.reindex(codes).to_numpy();rounding=pd.Series(p-x,index=s.Code).groupby(level=0).mean().reindex(codes).to_numpy();upper=s.groupby('Code').receipt.min().reindex(codes).to_numpy();cost=np.clip(old-rounding,0,upper) if args.cost_file is None else pd.read_csv(args.cost_file).set_index('Code').cost.reindex(codes).to_numpy();assert np.all((cost>=0)&(cost<=upper+1e-8));c=cost[idx]
    asks=np.arange(1,int(np.floor(P/1.03))+1);base=1.03*np.maximum(asks[None,:],np.ceil(c[:,None]/1.03-1e-10));marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);extra,_=rounded_proposal(c,marks,P);proposal_x=np.column_stack([base,extra]);proposal=np.ceil(proposal_x-1e-9)
    assert np.all(proposal<=P) and np.all(proposal_x>=c[:,None]-1e-8)
    with threadpool_limits(limits=2):
        logits=offset[:,None]+b[:,None]*proposal+gp*np.maximum(proposal-rival[:,None],0)+gn*np.minimum(proposal-rival[:,None],0)
        own=expit(logits)*(proposal_x-c[:,None])-qa[:,None]*(x-c)[:,None];spl,sph=peer_spillover_bounds(proposal,proposal,p,z,n,groups,x-c,x-c,bp,P);assert np.max(abs(spl-sph))<1e-8
        rstar=np.maximum(0,(own+spl).max(1));C=maximum_own_profit(c,offset,b,gp,gn,rival,P)
        for ids in groups:
            m=len(ids)
            if m<=1:continue
            diff=expit(z[ids][None,:]+bp*(P-p[ids,None])/(m-1))-expit(z[ids][None,:]+bp*(c[ids,None]-p[ids,None])/(m-1));C[ids]+=(diff*(1-np.eye(m)))@(x[ids]-c[ids])
    loss=float(np.log1p(.99*rstar[active]/C[active]).sum());threshold=float(np.log(4*(len(asks)+len(marks)+1))+np.log(1000000)+np.log(1/.005));events=s.sale_terminal_status.to_numpy();anchor=s.groupby('Code').initial_reference.first().reindex(codes).to_numpy()
    report={'cost_file':args.cost_file,'cost_sha256':hashlib.sha256(Path(args.cost_file).read_bytes()).hexdigest() if args.cost_file else None,'fit':args.fit,'model':'integer_ask_exact_receipts_continuous_upper_range','logwealth':loss,'threshold':threshold,'passes':loss<=threshold,'max_rho0746_residual':float(abs(cost-.746*anchor).max()),'average_rounding_dollars':float((p-x).mean()),'targets':{}}
    for name,w in [('all_sales',events),('continuation_sales',events*active)]:
        report['targets'][name]={'cost':float(w@c/w.sum()),'receipt':float(w@x/w.sum()),'margin':float(w@(x-c)/w.sum()),'n':int(w.sum())}
    stem='ask_grid_witness_'+(args.tag if args.tag else args.fit.replace('.joblib','')+('_supplied' if args.cost_file else ''));pd.DataFrame({'Code':codes,'cost':cost}).to_csv(OUT/(stem+'_costs.csv'),index=False);(OUT/(stem+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
