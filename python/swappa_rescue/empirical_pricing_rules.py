"""Chronological validation of fixed, empirically motivated pricing rules.

The design is frozen in docs/evidence/gadgetpickup_rule_design_20261020.json.
No demand model or cost estimate is loaded by this module.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from audit_panel import OUT, DATA_VERSION
from demand_timing_sensitivity import terminal_status_data

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT/'docs/evidence/gadgetpickup_rule_design_20261020.json'
RUN = OUT/'empirical_rules_20261020'
KEYS = ['device_id','Storage','Condition']


def display(raw):
    """Project a proposed displayed price onto the declared asking-price grid."""
    ask = np.clip(np.rint(np.asarray(raw)/1.03), 10, 1941)
    return np.ceil(1.03*ask-1e-9)


def features(s):
    """Construct predictors using previous dates, never the current outcome."""
    s = s.sort_values(['Code','date']).copy().reset_index(drop=True)
    s['category'] = s[KEYS].astype(str).agg('|'.join,axis=1)
    g = s.groupby('Code',sort=False)
    s['previous_price'] = g.Price.shift()
    s['previous_rival_min'] = g.rival_min.shift()
    s['gap_days'] = g.date.diff().dt.days
    changed = s.Price.ne(s.previous_price)
    s['last_change_date'] = s.date.where(changed).groupby(s.Code).ffill()
    last = s.groupby('Code').last_change_date.shift()
    s['elapsed_since_change'] = (s.date-last).dt.days
    agg = s.groupby(['category','date']).agg(category_price=('Price','median'),sales=('sale_terminal_status','sum'))
    rows=[]
    for category,a in agg.groupby(level=0):
        a=a.droplevel(0).reindex(pd.date_range(s.date.min(),s.date.max(),freq='D'))
        a['previous_category_price']=a.category_price.shift(1)
        a['sales7']=a.sales.fillna(0).shift(1).rolling(7,min_periods=1).sum()
        a['category']=category;a['date']=a.index
        rows.append(a[['category','date','previous_category_price','sales7']])
    s=s.merge(pd.concat(rows,ignore_index=True),on=['category','date'],how='left',validate='many_to_one')
    train=s.date.le('2023-09-30')
    floors=s.loc[train].groupby('category').Price.agg(['min',lambda x:x.quantile(.1)]).rename(columns={'<lambda_0>':'q10'})
    s=s.merge(floors.rename(columns={'min':'floor_min','q10':'floor_q10'}),left_on='category',right_index=True,how='left',validate='many_to_one')
    s['new_category']=s.floor_min.isna()
    s[['floor_min','floor_q10']]=s[['floor_min','floor_q10']].fillna(10.)
    s['previous_category_price']=s.previous_category_price.fillna(s.previous_price)
    s['eligible']=s.gap_days.eq(1)&s.previous_price.notna()&s.previous_rival_min.notna()
    s['actual_change']=s.Price.ne(s.previous_price)
    s['cell_weight']=1/s.groupby(['category','date']).Code.transform('size')
    return s


def candidates(s,spec):
    prev=s.previous_price.fillna(s.Price).to_numpy(); rival=s.previous_rival_min.fillna(s.Price).to_numpy()
    weekday=s.date.dt.dayofweek.to_numpy();elapsed=s.elapsed_since_change.fillna(0).to_numpy()
    sales=s.sales7.fillna(0).to_numpy()
    yield 'persistence',{},prev
    yield 'category_anchor',{},display(s.previous_category_price.fillna(s.Price).to_numpy())
    floors={'none':np.full(len(s),10.),'minimum':s.floor_min.to_numpy(),'q10':s.floor_q10.to_numpy()}
    for offset,floor,review,deadband in itertools.product(spec['competitor_offsets'],floors,spec['review_weekdays'],spec['deadbands']):
        proposed=display(np.maximum(rival+offset,floors[floor]))
        active=(True if review=='daily' else weekday==review)&(abs(proposed-prev)>=deadband)
        yield 'competitor',dict(offset=offset,floor=floor,review=review,deadband=deadband),np.where(active,proposed,prev)
    for cooldown,step,review,floor in itertools.product(spec['markdown_cooldowns'],spec['price_steps'],['daily',0,4],floors):
        active=(elapsed>=cooldown)&(True if review=='daily' else weekday==review)
        proposed=display(np.maximum(prev-step,floors[floor]))
        yield 'age_markdown',dict(cooldown=cooldown,step=step,review=review,floor=floor),np.where(active,proposed,prev)
    for low,step,cooldown,review in itertools.product(spec['sales_low_thresholds'],spec['price_steps'],spec['markdown_cooldowns'],['daily',0,4]):
        direction=np.where(sales<=low,-1,np.where(sales>=2*low+2,1,0))
        active=(elapsed>=cooldown)&(True if review=='daily' else weekday==review)
        proposed=display(prev+step*direction)
        yield 'sales_feedback',dict(low=low,high=2*low+2,step=step,cooldown=cooldown,review=review),np.where(active,proposed,prev)


def scores(s,pred,mask):
    y=s.Price.to_numpy();prev=s.previous_price.to_numpy();w=s.cell_weight.to_numpy()
    ix=np.flatnonzero(mask);changed=s.actual_change.to_numpy()[ix];err=abs(y[ix]-pred[ix]);ww=w[ix]
    means=[float(np.average(err[changed==v],weights=ww[changed==v])) if np.any(changed==v) else None for v in [False,True]]
    pc=abs(pred[ix]-prev[ix])>1e-8
    tp=int(np.sum(pc&changed));fp=int(np.sum(pc&~changed));fn=int(np.sum(~pc&changed))
    return dict(n=len(ix),changes=int(changed.sum()),mae=float(np.average(err,weights=ww)),balanced_mae=float(np.mean([x for x in means if x is not None])),unchanged_mae=means[0],changed_mae=means[1],within_one=float(np.average(err<=1,weights=ww)),change_precision=tp/(tp+fp) if tp+fp else 0.,change_recall=tp/(tp+fn) if tp+fn else 0.,change_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,direction_accuracy_on_changes=float(np.mean(np.sign(pred[ix][changed]-prev[ix][changed])==np.sign(y[ix][changed]-prev[ix][changed]))) if changed.any() else None)


def main():
    assert DATA_VERSION=='fullsource_refscreen_known'
    spec=json.loads(DESIGN.read_text());RUN.mkdir(exist_ok=True)
    d=terminal_status_data();reports=[];all_validation=[]
    for seller in spec['sellers']:
        s=features(d[d.Seller.eq(seller)].copy())
        validation=(s.eligible&s.date.gt(spec['train_end'])&s.date.le(spec['validation_end'])).to_numpy()
        test=(s.eligible&s.date.ge(spec['test_start'])).to_numpy()
        assert validation.any() and test.any()
        best={};count=0
        for family,params,pred in candidates(s,spec):
            val=scores(s,pred,validation);count+=1
            all_validation.append(dict(seller=seller,family=family,params=json.dumps(params,sort_keys=True),mae=val['mae'],balanced_mae=val['balanced_mae']))
            for criterion in ['mae','balanced_mae']:
                key=(family,criterion)
                if key not in best or val[criterion]<best[key][0]-1e-12:
                    best[key]=(val[criterion],params,pred.copy(),val)
        report=dict(seller=seller,rows=len(s),categories=s.category.nunique(),candidates=count,unseen_category_test_rows=int((s.new_category&test).sum()),selected=[])
        forecast=s[['Code','date','Seller','Price','category','eligible','previous_price']].copy()
        for (family,criterion),(score,params,pred,val) in best.items():
            name=family+'__'+criterion
            forecast[name]=pred
            rec=dict(family=family,selector=criterion,params=params,validation=val,test=scores(s,pred,test))
            report['selected'].append(rec)
        for step in [-10,-5,5,10]:
            # Amount is in integer asking-price dollars, as preregistered.
            ask=np.floor(s.previous_price.fillna(s.Price).to_numpy()/1.03+1e-9)
            forecast['ask_shift_'+str(step)]=np.ceil(1.03*np.clip(ask+step,10,1941)-1e-9)
        stem=seller.replace(' ','_').replace(',','').replace('.','')
        forecast.to_parquet(RUN/(stem+'_forecasts.parquet'),index=False)
        report['forecast_file']=str(RUN/(stem+'_forecasts.parquet'))
        reports.append(report)
        print(json.dumps(dict(seller=seller,selected=[dict(family=r['family'],selector=r['selector'],params=r['params'],test=r['test']) for r in report['selected']]),default=int),flush=True)
    pd.DataFrame(all_validation).to_csv(RUN/'validation_candidates.csv',index=False)
    result=dict(design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((OUT/'panel.parquet').read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),interpretation='One-step forecasts on retained consecutive observations. Prices only; no costs or demand used for model selection.',reports=reports)
    (RUN/'rule_validation.json').write_text(json.dumps(result,indent=2,default=int)+'\n')
    (ROOT/'docs/evidence/gadgetpickup_rule_validation_20261020.json').write_text(json.dumps(result,indent=2,default=int)+'\n')
    print('RULE_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
