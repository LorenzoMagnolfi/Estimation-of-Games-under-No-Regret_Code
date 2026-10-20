"""Exploratory nonextrapolating demand models and review/feedback audit."""
import hashlib,json,subprocess
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits
from gadget_demand_audit import GadgetDemand,metrics,RUN,SPEC
from demand_timing_sensitivity import terminal_status_data
from audit_panel import ROOT,OUT

class NoTrend(GadgetDemand):
    def __init__(self):super().__init__('category')
    def continuous(self,d,action=None):return super().continuous(d,action).drop(columns='calendar_time')

class TreeDemand:
    def __init__(self):
        self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        self.model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=200,max_leaf_nodes=7,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=20261020)
        self.base=GadgetDemand('category')
    def matrix(self,d,fit=False):
        z=self.base.continuous(d).copy();z['price']=d.Price.to_numpy()/100
        cat=d[['device_id','Storage','Condition']].astype(str).copy();cat['weekday']=d.date.dt.dayofweek.astype(str).to_numpy()
        c=self.encoder.fit_transform(cat) if fit else self.encoder.transform(cat)
        if fit:self.model.set_params(categorical_features=list(range(len(z.columns),len(z.columns)+c.shape[1])))
        return np.c_[z.to_numpy(),c]
    def fit(self,d,y):self.model.fit(self.matrix(d,True),y);return self
    def predict(self,d):return self.model.predict_proba(self.matrix(d))[:,1]

def main():
    spec=json.loads(SPEC.read_text());d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1);records=[]
    for end,start,stop in spec['chronological_splits']:
        tr=s.date.le(end);te=s.date.between(start,stop)
        for name,fit in [('category_no_trend',NoTrend()),('tree',TreeDemand())]:
            fit.fit(s[tr],s.loc[tr,'sale_terminal_status']);q=fit.predict(s[te]);r=dict(kind=name,train_end=end,test_start=start,test_end=stop,metrics=metrics(s[te],q));records.append(r);print(json.dumps(r),flush=True)
            if start=='2023-11-01':
                joblib.dump(fit,RUN/('gadget_'+name+'_oct31.joblib'))
                p=s.loc[te,['Code','date','category','Price','sale_terminal_status']].copy();p['prediction']=q;p.to_parquet(RUN/('forecast_'+name+'.parquet'),index=False)
    # Count price decisions and distinct prices without treating inventory copies as independent rounds.
    g=s.groupby('Code');s['dp']=g.Price.diff();s['gap']=g.date.diff().dt.days;s['cut']=s.dp.lt(0)&s.gap.eq(1)
    cell=s.groupby(['category','date']).agg(rows=('Code','size'),price=('Price','median'),distinct_prices=('Price','nunique'),sales=('sale_terminal_status','sum'),cuts=('cut','sum')).reset_index().sort_values(['category','date'])
    cell['gap']=cell.groupby('category').date.diff().dt.days;cell['dprice']=cell.groupby('category').price.diff();cell['common']=cell.distinct_prices.eq(1)
    categories=cell.groupby('category').agg(days=('date','size'),exposures=('rows','sum'),sales=('sales','sum'),distinct_median_prices=('price','nunique'),common_price_days=('common','sum'),review_changes=('dprice',lambda x:int((x.fillna(0)!=0).sum()))
    train=s[s.date.le('2023-09-30')].groupby('category').size();chosen=train.sort_values(ascending=False).head(5).index
    bymonth=s.assign(month=s.date.dt.strftime('%Y-%m')).groupby('month').agg(rows=('Code','size'),sales=('sale_terminal_status','sum'),mean_price=('Price','mean'));bymonth['sale_rate']=bymonth.sales/bymonth.rows
    batches=s.groupby('date').agg(cuts=('cut','sum'),listings=('Code','size'),sales=('sale_terminal_status','sum'))
    eligible=s.gap.eq(1);change=s.dp.ne(0)&eligible
    review=dict(calendar_days=int(s.date.nunique()),listing_rows=len(s),category_days=len(cell),categories=int(s.category.nunique()),consecutive_listing_rows=int(eligible.sum()),changed_listing_rows=int(change.sum()),common_price_multi_listing_share=float(cell.loc[cell.rows.gt(1),'common'].mean()),days_with_changes=int(s.loc[change,'date'].nunique()),days_with_at_least_five_cuts=int(batches.cuts.ge(5).sum()),top_categories_selected_by_pre_october_exposures=categories.loc[chosen].reset_index().to_dict('records'),monthly=bymonth.reset_index().to_dict('records'))
    cell.to_parquet(RUN/'category_days.parquet',index=False)
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),extension_design_sha256=hashlib.sha256((ROOT/'docs/evidence/gadgetpickup_demand_extension_20261020.json').read_bytes()).hexdigest(),records=records,review_audit=review)
    for p in [RUN/'extension.json',ROOT/'docs/evidence/gadgetpickup_demand_extension_results_20261020.json']:p.write_text(json.dumps(out,indent=2)+'\n')
    print('GADGET_EXTENSION_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
