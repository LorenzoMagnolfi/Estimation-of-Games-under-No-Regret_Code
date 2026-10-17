"""Explicitly conditional conversion from receipt-equivalent costs.

Illustrative constant PayPal payout schedule, not observed seller invoices
or transaction-specific fees. No fee is subtracted twice from gross margins.
"""
import hashlib
import json
from pathlib import Path
from audit_panel import OUT


def main():
    b=s=.03;f=.0349;fixed=.49;tax_range=[0,.10];a_min=(1-s)/(1+b)-f*(1+tax_range[1]);a_max=(1-s)/(1+b)-f*(1+tax_range[0]);records=[]
    for stem in ['daily_multisku_seller_d0_h02','daily_multisku_seller_d5_h02','daily_multisku_seller_family10000_d5_h05']:
        source=OUT/f'{stem}_inference.json';d=json.loads(source.read_text());profile=json.loads((OUT/d['profile']).read_text());cost=d['asymptotic_cost_outer_interval'];margin=d['asymptotic_margin_outer_interval'];price=profile['mean_receipt'];net_cost=[a_min*cost[0]-fixed,a_max*cost[1]-fixed];net_margin=[a_min*margin[0],a_max*margin[1]]
        records.append({'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'receipt_equivalent_cost_interval':cost,'receipt_equivalent_margin_interval':margin,'conditional_net_cost_interval':net_cost,'conditional_net_operating_margin_interval':net_margin,'conditional_mean_net_receipt_interval':[a_min*price-fixed,a_max*price-fixed]})
    report={'buyer_fee_rate':b,'seller_fee_rate':s,'processing_rate':f,'fixed_processing_fee':fixed,'assumed_expected_tax_rate_range':tax_range,'affine_receipt_coefficient_range':[a_min,a_max],'sources':['https://swappa.com/blog/swappa-fees/','https://swappa.com/blog/who-pays-the-fees-when-issuing-a-paypal-refund/'],'records':records,'interpretation':'Illustrative monetary-unit conversion assuming a constant affine PayPal payout schedule and an expected tax rate between zero and ten percent. Tax rates and the focal seller payout schedule are not observed or verified. Cost includes all per-sale incremental costs represented by the model, not hardware acquisition cost alone. Positive scaling cancels from the normalized learning event; net cost is a*c-F and operating margin is a*(receipt-c).'}
    (OUT/'daily_fee_conversion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
