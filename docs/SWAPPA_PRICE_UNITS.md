# Swappa price units and fees

Internal source audit, 2026-10-16. No fee-adjusted result has been promoted into the paper.

The historical Swappa notebook reads the displayed listings_table. The Stata cleaner removes the dollar sign and destrings Price, without subtracting platform or payment fees. This supports a buyer-facing displayed-price interpretation. The historical rounding rule still requires verification. Do not execute the historical notebook, which contains obsolete request credentials; it is unnecessary for this audit and excluded from the OSC package.

Swappa's current Ask Price FAQ distinguishes ask from listing price: a $100 ask appears as $103 to buyers before sales tax, and sellers must include shipping in their ask. The partner-fee FAQ states that the seller fee is 3% of ask, while PayPal processing is 3.49% plus $.49 of the total order. Sources accessed 2026-10-16: https://swappa.com/faq/answer/ask-price and https://swappa.com/faq/answer/partner-fee.

The indexed official announcement dated February 23, 2022 describes the transition to 3% fees on both sides and the $100/$103 example: https://swappa.com/blog/swappa-fees/. Its direct fetch returned 404, although its indexed text was available. This supports the fee structure predating the sample, but does not verify each seller's processor, negotiated rate, tax base, or the historical display-rounding rule.

Under a common affine receipt schedule, let p be displayed price before tax, b the buyer fee on ask, s the seller fee on ask, f the processing percentage, F the fixed processing charge, and tau the expected tax rate entering that charge. Then a=(1-s)/(1+b)-f*(1+tau), and net receipts are a*p-F. For remaining per-sale cost k, profit equals a*(p-c_eff)*q(p), where c_eff=(k+F)/a. Thus k=a*c_eff-F and net operating margin is a*(p-c_eff).

Fees should not be subtracted a second time from the inferred effective gross-price margin. They already enter c_eff through the behavioral profit function. Shipping, handling, and refurbishment belong in k unless separately observed and removed. Inventory opportunity value is not automatically procurement cost.

For illustration, b=s=.03, f=.0349, F=.49, tau=0 give a=.9068476. Common positive scaling cancels from the fully normalized r/C event. This permits a transparent monetary-unit transformation under the stated schedule. Heterogeneous or action-dependent receipt coefficients do not cancel from multiproduct profits. Unknown buyer locations must not silently be assigned zero sales tax as a verified payout calculation.

A cost-quality restriction on c_eff is not the same as a procurement-cost restriction. Reference prices come from another retail channel and should not be converted by Swappa's fees. Historical display rounding, the attainable seller price grid, the focal seller's processor, and the fee base remain explicit empirical checks. The stated numerical policy menu cannot silently stand in for a different platform action set.
