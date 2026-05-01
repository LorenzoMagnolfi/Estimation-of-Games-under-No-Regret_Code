********************************************************************************
*    GENERATE DATA FOR MATLAB — MIN-PRICE AGGREGATION VARIANT                  *
*    (responds to R1.4.c: "what if sale prob depends on lowest competing      *
*    price, not average?")                                                      *
*                                                                              *
*    Mirror of generate_matlab_data.do but with competitor_price = min-of-     *
*    others instead of mean-of-others.  Outputs to *_minprice.xlsx so the     *
*    original files are preserved alongside.                                  *
********************************************************************************

do "stata/macros.do"

* New output file paths (parallel to $sale_prob and $seller_dist)
gl sale_prob_minprice    "matlab/data/sale_probability_5bins_res1_minprice.xlsx"
gl seller_dist_minprice  "matlab/data/SellerDistribution_15_sellers_res1_minprice.xlsx"

********************************************************************************
*    Helper: rebuild competitor_price as MIN-of-others, redo bins              *
********************************************************************************

capture program drop _rebuild_minprice_bins
program define _rebuild_minprice_bins
    * Drop variables that depend on the avg-based competitor_price; we'll rebuild.
    capture drop competitor_price
    capture drop comp_net_price_1
    capture drop comp_net_price_bins_1
    capture drop _rank_grp
    capture drop _smallest
    capture drop _second
    capture drop _Nseller_grp

    * Sort within (device, storage, condition, date) by Price, take rank.
    sort device_id Storage Condition date Price
    by device_id Storage Condition date: gen _rank_grp = _n
    by device_id Storage Condition date: gen _Nseller_grp = _N

    * Smallest price in the group, second-smallest in the group.
    by device_id Storage Condition date: gen _smallest = Price[1]
    by device_id Storage Condition date: gen _second   = Price[2]

    * For each row, "competitor min" = min over OTHER sellers' prices.
    * - If this row is the unique minimum (rank == 1), use second-smallest.
    * - Otherwise, use smallest (someone else is at the min).
    * - If group size == 1, no competitors -> drop later (NA).
    gen competitor_price = cond(_rank_grp == 1, _second, _smallest) if _Nseller_grp >= 2
    drop _rank_grp _smallest _second _Nseller_grp

    drop if missing(competitor_price)

    * Recompute residual and bins (same naming convention as the original code).
    gen comp_net_price_1 = competitor_price - Ref_Price
    xtile comp_net_price_bins_1 = comp_net_price_1, nq(5)

    * Self-bins should already exist; if not, rebuild defensively.
    capture confirm variable self_net_price_bins_1
    if _rc != 0 {
        capture drop self_net_price_1
        gen self_net_price_1 = deviation
        xtile self_net_price_bins_1 = self_net_price_1, nq(5)
    }
end

********************************************************************************
*    Sale probability — top 15 + all sellers                                   *
********************************************************************************

* Top 15 sellers
use "$Seller_15", clear
_rebuild_minprice_bins
collapse (mean) Sale_Prob = Sold_Today, by(self_net_price_bins_1 comp_net_price_bins_1)
format Sale_Prob %9.2f
replace self_net_price_bins_1 = self_net_price_bins_1 - 1
replace comp_net_price_bins_1 = comp_net_price_bins_1 - 1
order self_net_price_bins comp_net_price_bins Sale_Prob
export excel using "$sale_prob_minprice", sheet("15sellers", replace) firstrow(variables)

* All sellers
use "$Seller_all", clear
_rebuild_minprice_bins
collapse (mean) Sale_Prob = Sold_Today, by(self_net_price_bins_1 comp_net_price_bins_1)
format Sale_Prob %9.2f
replace self_net_price_bins_1 = self_net_price_bins_1 - 1
replace comp_net_price_bins_1 = comp_net_price_bins_1 - 1
order self_net_price_bins comp_net_price_bins Sale_Prob
export excel using "$sale_prob_minprice", sheet("Allsellers", replace) firstrow(variables)

display "DONE: sale probability with min-price aggregation -> $sale_prob_minprice"

********************************************************************************
*    Seller Distribution (per-seller time-averaged 25-cell action histogram)   *
********************************************************************************

use "$Seller_15", clear
_rebuild_minprice_bins

preserve

* 25 indicator dummies for joint (self_bin, comp_bin)
forvalues s = 1/5 {
    forvalues c = 1/5 {
        gen i_`=`s'-1'`=`c'-1' = (self_net_price_bins_1 == `s') & (comp_net_price_bins_1 == `c')
    }
}

* Order observations within seller for time-average computation
bys Seller (date min_date device_id Code): gen device_day = _n

* Initialize time-average at first row
forvalues s = 0/4 {
    forvalues c = 0/4 {
        gen TimeAverage_`s'`c' = i_`s'`c' if device_day == 1
    }
}

* Cumulative time-average within seller
forvalues s = 0/4 {
    forvalues c = 0/4 {
        by Seller: replace TimeAverage_`s'`c' = (TimeAverage_`s'`c'[_n-1] * (_n - 1) + i_`s'`c') / _n if _n >= 2
    }
}

format TimeAverage_00-TimeAverage_44 %9.2f

forvalues j = 1/15 {
    so Seller_number date min_date device_id Code
    export exc TimeAverage_00-TimeAverage_44 using "$seller_dist_minprice" if Seller_number == `j', sh("Seller_`j'", replace) first(var) nolabel
}

restore

display "DONE: seller distribution with min-price aggregation -> $seller_dist_minprice"

********************************************************************************
*    Mean / Median actions (same as original; needed for matlab consumption)  *
********************************************************************************

use "$Seller_15", clear
_rebuild_minprice_bins

preserve
collapse (mean) mean_deviation = self_net_price_1 (median) median_deviation = self_net_price_1, by(self_net_price_bins_1)
format mean_deviation median_deviation %9.2f
gen id = 1
tostring self_net_price_bins_1, replace
replace self_net_price_bins_1 = "_bin_" + self_net_price_bins_1
reshape wide mean_deviation median_deviation, i(id) j(self_net_price_bins_1, string)
drop id

export exc mean_*   using "$seller_dist_minprice" if _n == 1, sh("Actions_Mean",   replace) first(var) nolabel
export exc median_* using "$seller_dist_minprice" if _n == 1, sh("Actions_Median", replace) first(var) nolabel
restore

display "=== MIN-PRICE REBUILD DONE ==="
