
********************************************************************************
*                       GENERATE DATA FOR MATLAB                               *
********************************************************************************

do "stata/macros.do"


**# Compute Sale Probability
**## Top 15 Sellers
use "$Seller_15", clear
collapse (mean) Sale_Prob = Sold_Today, by(self_net_price_bins_1 comp_net_price_bins_1)
format Sale_Prob %9.2f
replace self_net_price_bins_1= self_net_price_bins_1 - 1
replace comp_net_price_bins_1= comp_net_price_bins_1 - 1
order self_net_price_bins comp_net_price_bins Sale_Prob
export excel using "$sale_prob", sheet("15sellers",replace) firstrow(variables)

**## All sellers
use "$Seller_all", clear
collapse (mean) Sale_Prob = Sold_Today, by(self_net_price_bins_1 comp_net_price_bins_1)
format Sale_Prob %9.2f
replace self_net_price_bins_1= self_net_price_bins_1 - 1
replace comp_net_price_bins_1= comp_net_price_bins_1 - 1
order self_net_price_bins comp_net_price_bins Sale_Prob
export excel using "$sale_prob", sheet("Allsellers",replace) firstrow(variables)


**# Compute Seller Distribution
use "$Seller_15",clear

**## Generate Data for Seller's Pricing Distribution
preserve

* seller time paths
gen i_00 = (self_net_price_bins_1 == 1) & (comp_net_price_bins_1 == 1)
gen i_01 = (self_net_price_bins_1 == 1) & (comp_net_price_bins_1 == 2)
gen i_02 = (self_net_price_bins_1 == 1) & (comp_net_price_bins_1 == 3)
gen i_03 = (self_net_price_bins_1 == 1) & (comp_net_price_bins_1 == 4)
gen i_04 = (self_net_price_bins_1 == 1) & (comp_net_price_bins_1 == 5)

gen i_10 = (self_net_price_bins_1 == 2) & (comp_net_price_bins_1 == 1)
gen i_11 = (self_net_price_bins_1 == 2) & (comp_net_price_bins_1 == 2)
gen i_12 = (self_net_price_bins_1 == 2) & (comp_net_price_bins_1 == 3)
gen i_13 = (self_net_price_bins_1 == 2) & (comp_net_price_bins_1 == 4)
gen i_14 = (self_net_price_bins_1 == 2) & (comp_net_price_bins_1 == 5)

gen i_20 = (self_net_price_bins_1 == 3) & (comp_net_price_bins_1 == 1)
gen i_21 = (self_net_price_bins_1 == 3) & (comp_net_price_bins_1 == 2)
gen i_22 = (self_net_price_bins_1 == 3) & (comp_net_price_bins_1 == 3)
gen i_23 = (self_net_price_bins_1 == 3) & (comp_net_price_bins_1 == 4)
gen i_24 = (self_net_price_bins_1 == 3) & (comp_net_price_bins_1 == 5)

gen i_30 = (self_net_price_bins_1 == 4) & (comp_net_price_bins_1 == 1)
gen i_31 = (self_net_price_bins_1 == 4) & (comp_net_price_bins_1 == 2)
gen i_32 = (self_net_price_bins_1 == 4) & (comp_net_price_bins_1 == 3)
gen i_33 = (self_net_price_bins_1 == 4) & (comp_net_price_bins_1 == 4)
gen i_34 = (self_net_price_bins_1 == 4) & (comp_net_price_bins_1 == 5)

gen i_40 = (self_net_price_bins_1 == 5) & (comp_net_price_bins_1 == 1)
gen i_41 = (self_net_price_bins_1 == 5) & (comp_net_price_bins_1 == 2)
gen i_42 = (self_net_price_bins_1 == 5) & (comp_net_price_bins_1 == 3)
gen i_43 = (self_net_price_bins_1 == 5) & (comp_net_price_bins_1 == 4)
gen i_44 = (self_net_price_bins_1 == 5) & (comp_net_price_bins_1 == 5)

* first entries
bys Seller (date min_date device_id Code): gen device_day=_n

gen TimeAverage_00 = i_00 if device_day == 1
gen TimeAverage_01 = i_01 if device_day == 1
gen TimeAverage_02 = i_02 if device_day == 1
gen TimeAverage_03 = i_03 if device_day == 1
gen TimeAverage_04 = i_04 if device_day == 1

gen TimeAverage_10 = i_10 if device_day == 1
gen TimeAverage_11 = i_11 if device_day == 1
gen TimeAverage_12 = i_12 if device_day == 1
gen TimeAverage_13 = i_13 if device_day == 1
gen TimeAverage_14 = i_14 if device_day == 1

gen TimeAverage_20 = i_20 if device_day == 1
gen TimeAverage_21 = i_21 if device_day == 1
gen TimeAverage_22 = i_22 if device_day == 1
gen TimeAverage_23 = i_23 if device_day == 1
gen TimeAverage_24 = i_24 if device_day == 1

gen TimeAverage_30 = i_30 if device_day == 1
gen TimeAverage_31 = i_31 if device_day == 1
gen TimeAverage_32 = i_32 if device_day == 1
gen TimeAverage_33 = i_33 if device_day == 1
gen TimeAverage_34 = i_34 if device_day == 1

gen TimeAverage_40 = i_40 if device_day == 1
gen TimeAverage_41 = i_41 if device_day == 1
gen TimeAverage_42 = i_42 if device_day == 1
gen TimeAverage_43 = i_43 if device_day == 1
gen TimeAverage_44 = i_44 if device_day == 1

* time-moving average
by Seller: replace TimeAverage_00 = (TimeAverage_00[_n-1]*(_n-1) + i_00)/(_n) if _n>=2
by Seller: replace TimeAverage_01 = (TimeAverage_01[_n-1]*(_n-1) + i_01)/(_n) if _n>=2
by Seller: replace TimeAverage_02 = (TimeAverage_02[_n-1]*(_n-1) + i_02)/(_n) if _n>=2
by Seller: replace TimeAverage_03 = (TimeAverage_03[_n-1]*(_n-1) + i_03)/(_n) if _n>=2
by Seller: replace TimeAverage_04 = (TimeAverage_04[_n-1]*(_n-1) + i_04)/(_n) if _n>=2

by Seller: replace TimeAverage_10 = (TimeAverage_10[_n-1]*(_n-1) + i_10)/(_n) if _n>=2
by Seller: replace TimeAverage_11 = (TimeAverage_11[_n-1]*(_n-1) + i_11)/(_n) if _n>=2
by Seller: replace TimeAverage_12 = (TimeAverage_12[_n-1]*(_n-1) + i_12)/(_n) if _n>=2
by Seller: replace TimeAverage_13 = (TimeAverage_13[_n-1]*(_n-1) + i_13)/(_n) if _n>=2
by Seller: replace TimeAverage_14 = (TimeAverage_14[_n-1]*(_n-1) + i_14)/(_n) if _n>=2

by Seller: replace TimeAverage_20 = (TimeAverage_20[_n-1]*(_n-1) + i_20)/(_n) if _n>=2
by Seller: replace TimeAverage_21 = (TimeAverage_21[_n-1]*(_n-1) + i_21)/(_n) if _n>=2
by Seller: replace TimeAverage_22 = (TimeAverage_22[_n-1]*(_n-1) + i_22)/(_n) if _n>=2
by Seller: replace TimeAverage_23 = (TimeAverage_23[_n-1]*(_n-1) + i_23)/(_n) if _n>=2
by Seller: replace TimeAverage_24 = (TimeAverage_24[_n-1]*(_n-1) + i_24)/(_n) if _n>=2

by Seller: replace TimeAverage_30 = (TimeAverage_30[_n-1]*(_n-1) + i_30)/(_n) if _n>=2
by Seller: replace TimeAverage_31 = (TimeAverage_31[_n-1]*(_n-1) + i_31)/(_n) if _n>=2
by Seller: replace TimeAverage_32 = (TimeAverage_32[_n-1]*(_n-1) + i_32)/(_n) if _n>=2
by Seller: replace TimeAverage_33 = (TimeAverage_33[_n-1]*(_n-1) + i_33)/(_n) if _n>=2
by Seller: replace TimeAverage_34 = (TimeAverage_34[_n-1]*(_n-1) + i_34)/(_n) if _n>=2

by Seller: replace TimeAverage_40 = (TimeAverage_40[_n-1]*(_n-1) + i_40)/(_n) if _n>=2
by Seller: replace TimeAverage_41 = (TimeAverage_41[_n-1]*(_n-1) + i_41)/(_n) if _n>=2
by Seller: replace TimeAverage_42 = (TimeAverage_42[_n-1]*(_n-1) + i_42)/(_n) if _n>=2
by Seller: replace TimeAverage_43 = (TimeAverage_43[_n-1]*(_n-1) + i_43)/(_n) if _n>=2
by Seller: replace TimeAverage_44 = (TimeAverage_44[_n-1]*(_n-1) + i_44)/(_n) if _n>=2

* Change the format
format TimeAverage_00-TimeAverage_44 %9.2f

* Export as .xlsx form for further analysis
forvalues j = 1(1)15{
  so Seller_number date min_date device_id Code
  export exc TimeAverage_00-TimeAverage_44 using "$seller_dist" if Seller_number==`j', sh("Seller_`j'",replace) first(var) nolabel
}

restore

**## Generate Mean and Median actions
preserve

collapse (mean) mean_deviation = self_net_price_1 (median) median_deviation = self_net_price_1, by(self_net_price_bins_1)
format mean_deviation median_deviation %9.2f

gen id=1
tostring self_net_price_bins_1, replace
replace self_net_price_bins_1 = "_bin_" + self_net_price_bins_1
reshape wide mean_deviation median_deviation, i(id) j(self_net_price_bins_1, string)
drop id

export exc mean_* using "$seller_dist" if _n == 1, sh("Actions_Mean",replace) first(var) nolabel
export exc median_* using "$seller_dist" if _n == 1, sh("Actions_Median",replace) first(var) nolabel

restore


**# Generate Data for Selected Devices
**## All Listings
local list "Price Ref_Price"

foreach i in `list'{

use "$Seller_15", clear
keep if Seller == "GadgetPickup" | Seller == "AMS Traders Inc." | Seller == "UPGRADE SOLUTION INC" | Seller == "SaveGadget" | Seller == "Certified Cell, Inc"

* compute average prices for top 5 sellers
preserve 
bysort Seller: egen Average=mean(`i')
bysort Seller: gen index=_n
keep if index==1
keep Average Seller

ds, has(type numeric)
    foreach var of varlist `r(varlist)' {
        replace `var' = round(`var', 0.01)
        format `var' %9.2f
    }
    
tempfile Swappa_average
save "`Swappa_average'"

restore

* define the selected categories (the selection is finished by considering both the number of total device listing and selling and the try to maintain a wide range of different generations)
gen category = device_id + "_" + Storage + "_" + Condition

keep if category=="apple-iphone-11_64 GB_Good"|category=="apple-iphone-se-2nd-gen_64 GB_Good"|category=="apple-iphone-12_64 GB_Good"|category=="apple-iphone-xr_64 GB_Good"|category=="apple-iphone-13_128 GB_Mint"|category=="apple-iphone-12-pro_128 GB_Good"|category=="apple-iphone-12-pro-max_128 GB_Good"|category=="apple-iphone-se-3rd-gen-2022_64 GB_Good"|category=="apple-iphone-13-pro_128 GB_Good"|category=="apple-iphone-14-pro-max_128 GB_Mint"|category=="apple-iphone-13_128 GB_Fair"|category=="apple-iphone-13_128 GB_Good"

bysort Seller category: egen average_price=mean(`i') 

bysort Seller category: gen index=_n

keep if index==1

replace category=subinstr(category," ","",.)
replace category=subinstr(category,"apple-iphone","",.)
replace category=subinstr(category,"-","",.)
replace category=subinstr(category,"2022","",.)
replace category=trim(category)
keep Seller average_price category
reshape wide average_price, i(Seller) j(category) string

rename average_price11_64GB_Good iPhone_11_64GB_Good
rename average_price12_64GB_Good  iPhone_12_64GB_Good
rename average_price12pro_128GB_Good iPhone_12_Pro_128GB_Good
rename average_price12promax_128GB_Good iPhone_12_Pro_Max_128GB_Good
rename average_price13_128GB_Mint iPhone_13_128GB_Mint
rename average_price13pro_128GB_Good iPhone_13_Pro_128GB_Good
rename average_price14promax_128GB_Mint iPhone_14_Pro_Max_128GB_Mint
rename average_pricese2ndgen_64GB_Good iPhone_SE_2nd_Gen_64GB_Good
rename average_pricese3rdgen_64GB_Good  iPhone_SE_3rd_Gen_64GB_Good
rename average_pricexr_64GB_Good iPhone_XR_64GB_Good
rename average_price13_128GB_Fair iPhone_13_128GB_Fair
rename average_price13_128GB_Good iPhone_13_128GB_Good

* order by the number of listings
order Seller iPhone_11_64GB_Good iPhone_12_Pro_Max_128GB_Good iPhone_12_Pro_128GB_Good iPhone_12_64GB_Good iPhone_13_Pro_128GB_Good iPhone_13_128GB_Mint iPhone_13_128GB_Fair iPhone_13_128GB_Good iPhone_14_Pro_Max_128GB_Mint iPhone_SE_2nd_Gen_64GB_Good iPhone_SE_3rd_Gen_64GB_Good iPhone_XR_64GB_Good

ds, has(type numeric)
    foreach var of varlist `r(varlist)' {
        replace `var' = round(`var', 0.01)
        format `var' %9.2f
    }

merge 1:1 Seller using "`Swappa_average'", keep(match) nogen

order Seller Average
gen order=1 if Seller=="AMS Traders Inc."
replace order=2 if Seller=="GadgetPickup"
replace order=3 if Seller=="SaveGadget"
replace order=4 if Seller=="Certified Cell, Inc"
replace order=5 if Seller=="UPGRADE SOLUTION INC"

sort order
drop order

export delimited using "matlab/data/Swappa_`i'.csv", replace

}

copy "matlab/data/Swappa_Price.csv" "$listing_price", replace
erase "matlab/data/Swappa_Price.csv" 

copy "matlab/data/Swappa_Ref_Price.csv" "$ref_price", replace
erase "matlab/data/Swappa_Ref_Price.csv"


**## All Sold Listings

local list "Price Ref_Price"
foreach i in `list'{
use "$Seller_15", clear
keep if Sold_Today==1
keep if Seller=="GadgetPickup"|Seller=="AMS Traders Inc."|Seller=="UPGRADE SOLUTION INC"|Seller=="SaveGadget"|Seller=="Certified Cell, Inc"

* compute the average prices for 5 sellers
preserve 
bysort Seller: egen Average=mean(`i')
bysort Seller: gen index=_n
keep if index==1
keep Average Seller

ds, has(type numeric)
    foreach var of varlist `r(varlist)' {
        replace `var' = round(`var', 0.01)
        format `var' %9.2f
    }

tempfile Swappa_average
save "`Swappa_average'", replace

restore

* define the selected categories (the selection is finished by considering both the number of total device listing and selling and the try to maintain a wide range of different generations)
gen category = device_id + "_" + Storage + "_" + Condition

keep if category=="apple-iphone-11_64 GB_Good"|category=="apple-iphone-se-2nd-gen_64 GB_Good"|category=="apple-iphone-12_64 GB_Good"|category=="apple-iphone-xr_64 GB_Good"|category=="apple-iphone-13_128 GB_Mint"|category=="apple-iphone-12-pro_128 GB_Good"|category=="apple-iphone-12-pro-max_128 GB_Good"|category=="apple-iphone-se-3rd-gen-2022_64 GB_Good"|category=="apple-iphone-13-pro_128 GB_Good"|category=="apple-iphone-14-pro-max_128 GB_Mint"|category=="apple-iphone-13_128 GB_Fair"|category=="apple-iphone-13_128 GB_Good"


bysort Seller category: egen average_price=mean(`i') 

bysort Seller category: gen index=_n

keep if index==1

replace category=subinstr(category," ","",.)
replace category=subinstr(category,"apple-iphone","",.)
replace category=subinstr(category,"-","",.)
replace category=subinstr(category,"2022","",.)
replace category=trim(category)
keep Seller average_price category
reshape wide average_price, i(Seller) j(category) string

rename average_price11_64GB_Good iPhone_11_64GB_Good
rename average_price12_64GB_Good  iPhone_12_64GB_Good
rename average_price12pro_128GB_Good iPhone_12_Pro_128GB_Good
rename average_price12promax_128GB_Good iPhone_12_Pro_Max_128GB_Good
rename average_price13_128GB_Mint iPhone_13_128GB_Mint
rename average_price13pro_128GB_Good iPhone_13_Pro_128GB_Good
rename average_price14promax_128GB_Mint iPhone_14_Pro_Max_128GB_Mint
rename average_pricese2ndgen_64GB_Good iPhone_SE_2nd_Gen_64GB_Good
rename average_pricese3rdgen_64GB_Good  iPhone_SE_3rd_Gen_64GB_Good
rename average_pricexr_64GB_Good iPhone_XR_64GB_Good
rename average_price13_128GB_Fair iPhone_13_128GB_Fair
rename average_price13_128GB_Good iPhone_13_128GB_Good

* order by the number of listings
order Seller iPhone_11_64GB_Good iPhone_12_Pro_Max_128GB_Good iPhone_12_Pro_128GB_Good iPhone_12_64GB_Good iPhone_13_Pro_128GB_Good iPhone_13_128GB_Mint iPhone_13_128GB_Fair iPhone_13_128GB_Good iPhone_14_Pro_Max_128GB_Mint iPhone_SE_2nd_Gen_64GB_Good iPhone_SE_3rd_Gen_64GB_Good iPhone_XR_64GB_Good

ds, has(type numeric)
    foreach var of varlist `r(varlist)' {
        replace `var' = round(`var', 0.01)
        format `var' %9.2f
    }

merge 1:1 Seller using "`Swappa_average'", keep(matched) nogen

order Seller Average
gen order=1 if Seller=="AMS Traders Inc."
replace order=2 if Seller=="GadgetPickup"
replace order=3 if Seller=="SaveGadget"
replace order=4 if Seller=="Certified Cell, Inc"
replace order=5 if Seller=="UPGRADE SOLUTION INC"

sort order
drop order

export delimited using "matlab/data/Swappa_`i'.csv", replace

}

copy "matlab/data/Swappa_Price.csv" "$soldprice", replace
erase "matlab/data/Swappa_Price.csv" 

copy "matlab/data/Swappa_Ref_Price.csv" "$soldrefprice", replace
erase "matlab/data/Swappa_Ref_Price.csv"


**# Gazelle Data for Matlab Used
* Merge sell and buy prices
/* Notes: Before running this part, one should run part1 of "gazelle_data.do" to get the gazelle data.  */
use "$raw_gazelle_sell", clear
merge 1:1 device storage condition date using "$raw_gazelle_buy", keep(3) nogen

bysort device storage condition: egen average_sell_price=mean(sell_price)
bysort device storage condition: egen average_buy_price=mean(buy_price)
sort device storage condition date

local obs_num = _N + 1
set obs `obs_num'
replace device = "Average" in `obs_num'
gen id = _n
replace id = 0 if id == `obs_num'
summarize sell_price
replace average_sell_price = r(mean) if device == "Average"

summarize buy_price
replace average_buy_price = r(mean) if device == "Average"

gen markup = average_sell_price - average_buy_price
gen perc_markup = average_buy_price / average_sell_price

gen category = device + "_" + storage + "_" + condition
order category
keep category average_buy_price average_sell_price markup perc_markup
bys category: keep if _n == 1

*selection 2
keep if category=="iPhone 11_64GB_Good" | category=="iPhone SE 2nd Gen_64GB_Good" | category=="iPhone 12_64GB_Good"  ///
  | category=="iPhone XR_64GB_Good" | category=="iPhone 13_128GB_Excellent" | category=="iPhone 12 Pro_128GB_Good"  ///
  | category=="iPhone 12 Pro Max_128GB_Good" | category=="iPhone SE 3rd Gen_64GB_Good" | category=="iPhone 13 Pro_128GB_Good" ///
  | category=="iPhone 14 Pro Max_128GB_Excellent" | category=="Average__" | category == "iPhone 13_128GB_Fair" | category == "iPhone 13_128GB_Good"

xpose, clear

rename v1 Average
rename v2 iPhone_11_64GB_Good
rename v3 iPhone_12_Pro_Max_128GB_Good
rename v4 iPhone_12_Pro_128GB_Good
rename v5 iPhone_12_64GB_Good
rename v6 iPhone_13_Pro_128GB_Good
rename v7 iPhone_13_128GB_Mint
rename v8 iPhone_13_128GB_Fair
rename v9 iPhone_13_128GB_Good
rename v10 iPhone_14_Pro_Max_128GB_Mint
rename v11 iPhone_SE_2nd_Gen_64GB_Good
rename v12 iPhone_SE_3rd_Gen_64GB_Good
rename v13 iPhone_XR_64GB_Good

drop if Average==.

ds, has(type numeric)
foreach var of varlist `r(varlist)' {
    format `var' %9.2f
}

export excel using "$gazelle_data", firstrow(variables) replace


**# Reference Prices of Top 2 Sellers
use "$Seller_15", clear

keep if Seller_num==1 | Seller_num==2

forvalues j = 1(1)2{
    so Seller_number date min_date device_id Code
    export exc Ref_Price using "$refpricetop2" if Seller_number==`j', sh("Seller_`j'",replace) first(var) nolabel
}
