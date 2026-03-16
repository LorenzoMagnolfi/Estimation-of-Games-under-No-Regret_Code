********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                        merge and clean two dataset                           *
********************************************************************************


clear

set more off, perm

***include the directories
do "stata/macros.do"


***Merge Swappa and decluttr
use "$raw_decluttr", clear

destring price, replace 
collapse (mean) price, by(date device_id Storage Condition)
tempfile decluttr_data_avg
save "`decluttr_data_avg'"


use "$raw_swappa", clear

merge m:1 date device_id Storage Condition using "`decluttr_data_avg'", keep(match) nogenerate
duplicates drop date Code, force
rename price Ref_Price
tempfile Swappa_Decluttr_matched
save "`Swappa_Decluttr_matched'"


***Wash the merged dataset

* Comparing the Swappa price with Apple official price and decluttr reference price
import excel "$raw_reference/Price_Apple.xlsx", firstrow clear
merge 1:m device_id using "`Swappa_Decluttr_matched'", keep(match) nogenerate


gen apple_price2=apple_price*1.2
keep if Price<=apple_price2

keep if (Price>=Ref_Price*0.5) & (Price<=Ref_Price*2) & (Price-Ref_Price<=250) & (Ref_Price-Price<=250)
drop apple_price apple_price2


* Dropping devices on sale for more than 100 days
gl obs_raw = _N

bys Code: egen soldtime = count(Sold_Today)
tab soldtime

gen daybfsold = date - Created if Sold_Today == 1

gen daybfcreated = date - Created
gen hdrd1 = (daybfcreated > 100 & daybfcreated != .)
bys Code: egen Dhdrd1 = max(hdrd1)
drop if Dhdrd1 == 1

di "The observation number of all sellers is:" _N
di "The percentage change of observation of all sellers is:" (${obs_raw} - _N) / ${obs_raw}

keep device_id - Ref_Price
sa "$raw_data_cleaned", replace


*** Select active sellers
* Active sellers in the latest 3 days
preserve
loc dates "1229 1230 1231"
clear
foreach n of loc dates{
  append using "$raw_reference/Swappa_Scrape_`n'_sale.dta"
}

replace Seller = regexr(Seller, " Ratings$", "")
replace Seller = regexr(Seller, "\d+$", "")
replace Seller = strrtrim(Seller)

duplicates drop Seller, force
keep Seller

tempfile active_listing_sellers
sa "`active_listing_sellers'"
restore

* Compute total sales of all sellers
use "$raw_data_cleaned", clear

collapse (sum) seller_total_sold = Sold_Today, by(Seller)

merge 1:1 Seller using "`active_listing_sellers'", keep(match) nogen
gsort -seller_total_sold
keep if _n <= 15
keep Seller
tempfile raw_seller_selection
save "`raw_seller_selection'"
*export exc using "$raw_seller_selection", replace first(var)


*** Generate data for all sellers
use "$raw_data_cleaned",clear

* Create date and time variables
bys Code: egen code_min_date = min(date)
bys Code: egen code_max_date = max(date)
egen min_date = min(date)
egen max_date = max(date)
gen scrape_days = date - min_date, after(max_date)
format *date %td

* Compute price deviation (res_1)
gen deviation = Price - Ref_Price

* Numerize sold_today
replace Sold_Today = (Sold_Today==1)


preserve
* Compute average price by device-storage-condition-date
bys device_id Storage Condition date: egen avg_price = mean(Price)

* Compute number of competitor by device-storage-condition-date
bys device_id Storage Condition date: egen num_competitor = count(Seller)

* Compute competitors' price
gen competitor_price = (avg_price * num_competitor - Price) / (num_competitor - 1)
drop if competitor_price == .

* Compute price residual 1 (res_1)
gen comp_net_price_1 = competitor_price - Ref_Price
gen self_net_price_1 = deviation

* Generate 5 bins by res_1
xtile comp_net_price_bins_1 = comp_net_price_1, nq(5)
xtile self_net_price_bins_1 = self_net_price_1, nq(5)

* save
sa "$Seller_all",replace
restore


*** Generate Data for top15 sellers

* Keep the top 15 sellers
preserve
*import exc using "$raw_seller_selection",clear first
use "`raw_seller_selection'", clear
levelsof Seller, loc(sellers)
restore

gen x = 0
foreach sellername of loc sellers{
  replace x = 1 if Seller == "`sellername'"
}
keep if x == 1
drop x

* Sort sellers by total listing from high to low
bys Seller: egen Seller_total_listing = count(Price)
gen temp = -Seller_total_listing
egen Seller_number = group(temp Seller)
drop temp

* Compute average price by device-storage-condition-date
bys device_id Storage Condition date: egen avg_price = mean(Price)

* Compute number of competitor by device-storage-condition-date
bys device_id Storage Condition date: egen num_competitor = count(Seller)

* Compute competitors' price
gen competitor_price = (avg_price * num_competitor - Price) / (num_competitor - 1)
drop if competitor_price == .

* Compute price residual 1 (res_1)
gen comp_net_price_1 = competitor_price - Ref_Price
gen self_net_price_1 = deviation

* Generate 5 bins by res_1
xtile comp_net_price_bins_1 = comp_net_price_1, nq(5)
xtile self_net_price_bins_1 = self_net_price_1, nq(5)

* save
sa "$Seller_15",replace
