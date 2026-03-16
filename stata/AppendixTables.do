
********************************************************************************
*                          Appendix Tables                                     *
********************************************************************************

***********************************************
**#      Statistics at Device-Day Level     ***
***********************************************

**## Panel A: Device-day level information (full sample)
use "$Seller_all", clear

* Compute how many days before devices being sold
bys Code: egen soldtime = count(Sold_Today)
tab soldtime
gen daybfsold = date - Created if Sold_Today == 1

* Count sellers
preserve
duplicates drop Seller, force
local seller_number: display string(_N, "%9.0gc")
restore

* Count observations
local obs_number: display string(_N, "%9.0gc")

* Generate summary statistics for Panel A
estpost tabstat Price Ref_Price daybfsold, s(mean sd p25 p75 min max) col(s)
est sto dscrpt_1
#delimit;
esttab dscrpt_1 using $appendixtable_device, replace
	nonumber nomtitle noobs
	cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
	collabels(none)
	coeflabels(Price "Listing Prices" Ref_Price "Reference prices" daybfsold "Days before sold")
	prehead("\renewcommand{\arraystretch}{1}"
			 "\begin{tabular}{@{\extracolsep{1pt}}lcccccc}"
			 "\toprule"
			 "&  Mean & SD & P25 & P75 & Min & Max \\"
			 "\cline{2-7}"
			 "\underline{\textit{Full sample (`seller_number' sellers, `obs_number' obs)}}&&&&&& \\")
	posthead("")
	prefoot("\addlinespace") postfoot("")
	;
#delimit cr


**## Panel B: Top 15 Sellers
* Select top-15 sellers
use "$Seller_15", clear

* Compute how many days before devices being sold
bys Code: egen soldtime = count(Sold_Today)
tab soldtime
gen daybfsold = date - Created if Sold_Today == 1

* Count observations
local obs_number15: display string(_N, "%9.0gc")

* Generate summary statistics for Panel B
estpost tabstat Price Ref_Price daybfsold, s(mean sd p25 p75 min max) col(s)
est sto dscrpt_2
#delimit;
esttab dscrpt_2 using $appendixtable_device, append
	nonumber nomtitle noobs
	cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
	collabels(none)
	coeflabels(Price "Listing Prices" Ref_Price "Reference prices" daybfsold "Days before sold")
	prehead("\underline{\textit{Top 15 sellers (`obs_number15' obs)}}&&&&&& \\")
	posthead("")
	prefoot("\addlinespace") postfoot("")
	;
#delimit cr



**## Panel C: Most Popular Devices
* Select most popular devices - iPhone 11, iPhone 12, iPhone 13
use "$Seller_all", clear
keep if inlist(device_id, "apple-iphone-11", "apple-iphone-12", "apple-iphone-13")

* Compute how many days before devices being sold
bys Code: egen soldtime = count(Sold_Today)
tab soldtime
gen daybfsold = date - Created if Sold_Today == 1

* Count observations
local obs_popdevice: display string(_N, "%9.0gc")

* Generate summary statistics for Panel B
estpost tabstat Price Ref_Price daybfsold, s(mean sd p25 p75 min max) col(s)
est sto dscrpt_3
#delimit;
esttab dscrpt_3 using $appendixtable_device, append
	nonumber nomtitle noobs
	cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
	collabels(none)
	coeflabels(Price "Listing Prices" Ref_Price "Reference prices" daybfsold "Days before sold")
	prehead("\underline{\textit{Most popular device (`obs_popdevice' obs)}}&&&&&& \\")
	posthead("")
	prefoot("\bottomrule"
	"\end{tabular}") postfoot("")
	;
#delimit cr



***********************************************
**#        Statistics at Seller Level       ***
***********************************************

**## Panel A: Seller level information (full sample)
use "$Seller_all", clear

* Count Sellers
preserve
duplicates drop Seller, force
local seller_number: display string(_N, "%9.0gc")
restore

* Count Observations
local obs_number: display string(_N, "%9.0gc")

* Count number of sellers per day
preserve
duplicates drop Seller date, force
bys date: gen sellernum = _N
duplicates drop date, force
tabstat sellernum, s(mean sd p25 p75 min max) col(s) save
matrix stats1 = r(StatTotal)'
restore

* Count number of devices on the platform per seller-date
preserve
bys Seller date: gen devicenum = _N
tabstat devicenum, s(mean sd p25 p75 min max) col(s) f(%9.0gc) save
matrix stats2 = r(StatTotal)'
restore

* Count number of devices on the platform per seller
preserve
bys Seller: gen devicenum = _N
tabstat devicenum, s(mean sd p25 p75 min max) col(s) save
matrix stats3 = r(StatTotal)'
restore

* Compute revenues from devices being sold
preserve
keep if Sold_Today == 1
collapse (sum) Price, by(Seller)
tabstat Price, s(mean sd p25 p75 min max) col(s) save
matrix stats4 = r(StatTotal)'
restore

* Generate Panel A
matrix stats = stats1 \ stats2 \ stats3 \ stats4
matrix rownames stats = Numseller Ndevselday Ndevsel revenues

#delimit;
esttab matrix(stats, fmt(%9.0fc)) using $appendixtable_seller, replace
  nonumber nomtitle noobs
  cells("mean(fmt(%9.0gc)) sd(fmt(%9.0gc)) p25(fmt(%9.0gc)) p75(fmt(%9.0gc)) min(fmt(%9.0gc)) max(fmt(%9.0gc))")
  collabels(none)
  varlabels(Numseller     "\textbf{\#} of sellers (per day)"
            Ndevselday    "\textbf{\#} of devices (per seller-date)"
            Ndevsel       "\textbf{\#} of device-days (per seller)"
            revenues      "Revenue from devices sold (\\$)")
  prehead("\renewcommand{\arraystretch}{1}"
           "\begin{tabular}{@{\extracolsep{1pt}}lcccccc}"
           "\toprule"
           "& Mean & SD & P25 & P75 & Min & Max \\"
           "\cline{2-7}"
           "\underline{\textit{Full sample (`seller_number' sellers, `obs_number' obs)}}&&&&&& \\")
  posthead("")
  prefoot("\addlinespace") postfoot("")
  ;
#delimit cr


**## Panel B: Top 15 Sellers
use "$Seller_15", clear

* Count observations
local obs_number15: display string(_N, "%9.0gc")

* Count number of devices on the platform (on sale)
preserve
bys Seller date: gen devicenum = _N
tabstat devicenum, s(mean sd p25 p75 min max) col(s) save
mat stats5 = r(StatTotal)'
restore

* Count total number of devices on the platform per seller
preserve
bys Seller: gen devicenum = _N
tabstat devicenum, s(mean sd p25 p75 min max) col(s) save
mat stats6 = r(StatTotal)'
restore

* Compute revenues from devices being sold
preserve
keep if Sold_Today == 1
collapse (sum) Price, by(Seller)
tabstat Price, s(mean sd p25 p75 min max) col(s) save
mat stats7 = r(StatTotal)'
restore

* Generate Panel B
matrix stats = stats5 \ stats6 \ stats7
matrix rownames stats = Ndevselday Ndevsel revenues

#delimit;
esttab matrix(stats, fmt(%9.0fc)) using $appendixtable_seller, append
  nonumber nomtitle noobs
  cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
  collabels(none)
  varlabels(Ndevselday    "\textbf{\#} of devices (per seller-date)"
            Ndevsel       "\textbf{\#} of device-days (per seller)"
            revenues      "Revenue from devices sold (\\$)")
  prehead("\underline{\textit{Top 15 sellers (`obs_number15' obs)}}&&&&&& \\")
  posthead("")
  prefoot("\addlinespace") postfoot("")
  ;
#delimit cr

**## Panel C: Top 5 sellers
use "$Seller_15", clear

* Select top-2 sellers
keep if Seller_number <= 2

* Count observations
local obs_number2: display string(_N, "%9.0gc")

* Count number of devices on the platform (on sale)
preserve
bys Seller date: gen devicenum = _N
tabstat devicenum, s(mean sd p25 p75 min max) col(s) save
mat stats8 = r(StatTotal)'
restore

* Count total number of devices on the platform per seller
preserve
bys Seller: gen devicenum = _N
tabstat devicenum, s(mean sd p25 p75 min max) col(s) save
mat stats9 = r(StatTotal)'
restore

* Compute revenues from devices being sold
preserve
keep if Sold_Today == 1
collapse (sum) Price, by(Seller)
tabstat Price, s(mean sd p25 p75 min max) col(s) save
mat stats10 = r(StatTotal)'
restore

* Generate Panel C
matrix stats = stats8 \ stats9 \ stats10
matrix rownames stats = Ndevselday Ndevsel revenues

#delimit;
esttab matrix(stats, fmt(%9.0fc)) using $appendixtable_seller, append
  nonumber nomtitle noobs
  cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
  collabels(none)
  varlabels(Ndevselday    "\textbf{\#} of devices (per seller-date)"
            Ndevsel       "\textbf{\#} of device-days (per seller)"
            revenues      "Revenue from devices sold (\\$)")
  prehead("\underline{\textit{Top 2 sellers (`obs_number2' obs)}}&&&&&& \\")
  posthead("")
  prefoot("\bottomrule"
  "\end{tabular}") postfoot("")
  ;
#delimit cr



***********************************************
**#              Large Sellers              ***
***********************************************
use "$Seller_15",clear

bysort Seller: egen Seller_Total_Sold=total(Sold_Today)

bysort Seller Code: gen unique_code = _n == 1
egen Seller_Total_Listing = total(unique_code), by(Seller)

bysort Seller: egen Seller_Device_Day = count(Price)

bysort Seller device_id: gen unique_model=_n==1
egen Seller_Total_Model=total(unique_model), by(Seller)


collapse (max)  Seller_Total_Sold (max)  Seller_Total_Listing (max) Seller_Device_Day (max) Seller_Total_Model , by(Seller)

gsort -Seller_Device_Day

order Seller Seller_Device_Day Seller_Total_Listing Seller_Total_Sold Seller_Total_Model


***write the table into latex codes
capture file close texfile
file open texfile using "$appendixtable_largeseller", write replace

file write texfile "\renewcommand{\arraystretch}{1}" _n
file write texfile "\begin{longtable}{lcccc}" _n
file write texfile "\toprule" _n
file write texfile "Seller Company Name & Device-Day Obs. &  Total Listings & Total Sold & Total Models \\" _n
file write texfile "\midrule" _n
file write texfile "\underline{\textit{Top-2 sellers}}&&&& \\" _n

forvalues i = 1/2 {
    local first = 1
    foreach var of varlist _all {
        if `first' {
            file write texfile "`=`var'[`i']'"
            local first = 0
        }
        else {
              file write texfile " &$ `: display %9.0fc `=`var'[`i']''$"
        }
    }
    file write texfile " \\" _n
}

file write texfile "\\" _n
file write texfile "\underline{\textit{Other Top-15 sellers}}&&&& \\" _n

forvalues i = 3/`=_N' {
    local first = 1
    foreach var of varlist _all {
        if `first' {
            file write texfile "`=`var'[`i']'"
            local first = 0
        }
        else {
              file write texfile " &$ `: display %9.0fc `=`var'[`i']''$"
        }
    }
    file write texfile " \\" _n
}

file write texfile "\bottomrule" _n
file write texfile "\end{longtable}" _n
file close texfile



***********************************************
**#           Markups from Gazelle          ***
***********************************************

* Merge sell and buy prices
use "$raw_gazelle/gazelle_sell_prices_0727_0908.dta", clear
merge 1:1 device storage condition date using "$raw_gazelle/gazelle_buy_prices_0729_0908.dta", keep(3) nogen

* Collapse to device-condition level
collapse (mean) sell_price buy_price, by(device storage condition)

* Define local macros
local total_obs: di string(_N, "%9.0gc")

local sellprice: display `"Sell/`\`Ask" Price (\$)"'
local buyprice:  display `"Buy/`\`Bid" Price (\$)"'

local upper_multiplier = 1.3
local lower_multiplier = 1.1

* Compute markups
gen price_gap    = sell_price - buy_price
gen mark_up_low  = sell_price - buy_price * `upper_multiplier'
gen mark_up_high = sell_price - buy_price * `lower_multiplier'

* Summary Statistics
estpost tabstat sell_price buy_price price_gap, s(mean sd p25 p75 min max) c(s)
est sto ds1

sum mark_up_low
local markup_low: di string(r(mean), "%9.2f")
di `markup_low'
sum mark_up_high
local markup_high: di string(r(mean), "%9.2f")
di `markup_high'

* Output .tex file
#delimit;
esttab ds1 using "$appendix_gazelle", replace
	nonumber nomtitle noobs
	cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
	collabels(none)
	coeflabels(sell_price "Sell/\`\`Ask'' Price (\\$)" buy_price "Buy/\`\`Bid'' Price (\\$)" price_gap "Bid-ask spread (\\$)")
	prehead("\renewcommand{\arraystretch}{1}"
	"\begin{tabular}{@{\extracolsep{1pt}}lcccccc}"
	"\toprule"
	"& {Mean} & {SD} & {P25} & {P75} & {Min} & {Max} \\"
	"\cline{2-7}"
	"\underline{\textit{Full Dataset (`total_obs' observations)}} &&&&&& \\")
	posthead("")
	prefoot("\vspace{0.2cm}") postfoot("Markups (\\$) &[`markup_low', `markup_high'] &  &  &  &  &  \\")
	;
#delimit cr

foreach con in "Excellent" "Good" "Fair"{
	preserve
	keep if condition == "`con'"
	local obs: di string(_N, "%9.0gc")

	estpost tabstat sell_price buy_price price_gap, s(mean sd p25 p75 min max) c(s)
	est sto ds_`con'

	sum mark_up_low
	local markup_low: di string(r(mean), "%9.2f")
	di `markup_low'
	sum mark_up_high
	local markup_high: di string(r(mean), "%9.2f")
	di `markup_high'

	if "`con'" == "Fair"{
		#delimit;
		esttab ds_`con' using "$appendix_gazelle", append
			nonumber nomtitle noobs
			cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
			collabels(none)
			coeflabels(sell_price "Sell/\`\`Ask'' Price (\\$)" buy_price "Buy/\`\`Bid'' Price (\\$)" price_gap "Bid-ask spread (\\$)")
			prehead("\underline{\textit{`con' (`obs' observations)}} &&&&&& \\")
			posthead("")
			prefoot("")
			postfoot("Markups (\\$) &[`markup_low', `markup_high'] &  &  &  &  &  \\"
			""
			"\bottomrule"
			"\end{tabular}")
			;
		#delimit cr
	}
	else{
		#delimit;
		esttab ds_`con' using "$appendix_gazelle", append
			nonumber nomtitle noobs
			cells("mean(fmt(%9.2fc)) sd(fmt(%9.2fc)) p25(fmt(%9.2fc)) p75(fmt(%9.2fc)) min(fmt(%9.2fc)) max(fmt(%9.2fc))")
			collabels(none)
			coeflabels(sell_price "Sell/\`\`Ask'' Price (\\$)" buy_price "Buy/\`\`Bid'' Price (\\$)" price_gap "Bid-ask spread (\\$)")
			prehead("\underline{\textit{`con' (`obs' observations)}} &&&&&& \\")
			posthead("")
			prefoot("\vspace{0.2cm}") postfoot("Markups (\\$) &[`markup_low', `markup_high'] &  &  &  &  &  \\")
			;
		#delimit cr
	}
	restore
}
