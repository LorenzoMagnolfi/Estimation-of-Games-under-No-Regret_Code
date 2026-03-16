********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                            Prepare Gazelle Data                              *
********************************************************************************

clear

set more off, perm

do "stata/macros.do"

capture mkdir "$raw_gazelle"
capture mkdir "$raw_gazelle/sell_prices"
capture mkdir "$raw_gazelle/buy_prices"

********************************************************************************
* Sell prices
********************************************************************************

clear
tempfile gazelle_sell_combined
save "`gazelle_sell_combined'", emptyok replace

local folder "$raw_gazelle/sell_prices"
local file_list: dir "`folder'" file "gazelle_sell_prices_*.csv"

foreach i in `file_list' {
	import delimited using "`folder'/`i'", clear varnames(1) stringcols(_all)
	keep device storage condition carrier price date
	tostring price date, replace force
	rename price sell_price

	append using "`gazelle_sell_combined'"
	save "`gazelle_sell_combined'", replace
}

use "`gazelle_sell_combined'", clear
replace date = strtrim(date)
gen date_num = date(date, "YMD")
format date_num %td
drop date
rename date_num date
destring sell_price, replace ignore(",$")
collapse (mean) sell_price, by(device storage condition date)
sort device storage condition date
save "$raw_gazelle_sell", replace

********************************************************************************
* Buy prices
********************************************************************************

clear
tempfile gazelle_buy_combined
save "`gazelle_buy_combined'", emptyok replace

local folder "$raw_gazelle/buy_prices"
local file_list: dir "`folder'" file "gazelle_buy_prices_*.csv"

foreach i in `file_list' {
	import delimited using "`folder'/`i'", clear varnames(1) stringcols(_all)
	keep device storage condition carrier price date
	tostring price date, replace force
	rename price buy_price

	append using "`gazelle_buy_combined'"
	save "`gazelle_buy_combined'", replace
}

use "`gazelle_buy_combined'", clear
replace date = strtrim(date)
gen date_num = date(date, "YMD")
format date_num %td
drop date
rename date_num date
destring buy_price, replace ignore(",$")
collapse (mean) buy_price, by(device storage condition date)
sort device storage condition date
save "$raw_gazelle_buy", replace
