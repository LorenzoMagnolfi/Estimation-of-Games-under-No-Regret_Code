********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                          raw data clean for swappa                           *
********************************************************************************


clear

set more off, perm

***include the directories
do "stata/macros.do"
capture mkdir "$swappa_daily_stata"


***append the data
local folder "$swappa_daily"
local file_list: dir "`folder'" file "*.xlsx"


foreach i in `file_list'{
	import excel "`folder'/`i'", clear firstrow
	local datasetname=subinstr("`i'", ".xlsx","",.)

	rename Unnamed8 Battery_Life
    drop A Pics Payment Memory Unnamed*

    save "$swappa_daily_stata/`datasetname'.dta", replace
}

clear

local folder "$swappa_daily_stata"
local file_list: dir "`folder'" file "*.dta"

foreach i in `file_list'{
	append using "`folder'/`i'"
}

***wash the data
drop sale

drop if Price=="No listings to display :("
replace Price=subinstr(Price, "$","",.)
destring Price, replace

rename date date_old
gen date=date(date_old,"YMD")
format date %td
drop date_old

drop Shipping Carrier

replace Seller= subinstr(Seller, "Ratings","",.)
replace Seller = strrtrim(Seller)
replace Seller = regexr(Seller, "\d+$", "")
replace Seller = strrtrim(Seller)

sort date

tempfile Swappa_0711_1223_raw
save "`Swappa_0711_1223_raw'"

export excel "$raw_reference/Swappa_0711_1223_raw.xlsx", replace

***merge the data with the selling status

import excel "$raw_reference/selling_status.xls", clear firstrow


merge 1:m Code using "`Swappa_0711_1223_raw'"


gen Sold_Today=.
replace Sold_Today=1 if Real_Sold==1 & date==Updated

drop _merge

save "$raw_swappa", replace



