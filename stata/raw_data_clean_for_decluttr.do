********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                         raw data clean for decluttr                          *
********************************************************************************


clear

set more off, perm

***include the directories
do "stata/macros.do"
capture mkdir "$decluttr_daily_stata"


***append the data
local folder "$decluttr_daily"
local file_list: dir "`folder'" file "*.xlsx"


foreach i in `file_list'{
	import excel "`folder'/`i'", clear firstrow
	local datasetname=subinstr("`i'", ".xlsx","",.)
	
	rename Grade Condition
	rename BuiltinMemory Storage
	keep price date ProductLine Storage Color Condition Network


    save "$decluttr_daily_stata/`datasetname'.dta", replace
}

clear

local folder "$decluttr_daily_stata"
local file_list: dir "`folder'" file "*.dta"

foreach i in `file_list'{
	append using "`folder'/`i'"
}

tempfile Decluttr_0711_1223_raw
save "`Decluttr_0711_1223_raw'"

import excel "$raw_reference/Swappa and Decluttr device match.xlsx", firstrow clear

merge 1:m ProductLine using "`Decluttr_0711_1223_raw'", keep(match) nogenerate

rename date date_old
gen date=date(date_old,"YMD")
format date %td
drop date_old

replace Condition="Fair" if Condition=="Good"
replace Condition="Good" if Condition=="Very Good"
replace Condition="Mint" if Condition=="Pristine"

destring price, replace

keep if Network=="UNLOCKED"

save "$raw_decluttr", replace

