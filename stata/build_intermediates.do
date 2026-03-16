********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                         Build Intermediate Data                              *
********************************************************************************

if "$PATH" == "" {
	global PATH "`c(pwd)'"
}
cd "$PATH"

capture mkdir "data"
capture mkdir "data/intermediate"
capture mkdir "data/Raw_Data_Clean"
capture mkdir "data/Swappa_Daily_Data_Stata"
capture mkdir "data/Decluttr_Daily_Data_Stata"
capture mkdir "data/gazelle_data"
capture mkdir "data/gazelle_data/sell_prices"
capture mkdir "data/gazelle_data/buy_prices"
capture mkdir "output"
capture mkdir "output/stata"

capture log close
log using "output/stata/build_intermediates.log", text replace

run "stata/macros.do"

capture confirm file "$raw_swappa"
if _rc != 0 {
	capture confirm file "$raw_reference/Swappa_Data.dta"
	if _rc == 0 {
		copy "$raw_reference/Swappa_Data.dta" "$raw_swappa", replace
	}
	else {
		run "stata/raw_data_clean_for_swappa.do"
	}
}

capture confirm file "$raw_decluttr"
if _rc != 0 {
	capture confirm file "$raw_reference/Decluttr_Data.dta"
	if _rc == 0 {
		copy "$raw_reference/Decluttr_Data.dta" "$raw_decluttr", replace
	}
	else {
		run "stata/raw_data_clean_for_decluttr.do"
	}
}

run "stata/merge_and_clean_decluttr_and_swappa.do"
run "stata/gazelle_data.do"
run "stata/generate_matlab_data.do"

log close
