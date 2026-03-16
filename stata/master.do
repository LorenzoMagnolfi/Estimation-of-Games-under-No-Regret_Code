********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                                Master File                                   *
********************************************************************************

if "$PATH" == "" {
	global PATH "`c(pwd)'" // Run from repo root, or set PATH manually before calling this file.
}
cd "$PATH"

capture mkdir "output"
capture mkdir "output/stata"
capture mkdir "output/stata/figures"
capture mkdir "output/stata/tables"
capture mkdir "data"
capture mkdir "data/intermediate"
capture mkdir "data/Raw_Data_Clean"
capture mkdir "data/Swappa_Daily_Data_Stata"
capture mkdir "data/Decluttr_Daily_Data_Stata"
capture mkdir "data/gazelle_data"
capture mkdir "data/gazelle_data/sell_prices"
capture mkdir "data/gazelle_data/buy_prices"

run "stata/macros.do"

run "stata/build_intermediates.do"

run "stata/Figure_master.do"

run "stata/AppendixTables.do"

