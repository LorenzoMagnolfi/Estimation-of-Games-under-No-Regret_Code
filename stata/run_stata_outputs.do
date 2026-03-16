if "$PATH" == "" {
	global PATH "`c(pwd)'"
}
cd "$PATH"

capture mkdir "output"
capture mkdir "output/stata"
capture mkdir "output/stata/figures"
capture mkdir "output/stata/tables"

capture log close
log using "output/stata/stata_outputs.log", text replace

run "stata/macros.do"
run "stata/Figure_master.do"
run "stata/AppendixTables.do"

log close
