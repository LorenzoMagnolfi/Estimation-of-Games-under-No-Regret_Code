if "$PATH" == "" {
	global PATH "`c(pwd)'"
}
cd "$PATH"

capture mkdir "output"
capture mkdir "output/stata"
capture mkdir "output/stata/tables"

capture log close
log using "output/stata/appendix_tables.log", text replace

run "stata/macros.do"
run "stata/AppendixTables.do"

log close
