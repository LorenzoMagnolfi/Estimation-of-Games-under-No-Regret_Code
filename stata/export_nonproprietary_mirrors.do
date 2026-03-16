clear all
set more off
version 18
capture log close

capture mkdir "output"
capture mkdir "output/stata"
log using "output/stata/export_nonproprietary_mirrors.log", replace text

local repo_root `"`c(pwd)'"'
local mirror_root `"`repo_root'/open_data"'

capture mkdir "`mirror_root'"
capture mkdir "`mirror_root'/data"
capture mkdir "`mirror_root'/matlab"
capture mkdir "`mirror_root'/matlab/data"

program define export_dta_dir
    syntax , Source(string) Target(string)

    capture mkdir "`target'"
    local files : dir "`source'" files "*.dta"

    foreach file of local files {
        di as txt "Converting DTA: `source'/`file'"
        use "`source'/`file'", clear
        local base = subinstr("`file'", ".dta", "", .)
        export delimited using "`target'/`base'.csv", replace
    }
end

program define export_excel_dir
    syntax , Source(string) Target(string)

    capture mkdir "`target'"

    local xlsx_files : dir "`source'" files "*.xlsx"
    foreach file of local xlsx_files {
        quietly import excel using "`source'/`file'", describe
        local n = r(N_worksheet)
        local base = subinstr("`file'", ".xlsx", "", .)

        if `n' == 1 {
            local sheet = r(worksheet_1)
            import excel using "`source'/`file'", sheet("`sheet'") firstrow clear
            export delimited using "`target'/`base'.csv", replace
        }
        else {
            capture mkdir "`target'/`base'"
            forvalues i = 1/`n' {
                local sheet = r(worksheet_`i')
                local safe = strtoname("`sheet'")
                import excel using "`source'/`file'", sheet("`sheet'") firstrow clear
                export delimited using "`target'/`base'/`safe'.csv", replace
            }
        }
    }

    local xls_files : dir "`source'" files "*.xls"
    foreach file of local xls_files {
        quietly import excel using "`source'/`file'", describe
        local n = r(N_worksheet)
        local base = subinstr("`file'", ".xls", "", .)

        if `n' == 1 {
            local sheet = r(worksheet_1)
            import excel using "`source'/`file'", sheet("`sheet'") firstrow clear
            export delimited using "`target'/`base'.csv", replace
        }
        else {
            capture mkdir "`target'/`base'"
            forvalues i = 1/`n' {
                local sheet = r(worksheet_`i')
                local safe = strtoname("`sheet'")
                import excel using "`source'/`file'", sheet("`sheet'") firstrow clear
                export delimited using "`target'/`base'/`safe'.csv", replace
            }
        }
    }
end

export_dta_dir, source("data/Raw_Data_Clean") target("`mirror_root'/data/Raw_Data_Clean")
export_dta_dir, source("data/intermediate") target("`mirror_root'/data/intermediate")
export_dta_dir, source("data/gazelle_data") target("`mirror_root'/data/gazelle_data")
export_dta_dir, source("data/Swappa_Daily_Data_Stata") target("`mirror_root'/data/Swappa_Daily_Data_Stata")

export_excel_dir, source("data/Raw_Data_Clean") target("`mirror_root'/data/Raw_Data_Clean")
export_excel_dir, source("data/Swappa_Daily_Data") target("`mirror_root'/data/Swappa_Daily_Data")
export_excel_dir, source("data/Decluttr_Daily_Data") target("`mirror_root'/data/Decluttr_Daily_Data")
export_excel_dir, source("matlab/data") target("`mirror_root'/matlab/data")

di as result "Finished exporting non-proprietary mirrors to `mirror_root'."
log close
