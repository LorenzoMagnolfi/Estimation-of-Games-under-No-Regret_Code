********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                   Figure 11: Distribution of 5 Bin over Time                 *
********************************************************************************


***figure for Seller 1
use "$Seller_15", clear


keep if date==td(01aug2023)  | date==td(01oct2023) | date==td(01dec2023)

preserve

keep if Seller_num==1

contract date self_net_price_bins_1, freq(count)


reshape wide count, i(date) j(self_net_price_bins_1)
reshape long count, i(date) j(self_net_price_bins_1)

by date: egen total=total(count)
by date: gen density=count/total

   
graph bar density, over(self_net_price_bins_1, relabel(1 "B1" 2 "B2" 3 "B3" 4 "B4" 5 "B5") label(labsize(small))) ///
    over(date, relabel(1 "2023.08.01" 2 "2023.10.01" 3 "2023.12.01") label(labsize(medium))) ///
    blabel(bar,  format(%9.2f))  ///
    ytitle("Density", size(large) j(center) m(vsmall))  /// 
    ylabel(0(0.1)0.5, labsize(large) nogrid) ///
    plotregion(lcolor(black) lwidth(thin))  /// 
    yscale(lcolor(none))  ///
    name(bar1, replace) ///
	title("Seller 1", size(large))
	
graph save bar1.gph, replace

restore

preserve

***figure for Seller 2
keep if Seller_num==2

contract date self_net_price_bins_1, freq(count)


reshape wide count, i(date) j(self_net_price_bins_1)
reshape long count, i(date) j(self_net_price_bins_1)

by date: egen total=total(count)
by date: gen density=count/total

   
graph bar density, over(self_net_price_bins_1, relabel(1 "B1" 2 "B2" 3 "B3" 4 "B4" 5 "B5") label(labsize(small))) ///
    over(date, relabel(1 "2023.08.01" 2 "2023.10.01" 3 "2023.12.01") label(labsize(medium))) ///
    blabel(bar,  format(%9.2f))  ///
    ytitle("Density", size(large) j(center) m(vsmall))  /// 
    ylabel(0(0.1)0.5, labsize(large) nogrid) ///
    plotregion(lcolor(black) lwidth(thin))  /// 
    yscale(lcolor(none))  ///
    name(bar2, replace) ///
	title("Seller 2", size(large))
graph save bar2.gph, replace

restore

clear

graph use bar1.gph

graph use bar2.gph

graph combine bar1 bar2, col(2)

**for windows
shell del "bar1.gph"
shell del "bar2.gph"

/*
**for mac
shell rm "bar1.gph"
shell rm "bar2.gph"
*/



