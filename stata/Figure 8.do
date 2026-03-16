********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                      Figure 8: Daily Price Decision                      *
********************************************************************************

***clean the data

use "$Seller_15", clear


sort Code date
by Code: gen price_change = (Price != Price[_n-1])
drop if price_change==0
by Code: gen time_interval = date - date[_n-1] if price_change
drop price_change 
drop if missing(time_interval)
replace time_interval=6 if time_interval>=6


***pie chart
graph pie, over(time_interval) ///
   pie(1, color(navy)) /// 
   pie(2, color(ebblue)) /// 
   pie(3, color(eltblue)) /// 
   pie(4, color(ltblue)) /// 
   pie(5, color(olive_teal)) ///
   pie(6, color(teal)) /// 
   plabel(_all percent, size(medium) format(%5.2f)) ///
   graphregion(color(white)) /// 
   plotregion(color(white)) /// 
   legend(size(medium) cols(6) rowgap(0.1pt) keygap(0.2pt) colgap(0.6pt) position(6) region(lcolor(black)) label(1 "1 Day") label(2 "2 Days") label(3 "3 Days") label(4 "4 Days") label(5 "5 Days") label(6 "More Than 5 Days"))  
  
   
