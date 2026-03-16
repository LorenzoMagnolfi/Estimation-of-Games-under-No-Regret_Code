********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                   Figure 12: Heatmap for Sale Probability                *
********************************************************************************


***Clean the data
use "$Seller_all", clear

collapse (mean) Sale_Prob=Sold_Today, by(comp_net_price_bins_1 self_net_price_bins_1)

reshape wide Sale_Prob, i(comp_net_price_bins_1) j(self_net_price_bins_1)


forvalues j = 1/5 {
    rename Sale_Prob`j' self_net_price_bins`j'
}

order self_net_price_bins5 self_net_price_bins4 self_net_price_bins3 self_net_price_bins2 self_net_price_bins1, after("comp_net_price_bins")

drop comp_net_price_bins_1

***Heatmap
mkmat *, matrix(Probability)

local overline = uchar(773)
heatplot Probability, values(format(%9.3f) size(medium)) color(plasma, intensity(.6)) aspectratio(1) ///
    xlabel(1 "Bin 5" 2 "Bin 4" 3 "Bin 3" 4 "Bin 2" 5 "Bin 1",labs(medium) nogrid) ///
    ylabel(1 "Bin 1" 2 "Bin 2" 3 "Bin 3" 4 "Bin 4" 5 "Bin 5",labs(medium) nogrid) ///
	xtitle("Own pricing residual {it:{&rho}{subscript:i,n} }",justification(center) margin(vsmall) size(medium)) ///
	ytitle("Competitors' average pricing residual {it:{&rho}`overline'{subscript:-i,n}}", size(medium) justification(center) margin(vsmall))  ///
	keylabels(,format(%9.3f))
	
