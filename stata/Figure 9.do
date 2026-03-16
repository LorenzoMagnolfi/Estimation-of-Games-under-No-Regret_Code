********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*             Figure 9: Listing Prices and Reference Prices - iPhone 11        *
********************************************************************************

***Clean the data

use "$Seller_all", clear


keep if device_id == "apple-iphone-11"


collapse (mean) ListingPrices = Price ReferencePrices = Ref_Price, by(device_id date)

tsset date 
format date %10.0g


***Plot the figure
tsline ListingPrices ReferencePrices,  ///
    ttitle("Date", size(medium) m(vsmall) j(center) bm(tiny))  /// 
    ytitle("Price ($)", size(medium)  m(vsmall) j(center) bm(tiny))  ///
    xlabel(23223 "2023.08.01" 23254 "2023.09.01" 23284 "2023.10.01" 23315 "2023.11.01" 23345 "2023.12.01", labsize(medium) nogrid)  ///
    ylabel(250(50)450, nogrid labsize(medium))  ///
    plotregion(lcolor(black) lwidth(thin)) ///
    legend(size(medium) cols(2) rowgap(0.1pt) keygap(0.2pt) colgap(0.6pt) position(6) region(lcolor(black)) lab(1 "Swappa Listing Price") lab(2 "Decluttr Reference Price"))  




