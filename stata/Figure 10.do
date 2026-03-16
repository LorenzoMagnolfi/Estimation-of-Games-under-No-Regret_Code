********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                        Figure 10: Residual by Models                          *
********************************************************************************


* Keep the device we are interested in

use "$Seller_all", clear

egen group_id = group(device_id)



* Generate the kdensity
kdensity deviation, nograph generate(x pdfx) 
kdensity deviation, nograph generate(pdf1) at(x)
kdensity deviation if device_id == "apple-iphone-11", nograph generate(pdf2) at(x)
kdensity deviation if device_id == "apple-iphone-12", nograph generate(pdf3) at(x)
kdensity deviation if device_id == "apple-iphone-13", nograph generate(pdf4) at(x)

label var pdf1 "All Devices"
label var pdf2 "iPhone 11"
label var pdf3 "iPhone 12"
label var pdf4 "iPhone 13"

* Plot the figure
line pdf1-pdf4 x, sort xtitle("Pricing residuals ($)", size(medium) j(center) m(vsmall) bm(tiny)) ///
    ytitle("Density",size(medium) j(center) m(tiny)) ///
	xlabel(-300(100)300, labsize(medium) nogrid) ///
	ylabel(, labsize(medium) nogrid) ///
	plotregion(lcolor(black)) ///
	lcolor(maroon brown dkgreen eltblue) ///
	legend(size(medium) row(1) rowg(0.1pt) keyg(0.2pt) colg(0.3pt) pos(6) bm(tiny) region(lc(black))) ///
	lpattern(shortdash dash longdash solid) 
