********************************************************************************
********************* Estimation of Games Under No Regret **********************
*******************************               **********************************
********************************************************************************
*                            Generate Figures 8-12                             *
********************************************************************************

**ssc install grtext 
**ssc install heatplot
**ssc install grstyle
**ssc install palettes
**ssc install colrspace

clear

graph drop _all

set more off, perm

**Default setting of the figures

set scheme s2color // sets Stata design scheme to default s2color
grstyle init

grstyle set horizontal // set tick labels horizontal
grstyle set compact // overall design compact

grstyle set size 12pt: heading 
grstyle set size 12pt: subheading axis_title 
grstyle set size 10pt: tick_label key_label

grstyle set legend 6, nobox //legend position on 6 o'clock, removes box
grstyle set linewidth thin: major_grid // set grid line thickness to thin
grstyle set linewidth thin: tick // set tick thickness to thin
grstyle set linewidth thin: axisline // set axislines thickness to thin
grstyle set linewidth vthin: xyline // reference line at 0

grstyle color background white // set overall background to white
grstyle set color black*.7: tick tick_label // set tick and tick label color
grstyle set color black*.08: plotregion plotregion_line //set plot area background color
grstyle set color White: axisline major_grid // set axis and grid line color
grstyle set color black*.7: small_body // set note color
grstyle set color white*.08, opacity(0): pbarline // graph bar outline transparent

grstyle set color economist, order(10 1 8 2 3 4 5 6 7 9 11 12 13 14 15)


**include macro
do "stata/macros.do"


**Figure 8: Daily Price Decision
do "$Figure8do"

graph export "$Figure8", as(pdf) replace	


**Figure 9: Listing Prices and Reference Prices - iPhone 11
do "$Figure9do"

graph export "$Figure9", as(pdf) replace	


**Figure 10: Residual by Models  
do "$Figure10do"

graph export "$Figure10", as(pdf) replace	


**Figure 11: Distribution of 5 Bin over Time
do "$Figure11do"

graph export "$Figure11", as(pdf) replace	

**Figure 12: Heatmap for Sale Probability
do "$Figure12do"

graph export "$Figure12", as(pdf) replace	


