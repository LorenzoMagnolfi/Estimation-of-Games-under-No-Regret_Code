***** This file defines file file and location global macros *****

* Source-data folders
gl swappa_daily "data/Swappa_Daily_Data"
gl swappa_daily_stata "data/Swappa_Daily_Data_Stata"
gl decluttr_daily "data/Decluttr_Daily_Data"
gl decluttr_daily_stata "data/Decluttr_Daily_Data_Stata"
gl raw_reference "data/Raw_Data_Clean"
gl intermediate_dir "data/intermediate"

* Raw data
gl raw_data_cleaned "data/intermediate/raw_data_cleaned.dta"
gl raw_decluttr "data/intermediate/raw_decluttr.dta"
gl raw_swappa "data/intermediate/raw_swappa.dta"
gl raw_gazelle "data/gazelle_data"  // folder of gazelle data
gl raw_gazelle_sell "$raw_gazelle/gazelle_sell_prices_0727_0908.dta"
gl raw_gazelle_buy "$raw_gazelle/gazelle_buy_prices_0729_0908.dta"

* Data for 15 and all sellers
gl Seller_15 "data/intermediate/price_res_5bins_15_sellers.dta"
gl Seller_all "data/intermediate/price_res_5bins_all_sellers.dta"

* Output data for matlab
gl sale_prob      "matlab/data/sale_probability_5bins_res1.xlsx"
gl seller_dist    "matlab/data/SellerDistribution_15_sellers_res1.xlsx"

gl listing_price  "matlab/data/SwappaListingPrice.csv"
gl ref_price	  "matlab/data/RefPrice.csv"
gl soldprice	  "matlab/data/SwappaSoldPrice.csv"
gl soldrefprice	  "matlab/data/RefSoldPrice.csv"

gl gazelle_data   "matlab/data/Gazelle_data.xlsx"

gl refpricetop2   "matlab/data/Ref_Price_for_Device_Day_Top2.xlsx"

* Figure
gl Figure8  "output/stata/figures/Figure 8.pdf"
gl Figure9  "output/stata/figures/Figure 9.pdf"
gl Figure10  "output/stata/figures/Figure 10.pdf"
gl Figure11  "output/stata/figures/Figure 11.pdf"
gl Figure12  "output/stata/figures/Figure 12.pdf"

gl Figure8do  "stata/Figure 8.do"
gl Figure9do  "stata/Figure 9.do"
gl Figure10do  "stata/Figure 10.do"
gl Figure11do  "stata/Figure 11.do"
gl Figure12do  "stata/Figure 12.do"

* Table
gl appendixtable_device 		"output/stata/tables/desc_deviceday.tex"
gl appendixtable_seller 		"output/stata/tables/desc_seller.tex"
gl appendixtable_largeseller 	"output/stata/tables/Large Seller Description.tex"
gl appendix_gazelle 			"output/stata/tables/gazelle_price_gap.tex"
