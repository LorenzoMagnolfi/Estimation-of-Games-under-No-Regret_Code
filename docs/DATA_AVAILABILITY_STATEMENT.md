# Data Availability Statement

## Overview

This replication package contains all data and code needed to reproduce the results in "Estimation of Games under No Regret: Structural Econometrics for AI" (Lomys and Magnolfi). All datasets used in the paper were collected by the authors and are included in the package. No restricted-access or proprietary data are used.

## Data Sources

### 1. Swappa Marketplace Data

**Description.** Daily listing-level data from Swappa (swappa.com), a decentralized US marketplace for used smartphones. Each observation is a device-day pair containing price, seller identity, device characteristics (model, storage, condition, color), and a unique listing code. A second pass over individual listing pages records transaction outcomes (sold status, creation and update dates).

**Collection method.** Web scraping using Python (`requests` + `BeautifulSoup`). A daily automated job ran at approximately 12:00 AM Central Time. The scraper iterated over a manually curated list of popular iPhone models (`DEVICES.xlsx`), collected all active listings for each model, and saved one Excel file per day. A separate pass scraped individual listing detail pages to obtain transaction status.

**Collection period.** July 11, 2023 to December 23, 2023 (166 daily files). Three supplemental daily files (December 29--31, 2023) were collected for the seller-selection step only.

**Access.** Swappa listing pages were publicly viewable without login. The scraper carried a browser cookie header for reliability, but authentication was not required to browse listings. No API key or paid subscription was required.

**Location in package.** `data/Swappa_Daily_Data/` (166 raw `.xlsx` files); `data/Raw_Data_Clean/selling_status.xls` (transaction outcomes); `data/Raw_Data_Clean/Swappa_Scrape_1229_sale.dta`, `_1230_sale.dta`, `_1231_sale.dta` (supplemental scrapes for seller selection).

**Script.** `python/Swappa_Scrape.ipynb` (scraping); `stata/raw_data_clean_for_swappa.do` (cleaning, called by `stata/master.do`).

### 2. Decluttr Reference Price Data

**Description.** Daily product-level data from Decluttr (decluttr.com), a centralized US platform for buying and selling used smartphones. Each observation is a device-condition-day triple containing the posted resale price. These prices serve as reference prices in the empirical model.

**Collection method.** Web scraping using Python (`requests`). The scraper queried Decluttr's public product browse API endpoint (`search-backend.eus.live.channels.em-infra.com/api//browse`) with pagination, extracting product name, price, and device attributes from the JSON response. One Excel file was saved per day.

**Collection period.** July 11, 2023 to December 23, 2023 (166 daily files), matching the Swappa collection window.

**Access.** The API endpoint was publicly accessible without authentication.

**Location in package.** `data/Decluttr_Daily_Data/` (166 raw `.xlsx` files).

**Script.** `python/Decluttr_Scrape.py` (scraping); `stata/raw_data_clean_for_decluttr.do` (cleaning).

### 3. Gazelle Price Data

**Description.** Daily device-level buy and sell prices from Gazelle (gazelle.com), a centralized buyback and resale platform for used smartphones. These data are used in the markup comparison between Swappa sellers and centralized platforms (Section 5.3 and Appendix Table in the paper).

**Collection method.** Web scraping using Python (`selenium` + ChromeDriver). Separate scripts collected sell prices (consumer-facing resale prices) and buy prices (trade-in offers) by automating navigation through Gazelle's product configuration pages (selecting storage, color, condition options). One CSV file was saved per day per price type.

**Collection period.**
- Sell prices (old format): July 1, 2024 to July 21, 2024 (20 files; one missing day: July 14).
- Gap: July 22--26, 2024 (no data; Gazelle website redesign).
- Sell prices (new format, includes color): July 27, 2024 to September 8, 2024 (44 files).
- Buy prices: July 29, 2024 to September 8, 2024 (39 files; missing dates: August 19, August 22, September 3).

**Access.** Gazelle product pages were publicly accessible without authentication.

**Location in package.** `data/gazelle_data/240701_240721/` (old sell format); `data/gazelle_data/sell_prices/` (new sell format); `data/gazelle_data/buy_prices/` (buy prices); `data/gazelle_data/notes_gazelle_data.txt` (collection notes).

**Script.** `python/scrape_gazelle_sell.py` (sell, new format); `python/scrape_gazelle_buy.py` (buy); `stata/gazelle_data.do` (cleaning). Note: the original sell-price scraper used for July 1--21 data (before the website redesign) is not included; the raw CSV files for that period are provided directly.

### 4. Apple Launch Prices

**Description.** A manually compiled lookup table of manufacturer suggested retail prices (MSRP) for each iPhone generation included in the analysis. Used as an upper-bound filter on listing prices (keeping listings priced at most 120% of MSRP).

**Collection method.** Manual Google searches; prices verified across multiple retail sources. Exact source URLs were not recorded at the time of collection.

**Location in package.** `data/Raw_Data_Clean/Price_Apple.xlsx`.

### 5. Crosswalk and Auxiliary Files

| File | Description | Location |
|------|-------------|----------|
| `DEVICES.xlsx` | List of iPhone models defining the Swappa scraping universe | `data/Raw_Data_Clean/` |
| `Swappa and Decluttr device match.xlsx` | Crosswalk mapping Decluttr `ProductLine` to Swappa `device_id` | `data/Raw_Data_Clean/` |
| `Swappa_0711_1223_raw.xlsx` | Combined raw Swappa data (intermediate, produced by cleaning script) | `data/Raw_Data_Clean/` |

## Data That Cannot Be Re-Collected

The following items relate to data that cannot be exactly reproduced by re-running the scraping scripts, because the underlying websites have changed or the original collection environment is no longer available:

1. **Swappa and Decluttr listings (2023).** The scraped data reflect the state of these marketplaces during the July--December 2023 collection window. Historical listings are no longer available on the websites. The raw daily files are included in the package.

2. **Gazelle prices (2024).** Gazelle's website was redesigned during the collection period (July 22--26, 2024 gap). The old sell-price script (`scrape_gazelle (V1_bf_240721).py`) and the scraper input files (`device_gazelle.xlsx`, `gazelle_device_links.csv`) are included for documentation, though they are not needed for replication because all raw Gazelle data files are provided.

3. **Python and browser environment.** Scraping scripts were run on Windows 11 with Python 3.11.5 and the following packages: `pandas` 2.2.1, `requests` 2.32.2, `BeautifulSoup` 4.12.2, `openpyxl` 3.1.2, `selenium` 4.9.0. Gazelle scrapers used Google Chrome 129.0.6668.70 with ChromeDriver 129.0.6668.70. The scripts contain hard-coded paths from the original collection machine; these paths must be updated to re-run the scrapers, though re-running is not required for replication.

## Rights

All data in this package were collected by the authors from publicly accessible websites. No terms-of-service restrictions prohibit redistribution of the scraped data for academic replication purposes. The authors have legitimate access to and permission to redistribute all data included in this package.

## Citation

When using these data, please cite:

> Lomys, Niccolò, and Lorenzo Magnolfi. "Estimation of Games under No Regret: Structural Econometrics for AI." *Journal of Political Economy* (forthcoming).

## Detailed Scraping Protocol

A comprehensive technical reconstruction of the scraping workflow, including field-level documentation, file naming conventions, cleaning transformations, and known caveats, is available in `docs/Scraping_Protocol_Summary_v2.md`.
