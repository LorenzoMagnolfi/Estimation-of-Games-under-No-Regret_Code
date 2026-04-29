# Scraping and Data-Construction Protocol Summary

This memo reconstructs the scraping workflow from code and packaged files. If a detail is not explicitly recorded in the repository, it is marked as **Unknown from files**.

## Key Evidence Files

- Swappa scraper notebook: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb#L59), [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb#L99), [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb#L159)
- Decluttr scraper: [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py#L16), [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py#L25)
- Gazelle scrapers: [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py#L26), [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py#L52)
- Swappa/Decluttr cleaning and matching: [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do#L18), [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do#L46), [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do#L38)
- Gazelle notes: [Data/gazelle_data/notes_gazelle_data.txt](Data/gazelle_data/notes_gazelle_data.txt#L4)

---

## Swappa

**Source:** Swappa listings pages plus listing detail pages  
**URL/domain:** swappa.com, specifically:
- https://swappa.com/listings/{device}
- https://swappa.com/listing/view/{code}

Evidence: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb#L59), [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb#L161)

**Date range scraped (from packaged files):** 2023-07-11 to 2023-12-23 (166 daily files) in [Data/Swappa_Daily_Data](Data/Swappa_Daily_Data)  
**Frequency/timing:** Daily (one file per day by filename); Daily; 0:00am Central Time.

**Script used:**
- Scrape: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb)
- Cleaning: [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do)

**Public/login/manual interaction:**
- Uses requests + HTML parsing (not Selenium).
- Includes a hard-coded Cookie header, suggesting session/cookie-assisted access in practice: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb#L39)
- Device list is read from [Data/Raw_Data_Clean/DEVICES.xlsx](Data/Raw_Data_Clean/DEVICES.xlsx), implying manual maintenance of device universe.

**Fields scraped (raw):**
- Listings table (example headers observed in daily files): `#`, `Price`, `Pics`, `Carrier`, `Color`, `Storage`, `Model`, `Condition`, `Seller`, `Location`, `Payment`, `Shipping`, `Code`, `device_id`, `Memory`, `sale`, `date`, plus unnamed columns.
- Listing detail pass (saved in status file): `Code`, `Created`, `Updated`, `Real_Sold`.

**Directly scraped vs constructed:**
- Direct: Listing-table variables and detail-page variables.
- Constructed later:
  - `device_id` assigned from looped input device.
  - Daily `date` stamp from runtime.
  - `Battery_Life` from `Unnamed8` rename.
  - `Sold_Today = 1` if `Real_Sold == 1` and `date == Updated`.

Evidence: [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do#L26), [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do#L76)

**How listing price, seller, device characteristics, sold/selling-status were obtained:**
- Listing price: from listings table `Price`.
- Seller: from listings table `Seller`, then text-cleaned (`Ratings` removed, trailing digits removed).
- Device characteristics: from listing table columns (`device_id`, `Storage`, `Condition`, etc.) and matching files.
- Sold/selling-status: second pass over listing-view pages generates `Real_Sold`, `Created`, `Updated`, merged back by `Code`--`Sold_Today = 1` if `Real_Sold == 1` and `date == Updated`.

**File naming convention:**
- `Swappa_Scrape_MMDD_sale.xlsx` (and some lowercase `Swappa_scrape_...` variants) in [Data/Swappa_Daily_Data](Data/Swappa_Daily_Data)

**Cleaning/harmonization notes:**
- Drops non-analytic columns and empty listing rows.
- Converts `Price` to numeric.
- Normalizes seller text fields.
- Removes duplicate `date`-`Code` pairs after matching.

Evidence: [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do#L44), [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do#L29)

**Missing days / caveats:**
- No obvious daily gaps in packaged files between 07/11 and 12/23.


---

## Decluttr

**Source:** Decluttr backend browse API  
**URL/domain:** https://search-backend.eus.live.channels.em-infra.com/api//browse with `TaxonPermalink=category/cell-phones/apple`

Evidence: [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py#L16), [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py#L25)

**Date range scraped (from packaged files):** 2023-07-11 to 2023-12-23 (166 files) in [Data/Decluttr_Daily_Data](Data/Decluttr_Daily_Data)  
**Frequency/timing:** Daily; 0:00am Central Time.

**Script used:**
- Scrape: [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py)
- Cleaning: [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do)

**Public/login/manual interaction:**
- Direct API calls using requests and User-Agent.
- No explicit login/cookie workflow in script.

**Fields scraped:**
- Directly from API docs and product properties: `name`, `price` (`variants[0]['price']`), date stamp, and many product attributes in `product_properties` (except `whatsinbox`).
- Full list of feilds scraped: `Name`, `Price`, `Date`, `ProductLine`, `BluetoothStandard`, `Brand`, `BuiltInFlash`, `Memory`, `CellularGeneration`, `Color`, `Depth`, `FrontCamera`, `FrontCameraResolution`, `GPS`, `Condition`, `Height`, `NFC`, `Network`, `OS`, `ProcessorCore`, `ProcessorBrand`, `ProcessorType`, `ProductName`, `ProductType`, `RearCameraResolution`, `ScreenResolution`, `ScreenSize`, `ScreenType`, `Sensor`, `RAM`, `Touchscreen`, `Weight`, `Width`, `WiFi`, `iOSSupported`, `USB`, `ProductColor`, `MemoryCardSlot`, `SIMSize`, `TalkTime`, `MPN`, `MSRP`, `GTIN`, `ProcessorSpeed`, `Bluetooth`, `StandbyTime`, `MSRPPrice`, `OIS`

**Directly scraped vs constructed:**
- Direct: API-provided item fields.
- Constructed:
  - Runtime `date` in scraper.
  - Standardized keep/rename set in cleaning (`price`, `date`, `ProductLine`, `Storage`, `Color`, `Condition`, `Network`).

Evidence: [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do#L28)

**File naming convention:**
- `data_MMDD.xlsx` in [Data/Decluttr_Daily_Data](Data/Decluttr_Daily_Data)

**Cleaning/harmonization notes:**
- Condition mapping:
  - `Good -> Fair`
  - `Very Good -> Good`
  - `Pristine -> Mint`
- Keep unlocked-only records (`Network == "UNLOCKED"`).
- Merge with crosswalk file [Data/Raw_Data_Clean/Swappa and Decluttr device match.xlsx](Data/Raw_Data_Clean/Swappa%20and%20Decluttr%20device%20match.xlsx).

Evidence: [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do#L55), [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do#L61)

**Missing days / caveats:**
- No obvious daily gap visible in packaged files for 07/11 to 12/23.

---

## Gazelle

**Source:** Gazelle product/configuration pages; buy and sell collected by separate scripts  
**URL/domain:**
- Buy flow: https://www.gazelle.com/iphone/{device}/unlocked
- Sell flow: URL list loaded from input CSV

Evidence: [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py#L52), [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py#L26)

**Date range scraped:**
- Notes file says sell scraping began 2024-07-01.
- 2024-07-22 to 2024-07-26: no data due to website update.
- New sell script from 2024-07-27 to 2024-09-08, now including color.
- Buy scraping started 2024-07-29 to 2024-09-08.

Evidence: [Data/gazelle_data/notes_gazelle_data.txt](Data/gazelle_data/notes_gazelle_data.txt#L4)

**Frequency/timing:**
- Appears daily by filename.
- Time zone/time of day not documented.

**Script used:**
- Sell: [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py)
- Buy: [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py)

**Public/login/manual interaction:**
- Selenium automation with Chrome/ChromeDriver.
- No login/cookie handling in these scripts.
- Automated clicking through options (storage/color/condition/diagnostic yes-buttons).
- Input universe files referenced but not included in this package:
  - `device_gazelle.xlsx` (buy script)
  - `gazelle_device_links.csv` (sell script)

**Fields scraped:**
- Old sell format (`240701_240721`): `Device`, `Storage`, `Condition`, `Price`, `date`
- New sell format (`sell_prices`): `Device`, `Storage`, `Condition`, `Color`, `Carrier`, `Price`, `date`
- Buy format (`buy_prices`): `Device`, `Storage`, `Condition`, `Carrier`, `Price`, `date`

**Directly scraped vs constructed:**
- Direct: Device title, storage, color, condition, displayed prices.
- Constructed/standardized:
  - Condition labels mapped in buy script to `Fair`, `Good`, `Excellent`.
  - `Carrier = Unlocked` added in both scripts.
  - Date from runtime stamp.
  - Device-string cleanup by regex.

Evidence: [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py#L87), [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py#L128), [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py#L112)

**How buy and sell were scraped separately:**
- Sell script iterates product links, then loops storage x color x condition combinations and reads sell prices.
- Buy script iterates devices and storage pages, clicks diagnostic and condition options, then reads payout amount.

**File naming convention:**
- `gazelle_sell_prices_YYMMDD.csv`
- `gazelle_buy_prices_YYMMDD.csv`

**Cleaning/harmonization notes:**
- Retry and stale-element handling are implemented; failed cases are logged.
- Website structure changed in late July, requiring new script and variable structure.

Evidence: [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py#L58), [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py#L101), [Data/gazelle_data/notes_gazelle_data.txt](Data/gazelle_data/notes_gazelle_data.txt#L6)

**Missing days / caveats:**
- Explicit gap 2024-07-22 to 2024-07-26 due to site update.
- Additional missing day visible in old sell folder (no `gazelle_240714.csv`).
- Buy folder has missing dates visible from filenames (e.g., 2024-08-19, 2024-08-22, 2024-09-03).

---

## Apple Launch Price Source

**Source:** Local Excel lookup table merged by `device_id`  
**URL/domain:** (https://www.google.com/search)

**File/script used:**
- Input table: [Data/Raw_Data_Clean/Price_Apple.xlsx](Data/Raw_Data_Clean/Price_Apple.xlsx)
- Imported in: [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do#L38)

**Fields in file:**
- `device_id`, `apple_price`

**Manual steps:**
- Directly search the iPhone luanch price for each device generation via Google, verifying through multiple sources

**How used in cleaning:**
- Apply cap `Price <= 1.2 * apple_price`.

Evidence: [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do#L42)

---

## Cross-Source Cleaning, Matching, and Selection Rules

### Product Matching and Harmonization

- Decluttr to Swappa mapping via [Data/Raw_Data_Clean/Swappa and Decluttr device match.xlsx](Data/Raw_Data_Clean/Swappa%20and%20Decluttr%20device%20match.xlsx) in [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do#L46)
- Main merge key between Swappa and Decluttr: `date device_id Storage Condition` in [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do#L28)

### Outlier/Trimming Rules before Main Pipeline

Applied in [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do):
- Keep if `Price <= 1.2 * apple_price`
- Keep if `Price` is between `0.5*Ref_Price` and `2*Ref_Price`
- Keep if absolute price gap to reference is <= 250 dollars both directions
- Drop listings with created-age > 100 days at scrape date

### Duplicates, Missing Entries, Broken Pages

- Duplicates dropped by `date Code` in merged panel.
- Scrapers generally continue across pages/options but log failures and may skip records on persistent failures.
- Gazelle scripts include retry loops for stale elements and timeouts.

### Seller Selection Rules

In [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do#L70):
- Define active sellers as appearing in latest 3 days (`1229`, `1230`, `1231`) using files in [Data/Raw_Data_Clean](Data/Raw_Data_Clean)
- Rank by total `Sold_Today`
- Keep top 15 sellers

---

## Code and Environment Notes

### Scripts/Notebook per Source

- Swappa: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb)
- Decluttr: [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py)
- Gazelle buy: [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py)
- Gazelle sell: [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py)

### Python/Package Versions

- Package imports are visible, but exact historical versions are not pinned in this repository.
- No project-level `requirements.txt` was found.

### Browser/Driver Details

- Gazelle scripts use Selenium + ChromeDriver with hard-coded local path strings and Windows paths.
- Swappa/Decluttr scripts use browser-like User-Agent strings (Chrome 114 in headers).

---

## File Organization and Raw-vs-Clean Status

- Source-like raw daily files:
  - [Data/Swappa_Daily_Data](Data/Swappa_Daily_Data)
  - [Data/Decluttr_Daily_Data](Data/Decluttr_Daily_Data)
  - [Data/gazelle_data](Data/gazelle_data)
- Lightly cleaned/derived files and crosswalks:
  - [Data/Raw_Data_Clean](Data/Raw_Data_Clean)
- Stata-converted daily files:
  - [Data/Swappa_Daily_Data_Stata](Data/Swappa_Daily_Data_Stata)
  - [Data/Decluttr_Daily_Data_Stata](Data/Decluttr_Daily_Data_Stata)

Because scripts frequently save with `replace`, some intermediate outputs can be regenerated/overwritten when rerun.

---


## Rights / Practical Notes to Document

- No explicit redistribution/license statement for scraped marketplace data was found in this repository.
- Suggested statement for replicators:
  - Raw files for Swappa and Decluttr are bundled, so re-scraping is generally not required.
  - Gazelle raw CSV files are bundled, but one missing preprocessing script means exact end-to-end rerun may require an additional conversion step.

---

## Other Protocol Notes Found

- Useful internal note: [Data/gazelle_data/notes_gazelle_data.txt](Data/gazelle_data/notes_gazelle_data.txt)
- No project-level README/protocol/email log specific to scraping was found in Code/Data root trees.

---

## Suggested “Unknowns” to Confirm Manually (if needed)

- Exact Apple launch-price source URLs and collection protocol.
- Exact run time of day and time zone for daily scrapes.
- Exact Python/package versions used at scrape time.
- Whether any raw files were overwritten between original scrape and archive finalization.
