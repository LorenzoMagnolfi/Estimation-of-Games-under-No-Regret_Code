# Scraping and Data-Construction Protocol Summary (v2)

This memo reconstructs the scraping workflow from code and packaged files.
Details not explicitly recorded in the repository are marked **Unknown from files**.
Changes from v1 are flagged with `[UPDATED]`, `[CORRECTED]`, or `[NEW]`.

## Key Evidence Files

- Swappa scraper notebook: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb)
- Decluttr scraper: [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py)
- Gazelle scrapers: [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py), [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py)
- Swappa/Decluttr cleaning and matching: [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do), [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do), [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do)
- Gazelle notes: [Data/gazelle_data/notes_gazelle_data.txt](Data/gazelle_data/notes_gazelle_data.txt)

---

## Swappa

**Source:** Swappa listings pages plus listing detail pages
**URL/domain:** swappa.com, specifically:
- `https://swappa.com/listings/{device}` — listings table
- `https://swappa.com/listing/view/{code}` — individual listing detail (sold/status pass)

**Date range scraped (from packaged files):** 2023-07-11 to 2023-12-23 (166 daily files) in [Data/Swappa_Daily_Data](Data/Swappa_Daily_Data)
**Frequency/timing:** Daily (one file per day by filename); 0:00am Central Time.

**Script used:**
- Scrape: [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb)
- Cleaning: [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do)

**Public/login/manual interaction:**
- Uses `requests` + `BeautifulSoup` HTML parsing (not Selenium).
- Includes a hard-coded `Cookie` header (containing `csrftoken`, `sessionid`, analytics IDs, etc.), indicating the scrape was run with an authenticated browser session. Access to listing data required this cookie.
- Device list is read from [Data/Raw_Data_Clean/DEVICES.xlsx](Data/Raw_Data_Clean/DEVICES.xlsx), maintained manually to define the device universe.
- `[UPDATED]` Hard-coded working directory in notebook: `C:/Users/DELL/Desktop/No Regret/replication codes`. Scripts cannot be re-run directly from the packaged folder path without updating this.

**`[UPDATED]` Two-pass scraping workflow:**
1. **Pass 1 (daily listings):** The main loop iterates over devices in DEVICES.xlsx, calls `scrape_device()` per device, concatenates results, stamps with runtime date, and saves `Swappa_Scrape_MMDD_sale.xlsx`.
2. **Pass 2 (selling status):** A second notebook section reads previously unscrapped listing codes from `Data/Raw_Data_Clean/Swappa_0711_1223_raw.xlsx` (the combined raw file produced by `raw_data_clean_for_swappa.do`), then scrapes each listing's detail page to extract `Real_Sold`, `Created`, and `Updated`. Results are saved to `Data/Raw_Data_Clean/selling_status.xls`. This pass is run after Pass 1 data have been consolidated, not nightly.

**Fields scraped (raw):**
- Listings table: `#`, `Price`, `Pics`, `Carrier`, `Color`, `Storage`, `Model`, `Condition`, `Seller`, `Location`, `Payment`, `Shipping`, `Code`, `device_id`, `Memory`, `sale`, `date`, plus unnamed columns.
- `Unnamed8` column contains battery life information.
- Listing detail pass (saved in status file): `Code`, `Created`, `Updated`, `Real_Sold`.

**Directly scraped vs constructed:**
- Direct: All listing-table variables and detail-page variables.
- Constructed:
  - `device_id` assigned from the loop device index.
  - Daily `date` stamp from runtime.
  - `Battery_Life` from rename of `Unnamed8` (retained in cleaned data).
  - `Sold_Today = 1` if `Real_Sold == 1` and `date == Updated`.

**How listing price, seller, device characteristics, and sold/selling-status were obtained:**
- **Listing price:** From listings table `Price` column; converted to numeric after stripping `$`.
- **Seller:** From listings table `Seller` column, then text-cleaned (substring `"Ratings"` removed, trailing whitespace and trailing digits stripped via regex).
- **Device characteristics:** From listing table columns (`device_id`, `Storage`, `Condition`, etc.) and DEVICES.xlsx crosswalk.
- **Sold/selling-status:** Second pass over listing-view pages produces `Real_Sold`, `Created`, `Updated`, merged back by `Code`. `Sold_Today = 1` if `Real_Sold == 1` and `date == Updated`.

**`[CORRECTED]` File naming convention:**
- Files follow `Swappa_Scrape_MMDD_sale.xlsx` / `Swappa_scrape_MMDD_sale.xlsx` with a documented case change mid-period:
  - **Uppercase** `Swappa_Scrape_`: 0711–0823 (Jul 11 – Aug 23, 2023)
  - **Lowercase** `Swappa_scrape_`: 0824–1107 (Aug 24 – Nov 7, 2023)
  - **Uppercase** `Swappa_Scrape_`: 1108–1223 (Nov 8 – Dec 23, 2023)
- The Stata cleaning script uses a wildcard glob (`*.xlsx`) so it is case-insensitive on Windows but may need attention on case-sensitive filesystems (Linux/Mac).

**Cleaning/harmonization notes (Stata — `raw_data_clean_for_swappa.do`):**
- Drops columns: `A`, `Pics`, `Payment`, `Memory`, `sale`, remaining `Unnamed*` (after renaming `Unnamed8` to `Battery_Life`).
- Drops rows where `Price == "No listings to display :("`.
- Strips `$` from `Price`, casts to numeric.
- Converts date string to Stata date format.
- Drops `Shipping` and `Carrier` columns.
- Normalizes `Seller` text as described above.
- Merges with `selling_status.xls` to add sold status variables.
- Removes duplicate `date`–`Code` pairs after matching.

**Missing days / caveats:**
- No obvious daily gaps in packaged files between 07/11 and 12/23.

---

## Decluttr

**Source:** Decluttr backend browse API
**URL/domain:** `https://search-backend.eus.live.channels.em-infra.com/api//browse` with query parameters `TaxonPermalink=category/cell-phones/apple`, `MaxResults=32`, and paginated `Offset`.

**Date range scraped (from packaged files):** 2023-07-11 to 2023-12-23 (166 files) in [Data/Decluttr_Daily_Data](Data/Decluttr_Daily_Data)
**Frequency/timing:** Daily; 0:00am Central Time.

**Script used:**
- Scrape: [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py)
- Cleaning: [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do)

**Public/login/manual interaction:**
- Direct API calls using `requests` with a browser-like `User-Agent` (Chrome 114).
- No explicit login or cookie required; the API endpoint was publicly accessible.
- `[UPDATED]` Hard-coded working directory: `C:/Users/DELL/Desktop/No Regret/replication codes`.

**Fields scraped (raw API → file column names):**
- `name` → `Name`
- `variants[0]['price']` → `Price`
- Runtime timestamp → `Date`
- From `product_properties` (all keys except `whatsinbox`): `ProductLine`, `BluetoothStandard`, `Brand`, `BuiltInFlash`, `Memory`, `CellularGeneration`, `Color`, `Depth`, `FrontCamera`, `FrontCameraResolution`, `GPS`, `Grade` *(condition label — see rename below)*, `Height`, `NFC`, `Network`, `OS`, `ProcessorCore`, `ProcessorBrand`, `ProcessorType`, `ProductName`, `ProductType`, `RearCameraResolution`, `ScreenResolution`, `ScreenSize`, `ScreenType`, `Sensor`, `RAM`, `Touchscreen`, `Weight`, `Width`, `WiFi`, `iOSSupported`, `USB`, `ProductColor`, `MemoryCardSlot`, `SIMSize`, `TalkTime`, `MPN`, `MSRP`, `GTIN`, `ProcessorSpeed`, `Bluetooth`, `StandbyTime`, `MSRPPrice`, `OIS`, `BuiltinMemory` *(storage label — see rename below)*

**`[NEW]` Raw API field names that differ from cleaned names:**
- `Grade` is the raw condition field — renamed to `Condition` in `raw_data_clean_for_decluttr.do`.
- `BuiltinMemory` is the raw storage field — renamed to `Storage` in the same script.
- These renames happen before any other transformations, so the raw `.xlsx` files contain `Grade` and `BuiltinMemory`, not `Condition` and `Storage`.

**Directly scraped vs constructed:**
- Direct: All API-provided item fields.
- Constructed:
  - Runtime `date` stamp added by scraper.
  - Column renames and a standardized keep set applied in cleaning (`price`, `date`, `ProductLine`, `Storage`, `Color`, `Condition`, `Network`).

**File naming convention:**
- `data_MMDD.xlsx` in [Data/Decluttr_Daily_Data](Data/Decluttr_Daily_Data)

**`[NEW] ⚠ MAJOR CAVEAT — Decluttr_Daily_Data_Stata folder is incomplete:**
- The packaged [Data/Decluttr_Daily_Data_Stata](Data/Decluttr_Daily_Data_Stata) folder contains Stata-converted files **only for 2023-07-11 through 2023-08-05** (roughly 26 files), not the full 166-day range.
- The raw `.xlsx` files in `Decluttr_Daily_Data` cover the full 07/11–12/23 range and are complete.
- `raw_data_clean_for_decluttr.do` generates the `.dta` files from the raw `.xlsx` files on the fly, so re-running the cleaning script will regenerate the missing Stata files and produce the full panel.
- Replicators should run `raw_data_clean_for_decluttr.do` to complete the Stata conversion before proceeding to the merge step.

**Cleaning/harmonization notes (Stata — `raw_data_clean_for_decluttr.do`):**
- Renames `Grade` → `Condition`, `BuiltinMemory` → `Storage`.
- Keeps only: `price`, `date`, `ProductLine`, `Storage`, `Color`, `Condition`, `Network`.
- Condition mapping (Decluttr labels → harmonized labels):
  - `Good` → `Fair`
  - `Very Good` → `Good`
  - `Pristine` → `Mint`
- Keeps unlocked-only records (`Network == "UNLOCKED"`).
- Merges with crosswalk file [Data/Raw_Data_Clean/Swappa and Decluttr device match.xlsx](Data/Raw_Data_Clean/Swappa%20and%20Decluttr%20device%20match.xlsx) on `ProductLine`.
- Note: `destring price, replace` also appears in `merge_and_clean_decluttr_and_swappa.do`, suggesting the price field may be stored as string in the intermediate `.dta` files.

**Missing days / caveats:**
- No obvious daily gap visible in packaged `.xlsx` files for 07/11 to 12/23.
- See Stata folder caveat above.

---

## Gazelle

**Source:** Gazelle product/configuration pages; buy and sell prices collected by separate scripts
**URL/domain:**
- Buy flow: `https://www.gazelle.com/iphone/{device}/unlocked` (device slug from `device_gazelle.xlsx`)
- Sell flow: per-product URLs loaded from `gazelle_device_links.csv`

**Date range scraped:**
- Old sell format (`240701_240721`): 2024-07-01 to 2024-07-21 (20 files; one gap on 2024-07-14).
- Gap 2024-07-22 to 2024-07-26: no data due to website redesign.
- New sell format (`sell_prices`): 2024-07-27 to 2024-09-08 (44 files, appears continuous).
- Buy prices (`buy_prices`): 2024-07-29 to 2024-09-08 (41 files; missing dates: 2024-08-19, 2024-08-22, 2024-09-03).

**Frequency/timing:**
- Appears daily by filename. Time zone/time of day: **Unknown from files**.

**Script used:**
- Sell (new format): [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py)
- Buy: [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py)
- `[NEW]` **Old sell script (pre-July 22 format) is NOT included in the package.** The 20 files in `Data/gazelle_data/240701_240721/` were generated by a first-version script that has not been archived. The new `scrape_gazelle_sell.py` cannot reproduce those files because it includes color, changing the unit of observation.

**Public/login/manual interaction:**
- Selenium automation with Chrome/ChromeDriver. No login or cookie handling.
- Automated clicking through options (storage / color / condition / diagnostic "Yes" buttons).
- `[UPDATED]` Hard-coded paths: ChromeDriver at `D:/chromedriver-win32/chromedriver.exe`; working directory `D:/Dropbox/RA_with_Lorenzo/scrape`; output paths `D:/Dropbox/RA_with_Lorenzo/scrape/Gazelle/buy_prices/` and `.../sell_prices/`.
- `[NEW]` **Two input files are NOT included in the package:**
  - `device_gazelle.xlsx` — device slug list used by the buy script.
  - `gazelle_device_links.csv` — product URL list used by the sell script.
  - Replicators cannot re-run these scrapers without reconstructing these input files.

**Fields scraped:**
- Old sell format (`240701_240721`): `Device`, `Storage`, `Condition`, `Price`, `date`
- New sell format (`sell_prices`): `Device`, `Storage`, `Condition`, `Color`, `Carrier`, `Price`, `date`
- Buy format (`buy_prices`): `Device`, `Storage`, `Condition`, `Carrier`, `Price`, `date`

**Directly scraped vs constructed:**
- Direct: Device title, storage, color, condition labels, displayed prices.
- Constructed/standardized:
  - Condition labels in buy script mapped to `Fair` / `Good` / `Excellent` (from button text `"Scratched or scuffed"` / `"Lightly used"` / `"Flawless or like new"`).
  - `Carrier = Unlocked` hard-coded in both scripts.
  - Date from runtime stamp (`%Y-%m-%d` for date column; `%y%m%d` for filename).
  - Device-string cleanup: `(Unlocked)` suffix and storage size (e.g., `256GB`) removed from device name by regex.

**How buy and sell prices were scraped separately:**
- **Sell script:** Iterates URLs from `gazelle_device_links.csv` → for each product page, loops storage × color × condition combinations by clicking radio inputs; reads displayed sell price per combination.
- **Buy script:** Iterates device slugs from `device_gazelle.xlsx` → navigates to `gazelle.com/iphone/{device}/unlocked` → follows storage-specific links → clicks three diagnostic "Yes" buttons → clicks each condition button and reads payout amount.

**File naming convention:**
- Old sell: `gazelle_YYMMDD.csv` (in `240701_240721/`)
- New sell: `gazelle_sell_prices_YYMMDD.csv` (in `sell_prices/`)
- Buy: `gazelle_buy_prices_YYMMDD.csv` (in `buy_prices/`)

**Cleaning/harmonization notes:**
- Retry logic (up to 3 attempts) for stale-element exceptions on both storage and color loops.
- Failed cases are logged to console but not written to a separate error log.
- Website structure changed around July 22, 2024, requiring a new script and the addition of the color field.

**Missing days / caveats:**
- Explicit gap 2024-07-22 to 2024-07-26 due to site redesign (documented in notes file).
- Old sell data: missing 2024-07-14 (`gazelle_240714.csv` absent).
- Buy data: missing 2024-08-19, 2024-08-22, 2024-09-03.
- Old sell script is not in the package (see Script section above).

---

## Apple Launch Price Source

**Source:** Manually compiled lookup table; prices sourced by Google search verified across multiple sites
**URL/domain:** No single canonical URL; collected via Google searches for each device's launch price.

**File/script used:**
- Input table: [Data/Raw_Data_Clean/Price_Apple.xlsx](Data/Raw_Data_Clean/Price_Apple.xlsx)
- Imported in: [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do)

**Fields in file:**
- `device_id`, `apple_price`

**Manual steps:**
- For each device generation, the launch (MSRP) price was found via Google search and verified across multiple sources, then entered into the Excel file.
- Exact source URLs were not recorded.

**How used in cleaning:**
- Applied as an upper-price cap: keep only listings where `Price <= 1.2 * apple_price`.

---

## Cross-Source Cleaning, Matching, and Selection Rules

### Product Matching and Harmonization

- Decluttr `ProductLine` → Swappa `device_id` mapping via [Data/Raw_Data_Clean/Swappa and Decluttr device match.xlsx](Data/Raw_Data_Clean/Swappa%20and%20Decluttr%20device%20match.xlsx).
- Main merge key between Swappa and Decluttr (collapsed to daily means): `date device_id Storage Condition`.
- Decluttr prices are averaged to one observation per `date × device_id × Storage × Condition` before the merge.

### Outlier/Trimming Rules before Main Pipeline

Applied in [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do):
1. Keep if `Price <= 1.2 * apple_price`
2. Keep if `Price >= 0.5 * Ref_Price` and `Price <= 2 * Ref_Price`
3. Keep if `|Price - Ref_Price| <= 250`
4. Drop listings where `date - Created > 100` days (listing aged more than 100 days at scrape date)

### Duplicates, Missing Entries, Broken Pages

- Duplicate `date`–`Code` pairs dropped after merge.
- Scrapers continue across options/pages on persistent failures; failures are printed to console but not written to a separate log file.
- Gazelle scripts include retry loops (3 attempts) for stale-element exceptions.

### Seller Selection Rules

In [Code/merge_and_clean_decluttr_and_swappa.do](Code/merge_and_clean_decluttr_and_swappa.do):
- Active sellers defined as those with at least one listing in the last 3 scrape dates (Dec 29, 30, 31 — using `Data/Raw_Data_Clean/Swappa_Scrape_1229_sale.dta`, `_1230_sale.dta`, `_1231_sale.dta`).
- Note: these three `.dta` files cover dates beyond the 07/11–12/23 main window, indicating supplemental scrapes were run for seller-selection purposes.
- Sellers ranked by total `Sold_Today` count over the full sample.
- Top 15 sellers by that rank are retained.

---

## Code and Environment Notes

### Scripts/Notebook per Source

| Source | Scrape script | Cleaning script |
|---|---|---|
| Swappa | [Code/Swappa_Scrape.ipynb](Code/Swappa_Scrape.ipynb) | [Code/raw_data_clean_for_swappa.do](Code/raw_data_clean_for_swappa.do) |
| Decluttr | [Code/Decluttr_Scrape.py](Code/Decluttr_Scrape.py) | [Code/raw_data_clean_for_decluttr.do](Code/raw_data_clean_for_decluttr.do) |
| Gazelle sell (new) | [Code/scrape_gazelle_sell.py](Code/scrape_gazelle_sell.py) | [Code/gazelle_data.do](Code/gazelle_data.do) |
| Gazelle sell (old) | **Not in package** | — |
| Gazelle buy | [Code/scrape_gazelle_buy.py](Code/scrape_gazelle_buy.py) | [Code/gazelle_data.do](Code/gazelle_data.do) |

### Python/Package Versions

- Packages used (from import statements):
  - Swappa: `requests`, `bs4` (BeautifulSoup), `pandas`, `openpyxl`, `datetime`, `time`, `os`
  - Decluttr: `requests`, `pandas`, `datetime`, `os`
  - Gazelle: `selenium`, `pandas`, `re`, `time`, `datetime`, `os`
- No `requirements.txt` in the repository; exact historical versions unknown.
- User-Agent string in Swappa and Decluttr headers: Chrome 114.

### Browser/Driver Details (Gazelle)

- Selenium + ChromeDriver.
- Hard-coded ChromeDriver path: `D:/chromedriver-win32/chromedriver.exe` (Windows).
- Both Gazelle scripts use `webdriver.Chrome` with a `Service` object pointing to this path.
- Chrome/ChromeDriver version: **Unknown from files** (must match the Chrome installed at scrape time).

### Working Directory Paths (hard-coded in scripts)

- Swappa + Decluttr: `C:/Users/DELL/Desktop/No Regret/replication codes`
- Gazelle scripts: `D:/Dropbox/RA_with_Lorenzo/scrape`
- All paths assume a specific Windows machine; scripts must be modified to run from the packaged directory.

---

## File Organization and Raw-vs-Clean Status

| Folder | Contents | Status |
|---|---|---|
| `Data/Swappa_Daily_Data/` | 166 daily `.xlsx` files, 07/11–12/23 | Raw scrape output |
| `Data/Decluttr_Daily_Data/` | 166 daily `.xlsx` files, 07/11–12/23 | Raw scrape output |
| `Data/gazelle_data/240701_240721/` | 20 old-format `.csv` files | Raw scrape output |
| `Data/gazelle_data/sell_prices/` | 44 new-format `.csv` files | Raw scrape output |
| `Data/gazelle_data/buy_prices/` | 41 buy-price `.csv` files | Raw scrape output |
| `Data/Raw_Data_Clean/` | Crosswalks, combined raw xlsx, selling status, Stata cleaned files | Mixed: some raw, some derived |
| `Data/Swappa_Daily_Data_Stata/` | Stata-converted daily files (full range) | Derived (generated by cleaning script) |
| `Data/Decluttr_Daily_Data_Stata/` | **Stata-converted files only through 08/05** | **Incomplete — needs cleaning script re-run** |

Because Stata scripts use `save ..., replace`, intermediate outputs will be overwritten on re-run.

---

## Rights / Practical Notes for Replicators

- No explicit redistribution/license statement for scraped marketplace data found in the repository.
- **Raw files are bundled**, so re-scraping is not required for Swappa and Decluttr.
- **Gazelle raw CSVs are bundled**, but:
  - The old sell script is missing; the 20 files in `240701_240721/` cannot be exactly reproduced.s
- **The `Decluttr_Daily_Data_Stata` folder is incomplete.** Replicators must run `raw_data_clean_for_decluttr.do` to regenerate the full Stata panel before running the merge step.
- Scripts contain hard-coded Windows paths that must be updated before re-running.

---

## Summary of Issues / Unknowns to Confirm Manually

| Item | Status |
|---|---|
| Exact Apple launch-price source URLs | Unknown — manual Google search, no URLs recorded |
| Exact run time of day and time zone for daily scrapes | Stated as 0:00am Central Time; not verifiable from files alone |
| Exact Python/package versions at scrape time | Unknown — no requirements.txt |
| Chrome/ChromeDriver version used for Gazelle | Unknown from files |
| Whether raw files were overwritten between scrape and archive | Unknown |
| Old Gazelle sell script (pre-July 22) | Not in package |
| `device_gazelle.xlsx` and `gazelle_device_links.csv` | Not in package |
| Decluttr_Daily_Data_Stata completeness | **Confirmed incomplete — only through 08/05** |
| Swappa cookie/session — whether login was required for public listing data | Strongly implied by hard-coded `sessionid` cookie; confirm whether listing browsing alone required auth |
