# Provenance Recovery Notes

## Directly useful current-package provenance

- `Replication_2025/Data/gazelle_data/notes_gazelle_data.txt` documents the Gazelle collection timeline.
- It says sell prices were scraped starting on July 1, 2024.
- It records a website-change gap from July 22, 2024 to July 26, 2024 with no sell-price data collected.
- It says the sell-price scraper changed on July 27, 2024 to collect color, changing the observation unit from `Device-Storage-Condition-date` to `Device-Storage-Condition-Color-date`.
- It says buy-price scraping started on July 29, 2024.
- `Replication_2025/Code/Decluttr_Scrape.py` shows the Decluttr pull used a public backend endpoint at `https://search-backend.eus.live.channels.em-infra.com/api//browse` with pagination and wrote one daily workbook per run.

## Older Dropbox archive that helps with protocol history

The older folder `Dropbox/DynamicFoundations/Data` contains historical scraping and data-construction artifacts that are not the current 2023-2024 replication package, but they are still useful for provenance recovery.

- `README (Data Construction).docx` states the older pipeline into MATLAB was driven by `CleanData.do` followed by `ConstructTimeAveragePaths.do`.
- `Code/scrape_swappa.py` documents an hourly Swappa scraper for 10 high-volume unlocked iPhone devices.
- That script scraped listing rows for currently selling devices, recent suggested prices, and recently sold devices.
- It wrote timestamped Excel files using `pd.Timestamp("today")` and then slept for 3600 seconds before repeating.
- `Code/scrape_swappa_update.py` records a Swappa website update on November 3, 2019 and shows the revised scraper paginated through all listing pages, deduplicated by URL, and continued to save one workbook per device and scrape time.
- `Code/ScrapePagesHTML.py` documents a later HTML-archiving step by unique listing URL and notes two exogenous shutdown windows: November 15-20, 2019 and November 28-December 3, 2019.
- That same script also records use of `cfscrape` to bypass a JavaScript-based Cloudflare check.
- `Input/` contains large historical raw or near-raw files such as `unlocked_iphone_1103_0122_v2.csv`, `unlocked_iphone_1103_0122_v3.csv`, and older `.xlsx` and `.dta` intermediates.

## Caveat

These historical `DynamicFoundations/Data` files should be treated as helpful background on earlier collection practice, not as definitive documentation for the current 2023-2024 JPE package. For the current package, the most directly usable provenance still needs to come from the existing 2024 scrape scripts, the Gazelle notes file, and any clarifications from the former RAs.
