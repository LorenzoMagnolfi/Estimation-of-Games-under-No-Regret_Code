# Swappa empirical foundations: evidence and implementation record

Date: 2026-10-20. Status: broad public-source review and descriptive audit completed; no new cost estimator or seller-adoption claim. Read the human exposition in the paper sibling, `JPE Revision/swappa-empirical-foundations-note-2026-10-20.tex`. This file preserves evidence provenance and next-step specifications. The closed learning exercise is governed by `UPGRADE_LEARNING_RULE_CLOSEOUT_2026-10-20.md`.

## Decision

Public evidence supports examining scheduled or batched decisions, selected competitors, price limits, actual-sales feedback, inventory age, and individual-device accounting. It does not support the old expected-profit oracle, exact multiplicative updates, shared learning weights, a five-dollar cost span, or the old finite-sample allowance. The useful next result is an independently evaluated pricing rule and a transparent statement of what links that rule to cost. No informative, empirically validated new marginal-cost interval has been produced.

Do not choose behavioral restrictions by the width of the resulting cost intervals. Do not treat a sales-responsive heuristic as a mathematical no-regret learner without a theorem. A rule can fit every observed price and still leave acquisition costs unidentified.

## Search design and limits

Primary sources were prioritized: Swappa documentation and seller profiles; WholeCell announcements, manuals, and changelogs; Sellercloud channel documentation; commercial repricer manuals and dated announcements; and an original industry study. General search, site-restricted search, exact seller-name searches, and pre-2024 search filters were used. This was a broad targeted review, not a claim to have exhausted the internet.

Queries combined Swappa with repricer, automatic pricing, repricing, API, price synchronization, integration, case study, and 2023. Seller-name queries combined UPGRADE SOLUTION INC / Upgrade Solution, GadgetPickup, AMS Traders, Certified Cell, and SaveGadget with software, repricing, WholeCell, Sellercloud, Linnworks, and ChannelAdvisor. Vendor searches covered WholeCell, Sellercloud, Linnworks, Repricer.com, StreetPricer, Informed, Seller Snap, Amazon, Walmart, and Back Market. Dated queries covered bulk pricing, sales rules, and device accounting before or during 2023.

Result: no verified public disclosure tying any of the five examined sellers to a specific repricer. This is a negative search finding, not proof of non-adoption. There may be private tools, manual spreadsheets, unpublished integrations, or unindexed disclosures. No seller or vendor was contacted. No trial account was created and no proprietary code was inspected.

All URLs below were consulted on October 20, 2026. A displayed date is not an independently verified archival snapshot. Current page text can be revised. No Wayback snapshot was verified. Current functionality should not be silently assigned to 2023. Footer copyright years and years in download paths are not feature-release evidence.

## Evidence register

Each entry separates the proposition established by the source from the inference it cannot support. Labels match the human note where applicable. Summaries are deliberately short; consult the linked primary document for detail.

### Platform, inventory, and seller evidence

- **S1 — Upgrade profile (current)**: https://swappa.com/bp/upgrade-solution-inc . Business identity and service description. No software adoption disclosure. Its eBay-related shipping boilerplate is not proof that a specific 2023 device was cross-listed.
- **S2 — WholeCell/Swappa demo, April 22, 2021**: https://www.wholecell.io/blog/54-demo-swappa-b2b-exchange-wholecell-integrations . Pre-sample integration evidence. Keep B2B stock-list functions distinct from retail listing management; a B2B daily sync is not a retail review schedule.
- **S3 — WholeCell bulk pricing, July 24, 2018**: https://www.wholecell.io/blog/8-updates-7-24-2018-bulk-pricing-erasing-purchase-orders-etc . Bulk list-price editing was announced before the sample. This is a tool for implementing choices, not a documented optimizer.
- **S4 — Swappa WholeCell guide, July 1, 2023**: https://swappa.com/blog/wholecell-integration-guide/ . Professional inventory workflow. Does not demonstrate use by Upgrade.
- **S5 — Sellercloud current Swappa manual**: https://help.sellercloud.com/sellercloud-knowledge-base/integrations/channels/swappa/swappa-account-integration~7630831775736888700 . SKU mapping and automatic transmission of Site Price; no equation determining that price. Start date and focal-seller adoption unverified. Platform corroboration: https://swappa.com/partners/sellercloud .
- **S6 — Swappa transaction benchmark, July 20, 2018**: https://swappa.com/blog/iphone-pricing-data/ . Historical public sale-price information. Current https://swappa.com/prices values are not 2023 observations or wholesale costs.
- **S7 — Swappa buying guide, pre-sample page**: https://swappa.com/blog/swappa-how-to-buy/ . Negotiation mechanism motivates checking whether comments precede price changes. It does not identify how many changes were negotiated in our sample.
- **S8 — Current visibility FAQ**: https://swappa.com/faq/answer/what-is-featured-listing . Listing visibility can differ. Historical feature use needs source validation, not mechanical backdating.
- **S9 — WholeCell purchase-order analytics, November 13, 2023**: https://help.wholecell.io/en/articles/7241964-purchase-order-analytics . Paid values, age, margins, and returns can be recorded. This supports the feasibility of obtaining cost and inventory data, not the availability of unchosen-price demand.
- **S10 — WholeCell power-seller sync, March 8, 2024**: https://help.wholecell.io/en/articles/4659957-syncing-with-swappa-for-power-sellers . Automatic delisting on commitment elsewhere is dated Q1 2024. Changelog corroboration: https://www.wholecell.io/updates (March 6, 2024). This feature cannot justify assuming complete cross-channel removal in 2023.
- **S11 — Swappa fee article, April 25, 2022**: https://swappa.com/blog/?p=4371 . Historical buyer/seller fee components. Net receipts still require processor, shipping, and transaction records. Do not interpret the old x=1.03a as verified seller payout.
- **S12 — Incoming inventory**: https://swappa.com/blog/how-to-use-swappa-incoming-inventory/ . Bulk initial pricing is a staging feature; it does not establish ongoing automatic repricing. Associated documentation surfaced at https://static.swappa.com/static/documents/Swappa_Incoming_Inventory_Documentation.pdf ; no historical version verified.
- **S13 — Current enterprise tools**: https://help.wholecell.io/en/articles/5643992-syncing-with-swappa-for-enterprise-sellers (displayed May 28, 2026); https://swappa.com/en/partners/linnworks ; https://swappa.com/about/sellers ; https://swappa.com/partners . Current inventory/order capabilities and seller categories are useful leads, not a map of the 2023 observation unit.
- **S14 — Other seller identities**: https://adisunwireless.com/ links AMS Traders on Swappa; https://swappa.com/bp/savegadgetcom-llc/ratings?page=118 and https://swappa.com/listing/view/LZFD85094 provide current or post-sample business context. None identifies a repricer. Do not confuse unrelated companies with similar names.

### Repricing mechanisms outside Swappa

- **V1 — Repricer.com May 26, 2023 announcement**: https://www.repricer.com/blog/product-update-may-2023/ . Strong dated evidence that actual-sales-based repricing was offered before the sample. Earlier February announcement: https://www.repricer.com/blog/product-update-february-2023/ . Exact current settings may have changed.
- **V2 — Sales rule manual (current)**: https://support.repricer.com/an-introduction-to-sales-rules . Documented threshold adjustments based on orders or revenues. The manual limits this feature to Amazon. No Swappa integration or adoption was verified.
- **V3 — Sales timing (current)**: https://support.repricer.com/sales-rules-settings . Scheduled checks and temporary locks are explicit configurable features. They are not learned update frequencies in our data.
- **V4 — StreetPricer matching (current)**: https://streetpricer.zendesk.com/hc/en-us/articles/360013070412-Radar-Scan-Options . eBay competitor selection filters. https://streetpricer.com/features/ and https://streetpricer.com/faq/ advertise additional capabilities and integrations. A Sellercloud integration does not establish an end-to-end Swappa pricing channel.
- **V5 — Limits (current)**: https://support.repricer.com/minimum-and-maximum-prices . The below-item-cost option prevents using generic floor functionality as evidence of no selling below acquisition cost. Recorded item cost itself need not equal current opportunity cost.
- **V6 — Amazon (current)**: https://sell.amazon.com/blog/automate-pricing-rules and https://sell.amazon.com/tools/automate-pricing . Concrete competitor and sales rule examples; not Swappa's allocation mechanism.
- **V7 — Informed (current)**: https://help.informedrepricer.com/en/articles/2399964-upload-templates . Exposed target-velocity and override fields suggest potentially useful records to request. They do not disclose a complete learning algorithm.
- **V8 — Seller Snap (current marketing)**: https://sellersnap.io/ and https://sellersnap.io/walmart-repricer/ . Adaptive/game-theoretic claims are not finite-sample guarantees. No adopted update equation found.
- **V9 — Competitor rules (current)**: https://support.repricer.com/creating-an-amazon-repricing-rule . Competitor selection, offsets, price limits, and fallback rules motivate a restricted policy family. This does not certify its profit performance in Swappa.
- **V10 — Walmart API (current)**: https://developer.walmart.com/global-marketplace/docs/repricing-during-item-setup . Strategy and minimum/maximum fields provide another operational example, without focal-seller evidence.
- **V11 — Repricer API (current)**: https://developers.repricer.com/docs/api-parameter-option-list . Pricing, costs, limits, and rule identifiers can be exported in some products. A useful list of records to seek; none obtained here.
- **L1 — Calzolari and Hanspach, April 9, 2024**: https://philiphanspach.github.io/files/repricers.pdf . Original descriptive industry study. Their supplier-feature classification is relevant background, not verification of proprietary implementation or Swappa adoption.

### Leads not admitted as historical evidence

Back Market integration PDFs at https://help.sellercloud.com/wp-content/uploads/2019/06/Sellercloud-Backmarket-Integration.pdf and https://help.sellercloud.com/wp-content/uploads/2023/02/SellerCloud-Merchants-Desk.pdf appeared in search but returned unavailable/404 on opening. The year in the URL is insufficient. The primary help link https://merchant-support.backmarket.com/hc/en-us/articles/4415101044370-How-to-use-the-BackPricer- could not be read; https://api.backmarket.dev/ returned a largely empty parsed page. These are unresolved leads. Current buyer-facing https://help.backmarket.com/hc/en-gb/articles/26225666693276-Why-do-prices-vary-on-Back-Market concerns a different platform's selection mechanism.

Modern sites surfaced at https://taskmarket.org/ , https://sync.techfleet.dev/features/ai-ready , https://tradedeck-site.vercel.app/ , and https://constacloud.com/commercium/swappa-and-odoo-erp-integration.html . Claims range from bespoke services to beta/future capabilities and integration plumbing. No independently verified 2023 deployment or focal-seller use was found. Do not cite these as evidence that sample sellers used a commercial repricer.

Search results also returned this project's own paper, general software-upgrade pages, vendor promotion, and forum anecdotes. The paper is circular evidence for our assumptions. Unverified anecdotes and unrelated names were excluded. No adoption rate is estimated from this search.

## Reproducible data audit

Script: `python/swappa_rescue/pricing_practices_audit.py`.
Run from code root with `SWAPPA_DATA_VERSION=fullsource_refscreen_known` using the research Python environment.
Tracked output: `docs/evidence/pricing_practices_audit_20261020.json`. The same result is in the existing known-status output folder. The JSON records input and script SHA256, base commit, sample restrictions, and interpretation.

| Seller | Consecutive pairs | Changes | Increases | Decreases | Change dates | Common-price category-days |
|---|---:|---:|---:|---:|---:|---:|
| Upgrade | 8,633 | 148 | 22 | 126 | 29 | 33.01% |
| AMS Traders | 20,004 | 373 | 7 | 366 | 27 | 34.58% |
| GadgetPickup | 18,349 | 2,035 | 111 | 1,924 | 83 | 92.45% |
| Certified Cell | 9,091 | 726 | 52 | 674 | 24 | 44.16% |
| SaveGadget | 9,603 | 1,007 | 40 | 967 | 24 | 70.12% |

The last column is among category-days with more than one owned listing. Categories are device generation, storage, and condition, as in the current demand specification. Upgrade has 833 such category-days. All five sellers have 165 observed dates in this retained panel. The audit selection is five named businesses, not a claim that these are the five largest sellers.

Upgrade's 148 changes include 107 on dates with at least five changes (72.30%). Its maximum is 19 changes on one date. Five changed prices are within one dollar of the previous day's rival minimum. These counts motivate batching and rule comparisons; they neither reveal review frequency nor reject all competitor rules. Unknown offsets, fees, omitted rival attributes, and intraday changes remain relevant.

The older 159 changes on 32 dates includes comparisons across gaps. Preserve that number with its denominator instead of treating the two counts as conflicting. Retained rows with unchanged prices do not establish no review. Gaps can be missing scrape observations or changing availability. A single displayed price can hide several intraday changes.

## Specification before cost inversion

1. **Observation audit:** establish asking versus displayed price conversion, seller fees, known terminal outcome versus actual order date, initial versus continuation observations, and historical multi-quantity/individual-device listing types. Locate comments and seller-reference fields if retained. Do not impute adoption from a seller reference code alone.
2. **Review model:** fit changes and no-change spells separately. Examine common change dates and offsets across products. Treat observed changes as a lower bound on price modifications only under no missing intermediate-reversion concerns; they are not a count of reviews. Without intraday data, keep a set of plausible schedules rather than declaring daily updating.
3. **Candidate policies:** persistence; stable selected-competitor offset with limits; lagged public transaction benchmark; sales-threshold changes; inventory-age markdowns. Predetermine a small parameter grid and economically meaningful matching categories. Use features observed before the choice; contemporaneous minimum prices do not establish temporal ordering.
4. **Validation:** choose parameters in an earlier period and test in a later period; retain a persistence benchmark, error at change dates, signed change accuracy, and calibration by seller/category. A high all-days accuracy can be meaningless when prices rarely change. Do not present failed holdout performance as empirical validation.
5. **Identification:** separately specify floor-to-cost margins, cost records, executable outside offers, or a proved learning restriction. Report how each assumption changes the feasible set. If multiple models remain, use their union unless joint restrictions are independently justified.
6. **Inference:** model-selection uncertainty and demand uncertainty remain. A chronological holdout does not itself establish causal demand or correct confidence coverage. Do not choose the best-looking cost interval after testing many models.

No policy fitting beyond the descriptive audit was performed in this research pass. The task was to close the old exercise and identify defensible foundations, not silently replace it with another unvalidated fitted policy.

## What evidence would change the conclusion

A seller's 2023 settings export or decision log could identify its actual policy and version. A manual proves capability, not settings. Logs should include active/inactive rules, manual overrides, competitor exclusions, minimum and maximum prices, rule assignments across products, and when prices were reviewed. For a learning interpretation, obtain the observed feedback and any exploration probabilities or update parameters.

For the economic target, seek device-level purchase orders, refurbishment/repair expense, shipping, payment fees, returns, cross-channel disposition, and contemporaneous firm buyback quotes. Accounting costs and the value of retaining an already purchased phone are distinct. A five-dollar cost span cannot be recovered from common posted prices alone.

If only public listing data remain available, a credible outcome may be estimated price policies and wide cost bounds. The project should report that limit rather than select an unsupported guarantee. No canonical manuscript change or referee resolution is asserted here.

## Validation and publication record

The descriptive code ran successfully on the existing panel. Counts were reconciled with the earlier Upgrade totals and checked for arithmetic consistency. Human-note static checks and the authorized update to the old PDF are recorded in the paper handover. Source claims were separated from proposed assumptions, current capabilities from historical availability, and advertising from adoption. New cost estimates were not computed. All changes are local; no push, external message, or seller contact was made.
