# GadgetPickup: displayed order and observation timing

Date: October 20, 2026. Audience: Lorenzo and future research agents. Status: verified source-data findings and a limited public-page validation. No causal demand estimate, finite regret allowance, or cost interval is supplied by this audit.

> **Follow-up completed:** `gadgetpickup-identification-status-2026-10-20.md` reports the corrected refit, pagination-gap association, and corrected sales-information calculation.

## Result and decision

Two source-data problems must be addressed before promoting the earlier demand or cost calculations. First, the November 19 spreadsheet repeats every non-date cell of the November 18 spreadsheet. Second, all 3,121 duplicated listing-days in the original tables straddle 50-row page boundaries. Removing duplicate rows does not establish that the remaining rows represent a simultaneous census of offers.

The original spreadsheets also contain displayed row numbers that the cleaning code discarded. Recovering them shows that being cheapest within a model-storage-condition category is quite different from appearing first in the unfiltered device table. They provide evidence about how the market is displayed, but do not by themselves supply an instrument or a randomly assigned exposure measure.

A separate current public field, Date Closed, is available for historical listings. Fourteen public-page checks corroborate ordinary terminal timing in eight cases, explain two large Updated-date discrepancies, and help diagnose the copied snapshot. One checked listing closes two calendar days after its last appearance. These checks improve our knowledge of the observation process; they do not certify every order date or every listing's exposure duration.

**Decision.** Preserve earlier numerical results as historical diagnostics. Future daily-demand and learning calculations must address the copied date explicitly. Do not interpret the recovered scraper order as the position seen by all buyers, and do not treat fluctuations in that order as a random treatment without a mechanism.

## Inputs, provenance, and returns

The source is the 166 existing workbooks under `data/Swappa_Daily_Data/`. Extraction reads only the first worksheet and records Code, device, date, displayed number, physical table position, and price. It does not run the historical scraper, use its saved authentication material, or contact sellers. The original cleaning program `stata/raw_data_clean_for_swappa.do` drops the initial row-number field. The notebook collects separate listing pages sequentially and assigns a date after its collection loop; it does not save a request timestamp for each listing or page.

The extraction design was committed at `3cc6236`; the initial extraction ran at `1714d4f`. The extended summary and pagination checks ran at `758951f`, reusing the extraction only after verifying every source-workbook hash. `python/swappa_rescue/gadget_display_order_audit.py --from-cache` reproduces that summary. The copied-date calendar is produced by `gadget_snapshot_calendar.py` at `2ae290f`.

Tracked results are `docs/evidence/gadgetpickup_display_order_results_20261020.json`, `gadgetpickup_snapshot_calendar_results_20261020.json`, and `gadgetpickup_closed_date_validation_results_20261020.json`. Detailed extracted tables, matched positions, transition checks, and the observation calendar are in `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/gadget_demand_20261020/`. Set `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. A separate manifest records hashes of scripts and outputs; source-workbook hashes are in the display-order result.

**Returns.** A recovered historical-order dataset; a verified copied-date flag and the associated missing interval; fourteen public closure-date observations; and restrictions on which demand or learning interpretations these observations support. There is no revised first-stage fit or new cost result in this pass.

## Displayed order is not the same as the model's competitor minimum

The extraction contains 469,021 rows. Deduplicating listing and date gives 465,900 rows, all of which match the prepared full-source panel. All displayed numbers are numeric and agree with physical position within the device's concatenated table. All extracted prices agree with the prepared prices. There are 5,265 device-date tables.

In 5,035 tables, there is one downward price jump in displayed order. The remaining tables are nondecreasing in price. This pattern is consistent with separate ordered blocks, such as featured and ordinary listings. It is not an observed featured-status indicator. Swappa's older [buying guide](https://swappa.com/blog/swappa-how-to-buy/) describes ascending price order with featured listings preceding ordinary listings; this wording was available in the search index, while direct retrieval failed. The [October 2026 guide](https://swappa.com/blog/a-faster-way-to-shop-swappa-listings/) describes a January 2026 redesign and grouping of similar listings. That later interface must not be attributed to the 2023 sample.

Across GadgetPickup's 21,322 raw listing-days, median first table position is 28 and 18.8% are within the first ten positions. Among 2,721 observations at or below the cheapest other seller in the same model-storage-condition category, median position is 20; only 7.7% appear first, and 32.2% are within the first ten. The source table is filtered to unlocked devices, but not to the model's storage-condition category. Buyers can apply additional filters that the data do not record.

A sale probability depending on the lowest comparable rival price remains an economically plausible specification. The recovered ordering shows why that variable should not be described as a verified measure of being first on the website. A buyer's actual search set, featured status, other product variants, and own competing listings can all matter.

There are 1,237 changes in whether a continuing GadgetPickup listing is at or below the same-category rival minimum: 907 involve only a change in the rival minimum, 213 only a change in its own price, and 117 both. Rival-price changes directly change the buyer's alternatives. They are demand shifters, not excluded instruments for own price without an additional argument. This count is descriptive and precedes removal of the copied date.

## Pagination limits the interpretation of recovered ranks

There are exactly 3,121 duplicated listing-date pairs, each appearing twice at the same price. Every pair crosses a boundary between successive blocks of fifty rows; 195 pairs belong to GadgetPickup. The first examples in the source have one occurrence near positions 47--50 and another near 51--54. The observed row distance can be larger when equal-price blocks are large.

The notebook follows successive page links. The duplicates therefore establish that its concatenated table is not a coherent enumeration with each listing appearing once. Unstable ordering of equal-price offers and changes while pages are loaded are possible explanations. The current evidence does not distinguish them or identify all omitted listings. Deduplication removes repeated exposure counts; it does not recover missing offers or their sale hazards.

Among same-device, same-price pairs of GadgetPickup listings observed on consecutive dates, 9,360 of 24,355 relative-order comparisons reverse. This makes a deterministic oldest-first ordering implausible as a universal description, but it does not establish random assignment. The comparisons can span different storage or condition variants and different listing priorities. More fundamentally, if tie order changes between page requests, the order in the researcher's request need not persist for buyers. Estimating a causal sale effect from that single order would require a link from the captured order to buyers' exposure.

The 50-row boundary result and the reversal count are exploratory mechanism checks. They are not evidence that the seller ran a pricing experiment. The next source-data assessment should concentrate on missingness near page boundaries and the reliability of competition measures, rather than immediately fitting a rank-based demand curve.

## A copied date creates a missing observation interval

The November 18 and November 19 workbooks each contain 3,457 rows and twenty columns. Every cell outside the date column is identical, including listing identifiers, row numbers, prices, seller text, and all other retained fields. The date column changes from November 18 to November 19. After within-date deduplication, each contains the same 3,435 listings in the same order. No other adjacent pair has identical listing identifiers, prices, and positions in the extracted sample. This finding establishes repeated data, not why the repetition occurred.

GadgetPickup has 197 raw observations on each date. Its old terminal-status construction assigns thirty sales to November 19 and none to November 18. In the retained demand sample, the affected dates each have 195 observations and the latter date has twenty-nine sale labels. These are not thirty known sales that should all be moved mechanically to November 18.

The observation calendar flags November 19 as a repeated snapshot. The next fresh observation after November 18 is November 20, leaving a two-day interval. Daily hazards cannot treat that interval as a standard one-day exposure. For a diagnostic restricted to starts of one-day observed intervals, removing November 18 and November 19 from the retained GadgetPickup sample leaves 20,096 rows, 1,565 existing terminal-sale labels, and 163 dates. This is a restriction of the earlier proxy sample, not a newly validated daily outcome model. The final sample date remains censored as before.

For a two-day interval, converting the probability of a sale to a daily hazard would require assumptions about the within-interval price, competing offers, and hazard. For example, 1-(1-q)^2 would impose a constant daily hazard and appropriate conditional survival dynamics. Those assumptions are not supplied by the copied worksheet. We have not filled in the missing date or recomputed the old cost results.

## Public closure dates

After finding the Date Closed field, we fixed a fourteen-listing check at `51255c0`: eight ordinary cases selected by a SHA256 ordering of identifiers, three negative Updated-minus-last gaps selected the same way, and the three largest positive gaps with that tie rule. The largest discrepancy, LXLM45265, had already been inspected when this sample was fixed. This is a diagnostic sample with deliberate oversampling of discrepancies, not an estimate of the population error frequency.

| Listing | Last observed | Saved Updated | Public Date Closed |
|---|---|---|---|
| [LXLV88245](https://swappa.com/listing/view/LXLV88245) | Nov 9 | Nov 10 | Nov 10 |
| [LXNJ08178](https://swappa.com/listing/view/LXNJ08178) | Dec 15 | Dec 15 | Dec 15 |
| [LXIT64625](https://swappa.com/listing/view/LXIT64625) | Sep 18 | Sep 18 | Sep 18 |
| [LXKL51386](https://swappa.com/listing/view/LXKL51386) | Oct 27 | Oct 27 | Oct 27 |
| [LXMI54035](https://swappa.com/listing/view/LXMI54035) | Nov 17 | Nov 18 | Nov 18 |
| [LXJP12215](https://swappa.com/listing/view/LXJP12215) | Oct 3 | Oct 3 | Oct 3 |
| [LXMI95959](https://swappa.com/listing/view/LXMI95959) | Nov 23 | Nov 24 | Nov 24 |
| [LXLY32383](https://swappa.com/listing/view/LXLY32383) | Nov 10 | Nov 10 | Nov 10 |
| [LXMG54591](https://swappa.com/listing/view/LXMG54591) | Nov 19 | Nov 18 | Nov 18 |
| [LXMH38200](https://swappa.com/listing/view/LXMH38200) | Nov 19 | Nov 18 | Nov 18 |
| [LXMF74325](https://swappa.com/listing/view/LXMF74325) | Nov 19 | Nov 18 | Nov 18 |
| [LXLM45265](https://swappa.com/listing/view/LXLM45265) | Nov 3 | Nov 27 | Nov 4 |
| [LXHD25322](https://swappa.com/listing/view/LXHD25322) | Oct 6 | Oct 9 | Oct 7 |
| [LXJV91606](https://swappa.com/listing/view/LXJV91606) | Sep 22 | Sep 24 | Sep 24 |

All dates in the table are in 2023. Each current page identifies GadgetPickup and marks the listing sold. The live Updated field agrees with the saved Updated value in these fourteen checks. Date Closed is a separate closure marker. We do not assume it is a paid-order timestamp, and current page contents are not independently archived 2023 snapshots.

All eight ordinary cases close on their last observed date or the next date. Two large discrepancies become ordinary once closure replaces update: LXLM45265 closes twenty-three days before its Updated date, and LXHD25322 closes two days before its Updated date. The three checked negative-gap cases close on November 18, supporting the inference that their November 19 appearances are repeated observations. All eleven negative-gap GadgetPickup cases in the raw sold-listing audit have November 19 as their last observed date, although only three were checked publicly. LXJV91606 closes two days after last appearance and remains an unresolved exposure-timing discrepancy.

## Consequences for demand and regret

The new evidence supports retaining a terminal-exposure approach as a candidate observation model rather than equating Updated with the sale date. It also establishes exceptions that must be handled. The daily predictor and sequential feedback calculations in the earlier note include the repeated date; their reported numbers remain reproducible historical results, but they are not a final corrected analysis.

A pricing algorithm's finite regret guarantee also cannot be improved by dropping an unobserved date. Suppose a valid cumulative guarantee on the full sequence were Gamma(c), and D_t(c) bounded the absolute payoff difference between any compared action and the chosen action on an omitted date. For retained dates S and omitted dates M,

    sum_{t in S} [u_t(a;c)-u_t(a_t;c)]
      <= Gamma(c) + sum_{t in M} D_t(c).

This follows by subtracting the omitted regret from full-sequence regret and bounding how negative omitted regret can be. The range allowance is justified only by valid payoff support restrictions; it is not a number chosen to obtain tight cost endpoints. This observation does not establish the missing full-sequence guarantee Gamma(c). Alternatively, unobserved actions and states can remain latent in the new method's consistency set. Neither approach permits counting the copied price vector as an extra verified learner action or extra observed sales feedback.

The available evidence therefore changes the next action. Correct the observation calendar, assess pagination-related missingness, and investigate interval-censored outcomes before fitting another structural response. A causal-demand argument and a supported finite performance restriction remain separate requirements. No amount of improving the source chronology by itself supplies the latter.
