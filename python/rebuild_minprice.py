"""
Rebuild SellerDistribution and sale_probability files with MIN-OF-OTHERS
competitor-price aggregation, instead of the existing AVG-OF-OTHERS.

Replaces the Stata pipeline (Stata license expired 2025-10-23 — this is the
Python equivalent of generate_matlab_data_minprice.do).

Inputs:
  data/intermediate/price_res_5bins_15_sellers.dta
  data/intermediate/price_res_5bins_all_sellers.dta

Outputs (to matlab/data/):
  sale_probability_5bins_res1_minprice.xlsx
  SellerDistribution_15_sellers_res1_minprice.xlsx
"""

from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np


REPO = Path(__file__).resolve().parents[1]
INTERMEDIATE = REPO / "data" / "intermediate"
OUT_DIR = REPO / "matlab" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_KEYS = ["device_id", "Storage", "Condition", "date"]
N_BINS = 5
N_SELLERS_TOP = 15


def recompute_min_price(df: pd.DataFrame) -> pd.DataFrame:
    """Replace competitor_price (currently avg-of-others) with min-of-others.

    Logic per (device, storage, condition, date) group:
      - the group min Price       = global min of the group's prices.
      - the group second-smallest = min after dropping one occurrence of the min.
      - for a row with rank == 1 (it IS the min, possibly tied), use second-smallest
        if present, else NaN (singleton group => drop later).
      - for rows with rank >  1, use the group min (someone else has it).

    Tied minima: rows tied at the min get the same min as their "competitor min"
    (i.e., their tied peer's price).  That's the natural interpretation.
    """
    df = df.copy()

    # Sort within group by Price; rank within group.
    df = df.sort_values(GROUP_KEYS + ["Price"], kind="stable").reset_index(drop=True)
    df["_rank_grp"] = df.groupby(GROUP_KEYS).cumcount() + 1
    df["_grp_size"] = df.groupby(GROUP_KEYS)["Price"].transform("size")
    df["_min_grp"] = df.groupby(GROUP_KEYS)["Price"].transform("min")

    # Second-smallest: minimum of prices strictly greater than the group min.
    # If there are no strictly-greater prices (all rows tied at min),
    # _second_grp will be NaN.  In that case, use the min itself for tied rows
    # (they're tied with each other at the min).
    def _second_smallest(s: pd.Series) -> float:
        sorted_unique = np.sort(s.unique())
        if len(sorted_unique) >= 2:
            return float(sorted_unique[1])
        return float(sorted_unique[0])

    df["_second_grp"] = df.groupby(GROUP_KEYS)["Price"].transform(_second_smallest)

    # competitor_price logic:
    #   rank 1 (this row is the unique min): comp = _second_grp
    #   rank > 1 (someone else has the min): comp = _min_grp
    # Edge: if the seller's price tied with the group min and there is another
    # row tied at the min (still rank > 1 here because of stable sort), use min.
    is_unique_min = (df["_rank_grp"] == 1) & (df["Price"] < df["_second_grp"])
    df["competitor_price_min"] = np.where(is_unique_min, df["_second_grp"], df["_min_grp"])

    # Drop singletons: no competitors exist.
    df = df[df["_grp_size"] >= 2].copy()
    df.drop(columns=["_rank_grp", "_grp_size", "_min_grp", "_second_grp"], inplace=True)

    # Replace the existing competitor_price with the min-based one.
    df["competitor_price"] = df["competitor_price_min"]
    df.drop(columns=["competitor_price_min"], inplace=True)

    # Recompute the residual.
    df["comp_net_price_1"] = df["competitor_price"] - df["Ref_Price"]

    # Re-bin into 5 quantile bins (xtile equivalent).
    # pandas qcut returns 1..5 with `labels=False, duplicates='drop'`.
    df["comp_net_price_bins_1"] = (
        pd.qcut(df["comp_net_price_1"], q=N_BINS, labels=False, duplicates="drop") + 1
    ).astype(int)
    return df


def build_sale_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sale probability per (self_bin, comp_bin)."""
    # NB: existing pipeline subtracts 1 to make bins 0-indexed before export.
    sp = (
        df.groupby(["self_net_price_bins_1", "comp_net_price_bins_1"])["Sold_Today"]
        .mean()
        .reset_index(name="Sale_Prob")
    )
    sp["self_net_price_bins_1"] -= 1
    sp["comp_net_price_bins_1"] -= 1
    return sp[["self_net_price_bins_1", "comp_net_price_bins_1", "Sale_Prob"]]


def build_seller_distribution(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per-seller time-averaged 25-cell action histogram.

    Returns a dict keyed by seller number (1..15) of DataFrames with columns
    TimeAverage_00 .. TimeAverage_44, one row per (seller, observation) ordered
    by date / min_date / device_id / Code.  The LAST row is the final
    time-averaged distribution m_N for that seller.
    """
    # Build 25 indicator columns i_{self-1}{comp-1}.
    df = df.copy()
    for s in range(1, N_BINS + 1):
        for c in range(1, N_BINS + 1):
            col = f"i_{s-1}{c-1}"
            df[col] = (
                (df["self_net_price_bins_1"] == s) & (df["comp_net_price_bins_1"] == c)
            ).astype(float)

    # Order each seller's rows by (date, min_date, device_id, Code).
    df = df.sort_values(
        ["Seller", "date", "min_date", "device_id", "Code"], kind="stable"
    ).reset_index(drop=True)

    # Cumulative time-average within seller for each indicator.
    out: dict[int, pd.DataFrame] = {}
    for j in range(1, N_SELLERS_TOP + 1):
        sub = df[df["Seller_number"] == j].copy()
        if len(sub) == 0:
            continue
        n = len(sub)
        ts_cols = []
        for s in range(N_BINS):
            for c in range(N_BINS):
                col = f"i_{s}{c}"
                tcol = f"TimeAverage_{s}{c}"
                # cumulative mean
                sub[tcol] = sub[col].cumsum() / np.arange(1, n + 1)
                ts_cols.append(tcol)
        out[j] = sub[ts_cols].reset_index(drop=True)
    return out


def write_sale_prob_xlsx(out_path: Path, sp_top15: pd.DataFrame, sp_all: pd.DataFrame):
    """Write sale-probability xlsx with two sheets: 15sellers, Allsellers."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sp_top15.to_excel(writer, sheet_name="15sellers", index=False)
        sp_all.to_excel(writer, sheet_name="Allsellers", index=False)


def build_action_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and median of self_net_price_1 within each self_bin.

    Mirrors the original Stata code's Actions_Mean / Actions_Median sheets
    (used by df.setup.game_application via get_player_data_5acts to map
    bin index -> own-action price).
    """
    grouped = df.groupby("self_net_price_bins_1")["self_net_price_1"]
    mean_df = grouped.mean().to_frame(name="mean_deviation").T.reset_index(drop=True)
    median_df = grouped.median().to_frame(name="median_deviation").T.reset_index(drop=True)

    # Rename columns "mean_deviation_bin_<k>" / "median_deviation_bin_<k>"
    bin_cols = sorted(grouped.groups.keys())
    mean_df.columns = [f"mean_deviation_bin_{int(k)}" for k in bin_cols]
    median_df.columns = [f"median_deviation_bin_{int(k)}" for k in bin_cols]

    return mean_df, median_df


def write_seller_dist_xlsx(
    out_path: Path,
    dist_per_seller: dict[int, pd.DataFrame],
    mean_actions: pd.DataFrame,
    median_actions: pd.DataFrame,
):
    """Write seller-distribution xlsx with one sheet per seller, plus
    Actions_Mean and Actions_Median (consumed by MATLAB application setup)."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for j, df in dist_per_seller.items():
            df.to_excel(writer, sheet_name=f"Seller_{j}", index=False)
        mean_actions.to_excel(writer, sheet_name="Actions_Mean", index=False)
        median_actions.to_excel(writer, sheet_name="Actions_Median", index=False)


def main():
    print("\n=== Min-price rebuild (Python) ===")

    # Top 15
    print("\nTop-15 dta -> sale prob + seller distribution")
    df_top15, _ = pyreadstat.read_dta(str(INTERMEDIATE / "price_res_5bins_15_sellers.dta"))
    print(f"  loaded: {df_top15.shape}")
    df_top15 = recompute_min_price(df_top15)
    print(f"  after min-price recomp + drop singletons: {df_top15.shape}")
    sp_top15 = build_sale_prob(df_top15)
    print(f"  sale_prob shape: {sp_top15.shape} (expected up to 25 rows)")
    dist_top15 = build_seller_distribution(df_top15)
    print(f"  seller dist sheets: {list(dist_top15.keys())}")
    mean_actions, median_actions = build_action_summaries(df_top15)
    print(f"  Actions_Mean cols: {list(mean_actions.columns)}")
    print(f"  Actions_Median cols: {list(median_actions.columns)}")

    # All sellers
    print("\nAll-sellers dta -> sale prob")
    df_all, _ = pyreadstat.read_dta(str(INTERMEDIATE / "price_res_5bins_all_sellers.dta"))
    print(f"  loaded: {df_all.shape}")
    df_all = recompute_min_price(df_all)
    print(f"  after min-price recomp + drop singletons: {df_all.shape}")
    sp_all = build_sale_prob(df_all)
    print(f"  sale_prob shape: {sp_all.shape}")

    # Outputs
    sale_prob_path = OUT_DIR / "sale_probability_5bins_res1_minprice.xlsx"
    seller_dist_path = OUT_DIR / "SellerDistribution_15_sellers_res1_minprice.xlsx"

    print(f"\nWriting {sale_prob_path}")
    write_sale_prob_xlsx(sale_prob_path, sp_top15, sp_all)

    print(f"Writing {seller_dist_path}")
    write_seller_dist_xlsx(seller_dist_path, dist_top15, mean_actions, median_actions)

    print("\n=== Min-price rebuild done ===")


if __name__ == "__main__":
    main()
