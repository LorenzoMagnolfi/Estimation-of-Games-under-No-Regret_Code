"""
Build rich-state data files for Path C1.A.

State expansion: replace s = comp_bin (5 levels) with
    s_rich = (comp_bin, nc_bin)   where nc_bin = quartile bin of num_competitor

Outputs:
  matlab/data/sale_probability_5bins_res1_rich.xlsx
      Sheet '15sellers' has columns:
          self_net_price_bins_1, comp_net_price_bins_1, nc_bin, Sale_Prob
      Sheet 'Allsellers' has the same columns over the all-sellers .dta.
  matlab/data/SellerDistribution_15_sellers_res1_rich.xlsx
      Per-seller sheets with cumulative time-averaged 5x20 indicators
      i_{self_bin}_{rich_state} for richer state, columns
      TimeAverage_<self>_<rich>.

Notes on bin construction:
  * nc_bin uses GLOBAL quartiles of num_competitor (not per-seller). This
    keeps the state-space partition consistent across sellers and across
    the demand-primitive estimation (which uses Allsellers data).
  * The bins are: [2-7], [8-15], [16-25], [26+] (covers Seller 1 well;
    Seller 2 sits more in the lower bins, which is fine — rich state has
    fewer effective cells for Seller 2).
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
N_BINS_SELF = 5
N_BINS_COMP = 5
NC_BIN_EDGES = [0, 7, 15, 25, np.inf]   # nc_bin in {1, 2, 3, 4}
NC_BIN_LABELS = [1, 2, 3, 4]
N_NC_BINS = len(NC_BIN_LABELS)
N_RICH_STATES = N_BINS_COMP * N_NC_BINS  # 5 * 4 = 20
N_SELLERS_TOP = 15


def add_nc_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nc_bin"] = pd.cut(
        df["num_competitor"], bins=NC_BIN_EDGES, labels=NC_BIN_LABELS
    ).astype(int)
    return df


def build_sale_prob_rich(df: pd.DataFrame) -> pd.DataFrame:
    """Sale probability conditional on (self_bin, comp_bin, nc_bin).

    Self/comp bins are 1-indexed in the dta; we subtract 1 for export to
    match the existing pipeline convention.
    """
    sp = (
        df.groupby(["self_net_price_bins_1", "comp_net_price_bins_1", "nc_bin"])[
            "Sold_Today"
        ]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Sale_Prob", "count": "n_obs"})
    )
    sp["self_net_price_bins_1"] -= 1
    sp["comp_net_price_bins_1"] -= 1
    sp["nc_bin"] -= 1
    return sp[
        ["self_net_price_bins_1", "comp_net_price_bins_1", "nc_bin", "n_obs", "Sale_Prob"]
    ]


def build_seller_distribution_rich(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per-seller cumulative time-averaged 5 x 20 indicator histograms.

    Indicator i_{self}_{rich} is 1 for the (self_bin, rich_state) cell.
    Rich state index runs 0..19 from (comp_bin, nc_bin) flattened in
    row-major order: rich = (comp_bin - 1) * N_NC_BINS + (nc_bin - 1).
    """
    df = df.copy()
    df["rich_state"] = (df["comp_net_price_bins_1"] - 1) * N_NC_BINS + (df["nc_bin"] - 1)

    # 5 x 20 indicators
    for s in range(1, N_BINS_SELF + 1):
        for r in range(N_RICH_STATES):
            col = f"i_{s-1}_{r}"
            df[col] = ((df["self_net_price_bins_1"] == s) & (df["rich_state"] == r)).astype(float)

    df = df.sort_values(
        ["Seller", "date", "min_date", "device_id", "Code"], kind="stable"
    ).reset_index(drop=True)

    out: dict[int, pd.DataFrame] = {}
    for j in range(1, N_SELLERS_TOP + 1):
        sub = df[df["Seller_number"] == j].copy()
        if len(sub) == 0:
            continue
        n = len(sub)
        ts_cols = []
        for s in range(N_BINS_SELF):
            for r in range(N_RICH_STATES):
                col = f"i_{s}_{r}"
                tcol = f"TimeAverage_{s}_{r}"
                sub[tcol] = sub[col].cumsum() / np.arange(1, n + 1)
                ts_cols.append(tcol)
        out[j] = sub[ts_cols].reset_index(drop=True)
    return out


def build_action_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby("self_net_price_bins_1")["self_net_price_1"]
    mean_df = grouped.mean().to_frame(name="mean_deviation").T.reset_index(drop=True)
    median_df = grouped.median().to_frame(name="median_deviation").T.reset_index(drop=True)

    bin_cols = sorted(grouped.groups.keys())
    mean_df.columns = [f"mean_deviation_bin_{int(k)}" for k in bin_cols]
    median_df.columns = [f"median_deviation_bin_{int(k)}" for k in bin_cols]
    return mean_df, median_df


def write_sale_prob_xlsx(out_path: Path, sp_top15: pd.DataFrame, sp_all: pd.DataFrame):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sp_top15.to_excel(writer, sheet_name="15sellers", index=False)
        sp_all.to_excel(writer, sheet_name="Allsellers", index=False)


def write_seller_dist_xlsx(
    out_path: Path,
    dist_per_seller: dict[int, pd.DataFrame],
    mean_actions: pd.DataFrame,
    median_actions: pd.DataFrame,
    nc_bin_meta: pd.DataFrame,
):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for j, df in dist_per_seller.items():
            df.to_excel(writer, sheet_name=f"Seller_{j}", index=False)
        mean_actions.to_excel(writer, sheet_name="Actions_Mean", index=False)
        median_actions.to_excel(writer, sheet_name="Actions_Median", index=False)
        nc_bin_meta.to_excel(writer, sheet_name="nc_bin_meta", index=False)


def main():
    print("\n=== Rich-state rebuild (Python, Path C1.A) ===")
    print(f"  nc_bin edges: {NC_BIN_EDGES}, labels: {NC_BIN_LABELS}")
    print(f"  N_RICH_STATES = {N_RICH_STATES} (= {N_BINS_COMP} comp_bins x {N_NC_BINS} nc_bins)")

    # Top 15
    print("\nTop-15 dta -> rich sale prob + rich seller distribution")
    df_top15, _ = pyreadstat.read_dta(str(INTERMEDIATE / "price_res_5bins_15_sellers.dta"))
    print(f"  loaded: {df_top15.shape}")
    df_top15 = add_nc_bin(df_top15)
    print(f"  nc_bin distribution: {df_top15['nc_bin'].value_counts().sort_index().to_dict()}")
    sp_top15 = build_sale_prob_rich(df_top15)
    print(f"  sale_prob_rich shape: {sp_top15.shape}  (max possible: {N_BINS_SELF * N_BINS_COMP * N_NC_BINS} = 100)")
    print(f"  min n_obs/cell = {sp_top15['n_obs'].min()}, median = {int(sp_top15['n_obs'].median())}")

    dist_top15 = build_seller_distribution_rich(df_top15)
    print(f"  rich seller dist sheets: {list(dist_top15.keys())}")
    mean_actions, median_actions = build_action_summaries(df_top15)

    # All sellers
    print("\nAll-sellers dta -> rich sale prob")
    df_all, _ = pyreadstat.read_dta(str(INTERMEDIATE / "price_res_5bins_all_sellers.dta"))
    print(f"  loaded: {df_all.shape}")
    df_all = add_nc_bin(df_all)
    sp_all = build_sale_prob_rich(df_all)
    print(f"  sale_prob_rich shape: {sp_all.shape}")
    print(f"  min n_obs/cell = {sp_all['n_obs'].min()}, median = {int(sp_all['n_obs'].median())}")

    # nc_bin metadata for the MATLAB side
    nc_bin_meta = pd.DataFrame({
        "nc_bin": NC_BIN_LABELS,
        "lower": [NC_BIN_EDGES[i] for i in range(N_NC_BINS)],
        "upper": [NC_BIN_EDGES[i + 1] for i in range(N_NC_BINS)],
    })

    sale_prob_path = OUT_DIR / "sale_probability_5bins_res1_rich.xlsx"
    seller_dist_path = OUT_DIR / "SellerDistribution_15_sellers_res1_rich.xlsx"

    print(f"\nWriting {sale_prob_path}")
    write_sale_prob_xlsx(sale_prob_path, sp_top15, sp_all)
    print(f"Writing {seller_dist_path}")
    write_seller_dist_xlsx(
        seller_dist_path, dist_top15, mean_actions, median_actions, nc_bin_meta
    )

    # Per-seller cell counts for top-2 sellers (sanity check)
    print("\n=== Per-seller rich-state cell counts (top-2) ===")
    for j in [1, 2]:
        sub = df_top15[df_top15["Seller_number"] == float(j)]
        cells = sub.groupby(["comp_net_price_bins_1", "nc_bin"]).size()
        print(f"  Seller {j}: T = {len(sub)}, n_cells_with_obs = {len(cells)} / {N_RICH_STATES}, "
              f"min N_cell = {cells.min()}, median = {int(cells.median())}, max = {cells.max()}")
        # Final m_N(a | s_rich) at the last row of the cumulative timeseries
        m_N = cells / cells.sum()
        print(f"    m_comp_rich (sum should be ~1): {m_N.sum():.3f}")

    print("\n=== Rich-state rebuild done ===")


if __name__ == "__main__":
    main()
