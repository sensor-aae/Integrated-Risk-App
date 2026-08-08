from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


# --------- Column helpers ---------
_PD_ALIASES = ["pd", "probability_of_default", "probability of default", "p_default"]
_LGD_ALIASES = ["lgd", "loss_given_default", "loss given default"]
_EAD_ALIASES = ["ead", "exposure_at_default", "exposure at default"]
_SEG_ALIASES = ["segment", "bucket", "group", "portfolio", "business_unit", "bu", "rating", "grade", "class"]


def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in lower_map:
            return lower_map[a]
    return None


def _to_decimal_if_percent(s: pd.Series) -> pd.Series:
    """
    Convert only entries that look like percentages (>1 and <=100) to decimals.
    Leave already-decimal entries (0..1) unchanged.
    Also tolerates None/NaN safely.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    out = s_num.copy()
    mask = (out > 1.0) & (out <= 100.0)
    out.loc[mask] = out.loc[mask] / 100.0
    return out


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    out = pd.Series(np.zeros(len(n)), index=n.index, dtype=float)
    mask = d != 0
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


# --------- Public API ---------

def validate_and_standardize(
    df: pd.DataFrame,
    pd_col: str | None = None,
    lgd_col: str | None = None,
    ead_col: str | None = None,
) -> Tuple[pd.DataFrame, str, str, str, Optional[str]]:
    """
    Ensure PD/LGD/EAD columns exist and are numeric in [0,1] for PD/LGD and >=0 for EAD.
    Returns (df_std, PD_COL, LGD_COL, EAD_COL, SEG_COL_or_None).
    """
    # Find columns (case-insensitive, with aliases)
    PDc = pd_col or _find_col(df, _PD_ALIASES)
    LGDc = lgd_col or _find_col(df, _LGD_ALIASES)
    EADc = ead_col or _find_col(df, _EAD_ALIASES)
    if PDc is None or LGDc is None or EADc is None:
        raise ValueError("Missing PD/LGD/EAD columns. Expected columns like PD, LGD, EAD (case-insensitive).")

    SEGc = _find_col(df, _SEG_ALIASES)

    # Coerce numeric
    out = df.copy()
    for c in [PDc, LGDc, EADc]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Convert PD/LGD from % if needed; clamp to [0,1]
    pd_dec = _to_decimal_if_percent(out[PDc])
    lgd_dec = _to_decimal_if_percent(out[LGDc])

    # Count what we are about to clamp. Silently clipping out-of-range inputs
    # hides a data-quality finding: "3 facilities had PD > 1 and were capped" is
    # something a validator needs to see, not something to swallow.
    quality = {
        "pd_out_of_range": int(((pd_dec < 0) | (pd_dec > 1)).sum()),
        "lgd_out_of_range": int(((lgd_dec < 0) | (lgd_dec > 1)).sum()),
        "ead_negative": int((out[EADc] < 0).sum()),
        "pd_missing": int(pd_dec.isna().sum()),
        "lgd_missing": int(lgd_dec.isna().sum()),
        "ead_missing": int(out[EADc].isna().sum()),
    }

    out[PDc] = pd_dec.clip(0.0, 1.0)
    out[LGDc] = lgd_dec.clip(0.0, 1.0)
    out[EADc] = out[EADc].clip(lower=0.0)
    out.attrs["data_quality"] = quality

    return out, PDc, LGDc, EADc, SEGc


def apply_credit_shocks(
    df: pd.DataFrame,
    PDc: str, LGDc: str, EADc: str,
    pd_mult: float = 1.0,
    pd_add_bps: float = 0.0,   # additive in basis points (e.g., +25 = +0.0025)
    lgd_mult: float = 1.0,
    lgd_add_pct: float = 0.0,  # additive in percentage points (e.g., +5 = +0.05)
    ead_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Returns a new DataFrame with PD_final, LGD_final, EAD_final after shocks.
    All PD/LGD are clamped to [0,1], EAD_final >= 0.
    """
    out = df.copy()
    pd_add = pd_add_bps / 10_000.0
    lgd_add = lgd_add_pct / 100.0

    out["PD_final"] = (out[PDc] * float(pd_mult) + pd_add).clip(0.0, 1.0)
    out["LGD_final"] = (out[LGDc] * float(lgd_mult) + lgd_add).clip(0.0, 1.0)
    out["EAD_final"] = (out[EADc] * float(ead_mult)).clip(lower=0.0)
    return out


def compute_el_table(
    df: pd.DataFrame,
    pd_col: str | None = None,
    lgd_col: str | None = None,
    ead_col: str | None = None,
    pd_mult: float = 1.0,
    pd_add_bps: float = 0.0,
    lgd_mult: float = 1.0,
    lgd_add_pct: float = 0.0,
    ead_mult: float = 1.0,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Full pipeline: standardize -> apply shocks -> compute EL & EL%.
    Returns (df_result, SEG_COL_or_None).
    """
    std, PDc, LGDc, EADc, SEGc = validate_and_standardize(df, pd_col, lgd_col, ead_col)
    shocked = apply_credit_shocks(std, PDc, LGDc, EADc, pd_mult, pd_add_bps, lgd_mult, lgd_add_pct, ead_mult)
    shocked["EL"] = shocked["PD_final"] * shocked["LGD_final"] * shocked["EAD_final"]
    shocked["EL_pct_of_EAD"] = _safe_div(shocked["EL"], shocked["EAD_final"])
    shocked.attrs["data_quality"] = std.attrs.get("data_quality", {})
    return shocked, SEGc


def summarize_el(
    df_el: pd.DataFrame,
    seg_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Summaries: (grouped EL/EAD/EL%) and overall totals.
    """
    if seg_col and seg_col in df_el.columns:
        grp = df_el.groupby(seg_col, dropna=False).agg(
            total_EAD=("EAD_final", "sum"),
            total_EL=("EL", "sum"),
            avg_PD=("PD_final", "mean"),
            avg_LGD=("LGD_final", "mean"),
            count=("EAD_final", "size"),
        ).reset_index()
        grp["EL_pct_of_EAD"] = _safe_div(grp["total_EL"], grp["total_EAD"])
    else:
        grp = pd.DataFrame()

    total_ead = float(df_el["EAD_final"].sum())
    total_el = float(df_el["EL"].sum())

    # Exposure-weighted, matching the grouped rows above.
    #
    # This previously computed mean(per-facility EL/EAD) — an EQUAL-weighted
    # average of ratios, which let a $1,000 facility move the portfolio figure
    # as much as a $10m one, and disagreed with every grouped subtotal shown
    # beside it. Portfolio EL% is total EL over total EAD.
    totals = pd.Series({
        "total_EAD": total_ead,
        "total_EL": total_el,
        "EL_pct_of_EAD": (total_el / total_ead) if total_ead != 0 else 0.0,
        "facilities": int(len(df_el)),
    })
    return grp, totals
