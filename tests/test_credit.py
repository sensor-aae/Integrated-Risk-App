"""Tests for the credit Expected Loss pipeline."""

import numpy as np
import pandas as pd
import pytest

from risklib.credit import compute_el_table, summarize_el, validate_and_standardize


@pytest.fixture
def book():
    return pd.DataFrame({
        "Segment": ["Retail", "Retail", "Corporate", "Corporate"],
        "PD": [0.02, 0.05, 0.01, 0.03],
        "LGD": [0.40, 0.50, 0.35, 0.45],
        "EAD": [100_000, 150_000, 200_000, 250_000],
    })


def test_el_is_pd_times_lgd_times_ead(book):
    out, _ = compute_el_table(book)
    expected = book["PD"] * book["LGD"] * book["EAD"]
    np.testing.assert_allclose(out["EL"].values, expected.values, rtol=1e-12)


def test_percent_inputs_are_detected_and_converted():
    """Accepts PD/LGD as either decimals or percentages."""
    df = pd.DataFrame({"PD": [0.02, 2.0], "LGD": [0.4, 40.0], "EAD": [1000, 2000]})
    out, _ = compute_el_table(df)
    assert out["EL"].iloc[0] == pytest.approx(8.0)
    assert out["EL"].iloc[1] == pytest.approx(16.0)


def test_pd_of_one_is_not_treated_as_percent():
    """PD = 1.0 means certain default, not 1%."""
    df = pd.DataFrame({"PD": [1.0], "LGD": [0.5], "EAD": [1000]})
    out, _ = compute_el_table(df)
    assert out["PD_final"].iloc[0] == pytest.approx(1.0)


def test_column_aliases_are_resolved():
    df = pd.DataFrame({
        "probability_of_default": [0.02],
        "loss_given_default": [0.4],
        "exposure_at_default": [1000],
        "rating": ["BBB"],
    })
    out, seg = compute_el_table(df)
    assert seg == "rating"
    assert out["EL"].iloc[0] == pytest.approx(8.0)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Missing PD/LGD/EAD"):
        compute_el_table(pd.DataFrame({"PD": [0.02], "LGD": [0.4]}))


def test_facility_el_sums_to_portfolio_el(book):
    """Aggregation consistency: the parts must sum to the whole."""
    out, seg = compute_el_table(book)
    grp, totals = summarize_el(out, seg)
    assert totals["total_EL"] == pytest.approx(out["EL"].sum(), rel=1e-12)
    assert grp["total_EL"].sum() == pytest.approx(totals["total_EL"], rel=1e-12)
    assert grp["total_EAD"].sum() == pytest.approx(totals["total_EAD"], rel=1e-12)


def test_portfolio_el_pct_is_exposure_weighted(book):
    """
    REGRESSION. Portfolio EL% was mean(per-facility EL/EAD) — an equal-weighted
    average of ratios that disagreed with every grouped subtotal beside it and
    let a tiny facility move the portfolio figure as much as a huge one.
    """
    out, seg = compute_el_table(book)
    _, totals = summarize_el(out, seg)
    assert totals["EL_pct_of_EAD"] == pytest.approx(
        totals["total_EL"] / totals["total_EAD"], rel=1e-12
    )
    # And it must differ from the old equal-weighted calculation on this book.
    equal_weighted = (out["EL"] / out["EAD_final"]).mean()
    assert totals["EL_pct_of_EAD"] != pytest.approx(equal_weighted, rel=1e-6)


def test_grouped_and_portfolio_el_pct_use_the_same_definition(book):
    out, seg = compute_el_table(book)
    grp, totals = summarize_el(out, seg)
    for _, row in grp.iterrows():
        assert row["EL_pct_of_EAD"] == pytest.approx(row["total_EL"] / row["total_EAD"], rel=1e-12)
    assert totals["EL_pct_of_EAD"] == pytest.approx(totals["total_EL"] / totals["total_EAD"], rel=1e-12)


def test_pd_multiplier_scales_expected_loss(book):
    base, _ = compute_el_table(book)
    shocked, _ = compute_el_table(book, pd_mult=2.0)
    np.testing.assert_allclose(shocked["EL"].values, 2.0 * base["EL"].values, rtol=1e-12)


def test_additive_pd_shock_uses_basis_points(book):
    shocked, _ = compute_el_table(book, pd_add_bps=100.0)   # +1.00%
    np.testing.assert_allclose(shocked["PD_final"].values,
                               book["PD"].values + 0.01, rtol=1e-12)


def test_additive_lgd_shock_uses_percentage_points(book):
    shocked, _ = compute_el_table(book, lgd_add_pct=10.0)   # +0.10
    np.testing.assert_allclose(shocked["LGD_final"].values,
                               book["LGD"].values + 0.10, rtol=1e-12)


def test_shocks_are_clamped_to_valid_ranges(book):
    shocked, _ = compute_el_table(book, pd_mult=100.0, lgd_mult=100.0)
    assert (shocked["PD_final"] <= 1.0).all()
    assert (shocked["LGD_final"] <= 1.0).all()
    assert (shocked["EAD_final"] >= 0).all()


def test_zero_ead_does_not_produce_infinity():
    df = pd.DataFrame({"PD": [0.02, 0.03], "LGD": [0.4, 0.5], "EAD": [0, 1000]})
    out, _ = compute_el_table(df)
    assert np.isfinite(out["EL_pct_of_EAD"]).all()
    assert out["EL_pct_of_EAD"].iloc[0] == 0.0


def test_out_of_range_inputs_are_reported_not_just_clamped():
    """Silently clipping bad inputs hides a data-quality finding."""
    df = pd.DataFrame({"PD": [150.0], "LGD": [0.4], "EAD": [-500]})
    std, *_ = validate_and_standardize(df)
    q = std.attrs["data_quality"]
    assert q["pd_out_of_range"] == 1
    assert q["ead_negative"] == 1


def test_unlabelled_segments_form_their_own_bucket():
    df = pd.DataFrame({
        "Segment": ["Retail", None],
        "PD": [0.02, 0.03], "LGD": [0.4, 0.5], "EAD": [1000, 2000],
    })
    out, seg = compute_el_table(df)
    grp, totals = summarize_el(out, seg)
    assert len(grp) == 2
    assert grp["total_EL"].sum() == pytest.approx(totals["total_EL"], rel=1e-12)
