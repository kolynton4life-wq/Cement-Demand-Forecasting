import numpy as np
from preprocessing import run_preprocessing, validate_and_clean, engineer_features, SAFE_FEATURES, LEAKY_COLUMNS


def test_no_leaky_columns_in_safe_features():
    """The single most important regression test in this project: SAFE_FEATURES
    must never contain a column mathematically derived from same-day
    consumed_tonnes (see the Step 4 cross-review that caught this class of bug)."""
    for col in LEAKY_COLUMNS:
        assert col not in SAFE_FEATURES, f"Leaky column '{col}' found in SAFE_FEATURES!"


def test_balance_equation_holds_after_validation(sample_raw_df):
    df = validate_and_clean(sample_raw_df)
    expected_closing = df["opening_inventory_tonnes"] + df["deliveries_tonnes"] - df["consumed_tonnes"]
    diff = (df["closing_inventory_tonnes"] - expected_closing).abs()
    # allow for capacity-capping to legitimately break equality above capacity
    over_capacity = df["closing_inventory_tonnes"] >= df["silo_capacity"] - 1e-6
    assert (diff[~over_capacity] < 0.01).all(), "Balance equation violated on non-capped rows"


def test_no_negative_values_after_validation(sample_raw_df):
    df = validate_and_clean(sample_raw_df)
    for col in ["consumed_tonnes", "planned_pour_tonnes", "opening_inventory_tonnes",
                "deliveries_tonnes", "closing_inventory_tonnes", "rain_mm"]:
        assert (df[col] >= 0).all(), f"Negative values remain in {col}"


def test_no_duplicate_grain(sample_raw_df):
    df = validate_and_clean(sample_raw_df)
    dup = df.duplicated(subset=["date", "site_id", "cement_type"]).sum()
    assert dup == 0


def test_lag_1d_matches_literal_yesterday(sample_raw_df):
    """Regression test for the grain bug caught during Step 4 review:
    lag features must be grouped by site_id ONLY (continuous daily
    series), not by (site_id, cement_type) — which would fragment each
    site's series and make 'yesterday' mean something else entirely."""
    df = validate_and_clean(sample_raw_df)
    df = engineer_features(df)
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    for site_id in df["site_id"].unique():
        sub = df[df.site_id == site_id].sort_values("date").reset_index(drop=True)
        lag_vals = sub["consumed_lag_1d"].iloc[1:].values
        yesterday_vals = sub["consumed_tonnes"].iloc[:-1].values
        assert np.allclose(lag_vals, yesterday_vals), f"consumed_lag_1d broken for {site_id}"


def test_rolling_features_use_shift_before_rolling(sample_raw_df):
    """Regression test: rolling mean must NOT include the current day's
    own consumption (that would be direct target leakage)."""
    df = validate_and_clean(sample_raw_df)
    df = engineer_features(df)
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    site_id = df["site_id"].iloc[0]
    sub = df[df.site_id == site_id].sort_values("date").reset_index(drop=True)
    # roll_mean_7d at row i should be the mean of rows [i-7, i-1], NOT including row i
    row = 20
    expected = sub["consumed_tonnes"].iloc[row - 7: row].mean()
    actual = sub["consumed_roll_mean_7d"].iloc[row]
    assert abs(expected - actual) < 1e-6


def test_run_preprocessing_produces_all_safe_features(sample_raw_df):
    df = run_preprocessing(sample_raw_df)
    for col in SAFE_FEATURES:
        assert col in df.columns, f"Missing expected feature column: {col}"


def test_run_preprocessing_no_nulls_in_recent_rows(sample_raw_df):
    """After the first 28 days (longest lag window), features should be fully populated."""
    df = run_preprocessing(sample_raw_df)
    recent = df[df.groupby("site_id")["date"].rank(ascending=False) <= 30]
    for col in SAFE_FEATURES:
        if col in ("planned_pour_trust_adjusted", "adherence_roll_mean_14d"):
            continue  # can still be null on zero-planned days — expected, see notebook 03 caveat
        assert recent[col].notna().all(), f"Unexpected nulls in {col} on recent rows"
