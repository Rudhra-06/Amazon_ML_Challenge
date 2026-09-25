import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from src.preprocessing import (
    normalize_country,
    normalize_business_name,
    normalize_business_address,
    preprocess_dataframe
)


def legacy_preprocess_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    addr_col: str = "business_address",
    country_col: str = "country",
    strip_legal: bool = False
) -> pd.DataFrame:
    """Original slow row-wise preprocess_dataframe implementation for equivalence validation."""
    df_out = df.copy()
    raw_names = df_out[name_col].fillna("").astype(str)
    raw_addrs = df_out[addr_col].fillna("").astype(str)
    raw_countries = df_out[country_col].fillna("").astype(str)
    
    df_out["norm_country"] = raw_countries.apply(normalize_country)
    df_out["norm_name"] = raw_names.apply(lambda s: normalize_business_name(s, strip_legal=strip_legal))
    df_out["norm_address"] = raw_addrs.apply(normalize_business_address)
    return df_out


def test_equivalence():
    print("=== Testing Semantic Equivalence ===")
    
    representative_data = pd.DataFrame({
        "business_name": [
            "Café & Bakery, Inc.",
            "Acme Solutions Private Limited",
            "München Trading Co. & Sons",
            "  Global  Logistics   LLC ",
            "",
            None,
            np.nan,
            "12345",
            "Café, Inc.",
            "Alpha & Beta Ltd"
        ],
        "business_address": [
            "100 Main Street, Suite 400",
            "200 Post Office Box 123",
            "50 Broadway Avenue",
            " 10   High   Road ",
            "",
            None,
            np.nan,
            "789 Highway 101",
            "P.O. Box 999",
            "55 Boulevard Way"
        ],
        "country": [
            "USA",
            "United States",
            "Germany",
            " DEU ",
            "",
            None,
            np.nan,
            "India",
            "  uk ",
            "FRA"
        ]
    })
    
    # Run legacy scalar implementation
    df_legacy_no_strip = legacy_preprocess_dataframe(representative_data, strip_legal=False)
    df_legacy_strip = legacy_preprocess_dataframe(representative_data, strip_legal=True)
    
    # Run optimized implementation
    df_opt_no_strip = preprocess_dataframe(representative_data, strip_legal=False)
    df_opt_strip = preprocess_dataframe(representative_data, strip_legal=True)
    
    # Verify exact equality for strip_legal=False
    pd.testing.assert_series_equal(df_legacy_no_strip["norm_name"], df_opt_no_strip["norm_name"])
    pd.testing.assert_series_equal(df_legacy_no_strip["norm_address"], df_opt_no_strip["norm_address"])
    pd.testing.assert_series_equal(df_legacy_no_strip["norm_country"], df_opt_no_strip["norm_country"])
    
    # Verify exact equality for strip_legal=True
    pd.testing.assert_series_equal(df_legacy_strip["norm_name"], df_opt_strip["norm_name"])
    pd.testing.assert_series_equal(df_legacy_strip["norm_address"], df_opt_strip["norm_address"])
    pd.testing.assert_series_equal(df_legacy_strip["norm_country"], df_opt_strip["norm_country"])
    
    print("Semantic equivalence test passed! Optimized preprocessing output is 100% identical to original.\n")


def test_benchmark():
    print("=== Testing Performance Benchmark (100,000 synthetic records) ===")
    n = 100_000
    
    synthetic_df = pd.DataFrame({
        "business_name": [f"Company {i % 1000} & Co., Inc." for i in range(n)],
        "business_address": [f"{i % 500} Main Street, Suite {i % 50}" for i in range(n)],
        "country": ["USA" if i % 3 == 0 else ("DEU" if i % 3 == 1 else "IND") for i in range(n)]
    })
    
    t0 = time.time()
    df_prep = preprocess_dataframe(synthetic_df, strip_legal=True)
    t_elapsed = time.time() - t0
    
    print(f"Time taken to preprocess {n:,} rows: {t_elapsed:.4f} seconds")
    print(f"Throughput: {n / t_elapsed:,.2f} rows/second")
    
    assert "norm_name" in df_prep.columns
    assert "norm_address" in df_prep.columns
    assert "norm_country" in df_prep.columns
    assert len(df_prep) == n
    
    print("Benchmark test passed successfully!\n")


if __name__ == "__main__":
    test_equivalence()
    test_benchmark()
