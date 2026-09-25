import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from src.preprocessing import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    preprocess_dataframe,
    strip_legal_suffixes,
    extract_address_street_key
)
from src.blocking import (
    ExactNameCountryBlocking,
    ExactAddressCountryBlocking,
    NamePrefixCountryBlocking,
    AddressDerivedCountryBlocking,
    CombinedSignalBlocking,
    evaluate_blocking_strategy,
    evaluate_blocking_recall,
    calculate_reduction_ratio
)


def test_preprocessing():
    print("=== Testing Preprocessing ===")
    
    # 1. Accent and punctuation test
    raw_name = "Café & Bakery, Inc."
    norm_name = normalize_business_name(raw_name)
    assert norm_name == "cafe and bakery inc", f"Got: {norm_name}"
    
    # 2. Legal suffix stripping test
    stripped = normalize_business_name(raw_name, strip_legal=True)
    assert stripped == "cafe and bakery", f"Got: {stripped}"
    
    # 3. Address normalization test
    raw_addr = "123 Main Street, Suite 400"
    norm_addr = normalize_business_address(raw_addr)
    assert norm_addr == "123 main st ste 400", f"Got: {norm_addr}"
    
    # 4. Street key extraction
    street_key = extract_address_street_key(norm_addr)
    assert street_key == "123_main", f"Got: {street_key}"
    
    # 5. DataFrame preprocessing test
    df = pd.DataFrame({
        "entity_id": ["E1", "E2"],
        "business_name": ["Acme Corp.", "Báker Street Bakery LLC"],
        "business_address": ["456 Broadway Ave", "789 High Street"],
        "country": ["US", "UK"]
    })
    df_prep = preprocess_dataframe(df, strip_legal=True)
    assert "norm_name" in df_prep.columns
    assert "norm_address" in df_prep.columns
    assert "norm_country" in df_prep.columns
    print("Preprocessing tests passed successfully!\n")


def test_blocking_strategies():
    print("=== Testing Blocking Strategies & Metrics ===")
    
    # Mock Source 1 (Validation)
    s1 = pd.DataFrame({
        "entity_id": ["S1_1", "S1_2", "S1_3"],
        "business_name": ["Acme Solutions", "Global Trading Inc", "Apex Dynamics"],
        "business_address": ["100 Wall Street", "200 Market Rd", "300 Tech Park"],
        "country": ["USA", "USA", "DEU"]
    })
    
    # Mock Source 2
    s2 = pd.DataFrame({
        "entity_id": ["S2_1", "S2_2"],
        "business_name": ["Acme Solutions LLC", "Global Trading"],
        "business_address": ["100 Wall St", "200 Market Road"],
        "country": ["USA", "USA"]
    })
    
    # Mock Source 3
    s3 = pd.DataFrame({
        "entity_id": ["S3_1", "S3_2"],
        "business_name": ["Acme Sol", "Apex Dynamics GmbH"],
        "business_address": ["100 Wall St", "300 Tech Park"],
        "country": ["USA", "DEU"]
    })
    
    # Ground truth:
    # S1_1 matches S2_1 and S3_1
    # S1_2 matches S2_2
    # S1_3 matches S3_2
    gt = {
        "S1_1": {"S2_1", "S3_1"},
        "S1_2": {"S2_2"},
        "S1_3": {"S3_2"}
    }
    
    s1_prep = preprocess_dataframe(s1, strip_legal=True)
    s2_prep = preprocess_dataframe(s2, strip_legal=True)
    s3_prep = preprocess_dataframe(s3, strip_legal=True)
    
    strategies = [
        ExactNameCountryBlocking(),
        ExactAddressCountryBlocking(),
        NamePrefixCountryBlocking(prefix_length=4),
        AddressDerivedCountryBlocking(),
        CombinedSignalBlocking()
    ]
    
    for strat in strategies:
        report = evaluate_blocking_strategy(strat, s1_prep, [s2_prep, s3_prep], gt)
        print(f"Strategy: {report['strategy_name']}")
        print(f"  Num Val S1:          {report['num_validation_s1']}")
        print(f"  Fully Recovered:     {report['num_fully_recovered']}")
        print(f"  Blocking Recall:     {report['blocking_recall']}")
        print(f"  Avg Candidates:      {report['avg_candidates']}")
        print(f"  Median Candidates:   {report['median_candidates']}")
        print(f"  Max Candidates:      {report['max_candidates']}")
        print(f"  Reduction Ratio:     {report['reduction_ratio']}")
        print("-" * 50)

    print("Blocking tests passed successfully!\n")


if __name__ == "__main__":
    test_preprocessing()
    test_blocking_strategies()
