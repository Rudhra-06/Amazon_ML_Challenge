import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from src.preprocessing import preprocess_dataframe
from src.matching import (
    build_candidate_features,
    score_candidates,
    select_matches,
    ScoringConfig
)


def test_matching_engine():
    print("=== Testing Matching Engine & Features ===")

    # 1. Create synthetic Source 1 DataFrame
    s1_raw = pd.DataFrame({
        "entity_id": ["S1_1", "S1_2", "S1_3", "S1_4"],
        "business_name": [
            "Acme Solutions Inc",        # Exact name match with S2_1
            "Global Logistics Corp",      # High name & address match with S2_2
            "Metro Bakery Ltd",          # Country mismatch test with S2_3
            "Apex Tech Systems"          # Weak match test with S2_4
        ],
        "business_address": [
            "100 Wall Street Suite 500",
            "200 Market Road",
            "50 Baker Street",
            "10 Innovation Way"
        ],
        "country": ["USA", "USA", "USA", "DEU"]
    })

    # 2. Create synthetic Target (S2/S3) DataFrame
    target_raw = pd.DataFrame({
        "entity_id": ["S2_1", "S2_2", "S2_3", "S2_4"],
        "business_name": [
            "Acme Solutions Incorporated", # Exact/High name match
            "Global Logistics Corp",       # Exact name & exact address match
            "Metro Bakery Ltd",           # Identical name/address, but country DEU
            "Apex Enterprise Solutions"    # Weak name match
        ],
        "business_address": [
            "100 Wall St Ste 500",         # High address similarity
            "200 Market Road",            # Exact address match
            "50 Baker Street",            # Exact address
            "999 Random Blvd"             # Different address
        ],
        "country": ["USA", "USA", "DEU", "DEU"] # Note: S2_3 is DEU (mismatch with S1_3 USA)
    })

    s1_prep = preprocess_dataframe(s1_raw, strip_legal=True)
    target_prep = preprocess_dataframe(target_raw, strip_legal=True)

    # Candidate pairs
    candidate_pairs = [
        ("S1_1", "S2_1", 2), # Multi-strategy match
        ("S1_2", "S2_2", 3), # High agreement match
        ("S1_3", "S2_3", 1), # Country mismatch
        ("S1_4", "S2_4", 1)  # Weak agreement match
    ]

    # Build features
    df_features = build_candidate_features(s1_prep, target_prep, candidate_pairs)
    assert not df_features.empty, "Feature extraction returned empty DataFrame!"
    assert "exact_name_match" in df_features.columns
    assert "country_match" in df_features.columns
    assert "name_similarity" in df_features.columns

    # Score candidates
    config = ScoringConfig()
    df_scored = score_candidates(df_features, config)
    assert "match_score" in df_scored.columns

    # Map scores by pair
    scores = {}
    for row in df_scored.itertuples():
        pair = (row.source1_entity_id, row.candidate_entity_id)
        scores[pair] = row.match_score

    print("Scores by pair:")
    for pair, score in scores.items():
        print(f"  {pair}: {score}")

    # Assertion 1: Exact / high name & address matches receive high scores (> 0.70)
    assert scores[("S1_2", "S2_2")] > 0.70, f"S1_2/S2_2 score too low: {scores[('S1_2', 'S2_2')]}"
    assert scores[("S1_1", "S2_1")] > 0.50, f"S1_1/S2_1 score too low: {scores[('S1_1', 'S2_1')]}"

    # Assertion 2: Country mismatch is penalized compared to country match
    s1_3_score = scores[("S1_3", "S2_3")]
    s1_2_score = scores[("S1_2", "S2_2")]
    assert s1_3_score < s1_2_score, f"Country mismatch not penalized! S1_3: {s1_3_score} vs S1_2: {s1_2_score}"

    # Assertion 3: Strong name/address agreement scores higher than weak agreement
    s1_4_score = scores[("S1_4", "S2_4")]
    assert s1_2_score > s1_4_score, f"Strong agreement did not score higher than weak agreement! {s1_2_score} <= {s1_4_score}"

    # Select matches
    matches = select_matches(df_scored, threshold=0.45, mode="top_1_per_s1")
    assert isinstance(matches, pd.DataFrame)
    assert set(matches.columns) == {"source1_entity_id", "candidate_entity_id", "match_score"}

    matched_pairs = set(zip(matches["source1_entity_id"], matches["candidate_entity_id"]))
    print(f"\nMatches selected (threshold=0.45): {matched_pairs}")
    
    assert ("S1_1", "S2_1") in matched_pairs
    assert ("S1_2", "S2_2") in matched_pairs
    assert ("S1_4", "S2_4") not in matched_pairs

    print("Matching engine basic tests passed successfully!\n")


def test_strategy_count_and_score_bounds():
    print("=== Testing Strategy Count & Score Normalization Bounds ===")
    
    s1_df = pd.DataFrame({
        "entity_id": ["S1_A", "S1_B"],
        "business_name": ["Perfect Match Corp", "Single Strategy Co"],
        "business_address": ["123 Main Street", "456 Oak Avenue"],
        "country": ["USA", "USA"]
    })
    
    target_df = pd.DataFrame({
        "entity_id": ["T_A", "T_B"],
        "business_name": ["Perfect Match Corp", "Single Strategy Co"],
        "business_address": ["123 Main Street", "456 Oak Avenue"],
        "country": ["USA", "USA"]
    })
    
    s1_prep = preprocess_dataframe(s1_df, strip_legal=True)
    target_prep = preprocess_dataframe(target_df, strip_legal=True)
    
    candidate_pairs = [
        ("S1_A", "T_A", 3),
        ("S1_B", "T_B", 1)
    ]
    
    df_features = build_candidate_features(s1_prep, target_prep, candidate_pairs)
    
    feat_a = df_features[df_features["source1_entity_id"] == "S1_A"].iloc[0]
    feat_b = df_features[df_features["source1_entity_id"] == "S1_B"].iloc[0]
    
    # Assert strategy_count and is_multi_strategy derivation
    assert feat_a["strategy_count"] == 3, f"Expected strategy_count=3, got {feat_a['strategy_count']}"
    assert feat_a["is_multi_strategy"] == 1.0, f"Expected is_multi_strategy=1.0, got {feat_a['is_multi_strategy']}"
    
    assert feat_b["strategy_count"] == 1, f"Expected strategy_count=1, got {feat_b['strategy_count']}"
    assert feat_b["is_multi_strategy"] == 0.0, f"Expected is_multi_strategy=0.0, got {feat_b['is_multi_strategy']}"
    
    # Score candidates
    config = ScoringConfig()
    df_scored = score_candidates(df_features, config)
    
    score_a = df_scored[df_scored["source1_entity_id"] == "S1_A"]["match_score"].iloc[0]
    score_b = df_scored[df_scored["source1_entity_id"] == "S1_B"]["match_score"].iloc[0]
    
    print(f"  Perfect Match (strategy_count=3) Score: {score_a}")
    print(f"  Single Strategy Match (strategy_count=1) Score: {score_b}")
    
    # Assert score bounds [0.0, 1.0]
    assert 0.0 <= score_a <= 1.0, f"Score A out of bounds [0, 1]: {score_a}"
    assert 0.0 <= score_b <= 1.0, f"Score B out of bounds [0, 1]: {score_b}"
    
    # Assert perfect agreement candidate does not exceed 1.0
    assert score_a == 1.0, f"Perfect agreement candidate expected 1.0, got {score_a}"
    
    print("Strategy count and score bounds test passed successfully!\n")


if __name__ == "__main__":
    test_matching_engine()
    test_strategy_count_and_score_bounds()
