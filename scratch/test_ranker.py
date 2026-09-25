import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from src.preprocessing import preprocess_dataframe
from src.ranker import (
    RankerConfig,
    extract_vectorized_features,
    score_and_rank_candidates,
    select_matches_with_policy,
    evaluate_validation_predictions,
    stream_full_test_inference
)


def test_ranker_unit_and_metrics():
    print("=== Testing Ranker Unit Functions & Validation Metrics ===")

    # 1. Mock Source 1
    s1_raw = pd.DataFrame({
        "entity_id": ["S1_1", "S1_2", "S1_3", "S1_4"],
        "business_name": [
            "Acme Solutions Inc",        # Exact match
            "Global Logistics Corp",      # High match with margin
            "Metro Bakery Ltd",          # Country mismatch test
            "Generic Store"              # Ambiguous / generic match
        ],
        "business_address": [
            "100 Wall Street Suite 500",
            "200 Market Road",
            "50 Baker Street",
            "1 Main St"
        ],
        "country": ["USA", "USA", "USA", "USA"]
    })

    # 2. Mock Target (S2/S3)
    target_raw = pd.DataFrame({
        "entity_id": ["T_1A", "T_2A", "T_2B", "T_3A", "T_4A", "T_4B"],
        "business_name": [
            "Acme Solutions Inc",        # T_1A: Exact
            "Global Logistics Corp",       # T_2A: Exact Rank 1
            "Global Logistics LLC",        # T_2B: Rank 2 (close tie)
            "Metro Bakery Ltd",           # T_3A: Country mismatch (DEU)
            "Generic Store 1",             # T_4A: Ambiguous candidate 1
            "Generic Store 2"              # T_4B: Ambiguous candidate 2
        ],
        "business_address": [
            "100 Wall Street Suite 500",
            "200 Market Road",
            "200 Market Rd Ste 2",
            "50 Baker Street",
            "1 Main St",
            "1 Main Street"
        ],
        "country": ["USA", "USA", "USA", "DEU", "USA", "USA"]
    })

    s1_prep = preprocess_dataframe(s1_raw, strip_legal=True)
    tgt_prep = preprocess_dataframe(target_raw, strip_legal=True)

    # Candidate pairs DataFrame
    candidate_pairs = pd.DataFrame([
        {"source1_entity_id": "S1_1", "candidate_entity_id": "T_1A", "strategy_count": 3},
        {"source1_entity_id": "S1_2", "candidate_entity_id": "T_2A", "strategy_count": 2},
        {"source1_entity_id": "S1_2", "candidate_entity_id": "T_2B", "strategy_count": 1},
        {"source1_entity_id": "S1_3", "candidate_entity_id": "T_3A", "strategy_count": 1},
        {"source1_entity_id": "S1_4", "candidate_entity_id": "T_4A", "strategy_count": 1},
        {"source1_entity_id": "S1_4", "candidate_entity_id": "T_4B", "strategy_count": 1},
    ])

    # Extract vectorized features
    df_feats = extract_vectorized_features(s1_prep, tgt_prep, candidate_pairs)
    assert not df_feats.empty
    assert "exact_name_match" in df_feats.columns
    assert "name_addr_interact" in df_feats.columns

    # Score and rank
    config = RankerConfig(min_score_threshold=0.50, min_margin_threshold=0.04, selection_policy="margin_gated_top1")
    df_scored = score_and_rank_candidates(df_feats, config)
    assert "rank" in df_scored.columns
    assert "score_margin" in df_scored.columns

    print("\nScored Candidates Table:")
    for row in df_scored.itertuples():
        print(f"  S1: {row.source1_entity_id} | Tgt: {row.candidate_entity_id} | Score: {row.match_score:.4f} | Rank: {row.rank} | Margin: {row.score_margin:.4f}")

    # Select matches
    df_selected = select_matches_with_policy(df_scored, config)
    print(f"\nSelected Matches:\n{df_selected}")

    # Ground truth
    ground_truth = {
        "S1_1": {"T_1A"},
        "S1_2": {"T_2A"},
        "S1_3": {"T_3A"}
    }

    # Evaluate metrics
    val_report = evaluate_validation_predictions(df_selected, ground_truth, total_val_s1_count=4)
    print("\nValidation Report:")
    for k, v in val_report.items():
        print(f"  {k}: {v}")

    assert val_report["macro_f0_5"] > 0.0
    assert val_report["true_positives"] >= 2
    assert "no_match_false_positives" in val_report
    print("Unit & metrics tests passed successfully!\n")


def test_ranker_scale_and_streaming():
    print("=== Testing Ranker Synthetic Benchmark & Streaming (50,000 candidate pairs) ===")
    n_pairs = 50_000

    s1_mock = pd.DataFrame({
        "entity_id": [f"S1_{i % 5000}" for i in range(5000)],
        "business_name": [f"Company {i % 5000} LLC" for i in range(5000)],
        "business_address": [f"{i % 2000} Main St" for i in range(5000)],
        "country": ["USA" if i % 2 == 0 else "DEU" for i in range(5000)]
    })

    tgt_mock = pd.DataFrame({
        "entity_id": [f"T_{i}" for i in range(10000)],
        "business_name": [f"Company {i % 5000} Inc" for i in range(10000)],
        "business_address": [f"{i % 2000} Main Street" for i in range(10000)],
        "country": ["USA" if i % 2 == 0 else "DEU" for i in range(10000)]
    })

    cand_pairs = pd.DataFrame({
        "source1_entity_id": [f"S1_{i % 5000}" for i in range(n_pairs)],
        "candidate_entity_id": [f"T_{i % 10000}" for i in range(n_pairs)],
        "strategy_count": [2 if i % 3 == 0 else 1 for i in range(n_pairs)]
    })

    s1_prep = preprocess_dataframe(s1_mock, strip_legal=True)
    tgt_prep = preprocess_dataframe(tgt_mock, strip_legal=True)

    t0 = time.time()
    df_preds = stream_full_test_inference(s1_prep, [tgt_prep], cand_pairs, chunk_size=10_000)
    t_elapsed = time.time() - t0

    print(f"Time taken to score & rank {n_pairs:,} candidate pairs: {t_elapsed:.4f} seconds")
    print(f"Throughput: {n_pairs / t_elapsed:,.2f} pairs/second")
    print(f"Predicted Matches Count: {len(df_preds):,}")

    assert not df_preds.empty
    print("Scale & streaming benchmark passed successfully!\n")


if __name__ == "__main__":
    test_ranker_unit_and_metrics()
    test_ranker_scale_and_streaming()
