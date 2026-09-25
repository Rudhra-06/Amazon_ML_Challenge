"""
Advanced Candidate Ranking, Feature Interaction, and Disambiguation Engine for Business Entity Resolution.

Optimized for high precision, macro F0.5 optimization, and streaming memory-conscious execution over millions of candidate pairs.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Fast Vectorized Similarity Helpers
# -----------------------------------------------------------------------------

def vectorized_char_ngram_jaccard(
    s1_list: List[str],
    s2_list: List[str],
    n: int = 3
) -> np.ndarray:
    """
    Vectorized computation of character n-gram Jaccard similarity.
    """
    sims = np.zeros(len(s1_list), dtype=np.float32)
    for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
        if not s1 or not s2:
            sims[i] = 0.0
            continue
        if s1 == s2:
            sims[i] = 1.0
            continue
            
        s1_pad = f" {s1} "
        s2_pad = f" {s2} "
        
        if len(s1_pad) < n or len(s2_pad) < n:
            sims[i] = 1.0 if s1 == s2 else 0.0
            continue
            
        g1 = set(s1_pad[j:j+n] for j in range(len(s1_pad) - n + 1))
        g2 = set(s2_pad[j:j+n] for j in range(len(s2_pad) - n + 1))
        
        inter = len(g1.intersection(g2))
        union = len(g1.union(g2))
        sims[i] = float(inter / union) if union > 0 else 0.0
        
    return sims


def vectorized_token_jaccard_and_overlap(
    s1_list: List[str],
    s2_list: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of word token Jaccard similarity and token Overlap coefficient.
    """
    jaccards = np.zeros(len(s1_list), dtype=np.float32)
    overlaps = np.zeros(len(s1_list), dtype=np.float32)
    
    for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
        if not s1 or not s2:
            continue
            
        t1 = set(s1.split())
        t2 = set(s2.split())
        if not t1 or not t2:
            continue
            
        inter = len(t1.intersection(t2))
        union = len(t1.union(t2))
        min_len = min(len(t1), len(t2))
        
        jaccards[i] = float(inter / union) if union > 0 else 0.0
        overlaps[i] = float(inter / min_len) if min_len > 0 else 0.0
        
    return jaccards, overlaps


# -----------------------------------------------------------------------------
# Configurable Policy & Scoring Architecture
# -----------------------------------------------------------------------------

@dataclass
class RankerConfig:
    """Configurable scoring weights, interaction coefficients, and decision thresholds."""
    w_exact_name: float = 0.35
    w_exact_addr: float = 0.25
    w_name_sim: float = 0.20
    w_addr_sim: float = 0.15
    w_name_tok_sim: float = 0.10
    w_addr_tok_sim: float = 0.05
    w_multi_strategy: float = 0.10
    
    # Interaction weights
    w_name_addr_interact: float = 0.15
    w_exact_name_addr_interact: float = 0.10
    
    # Penalties
    country_mismatch_penalty: float = 0.50
    len_diff_penalty_factor: float = 0.005
    
    # Decision Policy Parameters
    min_score_threshold: float = 0.50
    min_margin_threshold: float = 0.05  # Minimum margin over rank-2 candidate for ambiguous matches
    selection_policy: str = "margin_gated_top1"  # Options: 'margin_gated_top1', 'high_precision_rank1_rank2', 'simple_top1'


# -----------------------------------------------------------------------------
# Vectorized Batch Feature Extraction
# -----------------------------------------------------------------------------

def extract_vectorized_features(
    s1_df: pd.DataFrame,
    target_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    candidate_pairs_df: pd.DataFrame,
    batch_size: int = 250_000
) -> pd.DataFrame:
    """
    Vectorized extraction of candidate pair features and interaction signals.
    """
    if candidate_pairs_df.empty:
        return pd.DataFrame()

    if isinstance(target_dfs, list):
        target_combined = pd.concat(target_dfs, ignore_index=True)
    else:
        target_combined = target_dfs

    # Ensure normalized column names
    s1_n = "norm_name" if "norm_name" in s1_df.columns else "business_name"
    s1_a = "norm_address" if "norm_address" in s1_df.columns else "business_address"
    s1_c = "norm_country" if "norm_country" in s1_df.columns else "country"

    tgt_n = "norm_name" if "norm_name" in target_combined.columns else "business_name"
    tgt_a = "norm_address" if "norm_address" in target_combined.columns else "business_address"
    tgt_c = "norm_country" if "norm_country" in target_combined.columns else "country"

    # Fast hash lookups
    s1_lookup = {
        str(row[0]): (str(row[1]) if pd.notna(row[1]) else "",
                      str(row[2]) if pd.notna(row[2]) else "",
                      str(row[3]) if pd.notna(row[3]) else "")
        for row in s1_df[["entity_id", s1_n, s1_a, s1_c]].itertuples(index=False)
    }

    tgt_lookup = {
        str(row[0]): (str(row[1]) if pd.notna(row[1]) else "",
                       str(row[2]) if pd.notna(row[2]) else "",
                       str(row[3]) if pd.notna(row[3]) else "")
        for row in target_combined[["entity_id", tgt_n, tgt_a, tgt_c]].itertuples(index=False)
    }

    n_pairs = len(candidate_pairs_df)
    feature_batches = []

    s1_col = "source1_entity_id" if "source1_entity_id" in candidate_pairs_df.columns else "s1_id"
    tgt_col = "candidate_entity_id" if "candidate_entity_id" in candidate_pairs_df.columns else "target_id"
    strat_col = "strategy_count" if "strategy_count" in candidate_pairs_df.columns else None

    for start in range(0, n_pairs, batch_size):
        batch_df = candidate_pairs_df.iloc[start:start+batch_size]
        
        s1_ids = batch_df[s1_col].astype(str).to_numpy()
        tgt_ids = batch_df[tgt_col].astype(str).to_numpy()
        strat_cnts = batch_df[strat_col].to_numpy() if strat_col else np.ones(len(batch_df), dtype=np.int32)

        n1_list, a1_list, c1_list = [], [], []
        n2_list, a2_list, c2_list = [], [], []

        for s1_id, tgt_id in zip(s1_ids, tgt_ids):
            info1 = s1_lookup.get(s1_id, ("", "", ""))
            info2 = tgt_lookup.get(tgt_id, ("", "", ""))
            n1_list.append(info1[0])
            a1_list.append(info1[1])
            c1_list.append(info1[2])
            n2_list.append(info2[0])
            a2_list.append(info2[1])
            c2_list.append(info2[2])

        # Vectorized string similarities
        name_sim = vectorized_char_ngram_jaccard(n1_list, n2_list, n=3)
        addr_sim = vectorized_char_ngram_jaccard(a1_list, a2_list, n=3)
        name_tok_sim, name_tok_over = vectorized_token_jaccard_and_overlap(n1_list, n2_list)
        addr_tok_sim, addr_tok_over = vectorized_token_jaccard_and_overlap(a1_list, a2_list)

        # Vectorized boolean signals
        exact_name = np.array([(n1 != "" and n1 == n2) for n1, n2 in zip(n1_list, n2_list)], dtype=np.float32)
        exact_addr = np.array([(a1 != "" and a1 == a2) for a1, a2 in zip(a1_list, a2_list)], dtype=np.float32)
        
        country_match = np.array([
            1.0 if (c1 == c2 and c1 != "unknown") else (0.5 if (c1 == "unknown" or c2 == "unknown") else 0.0)
            for c1, c2 in zip(c1_list, c2_list)
        ], dtype=np.float32)

        name_len_diff = np.array([abs(len(n1) - len(n2)) for n1, n2 in zip(n1_list, n2_list)], dtype=np.float32)
        addr_len_diff = np.array([abs(len(a1) - len(a2)) for a1, a2 in zip(a1_list, a2_list)], dtype=np.float32)
        
        is_multi_strat = np.where(strat_cnts >= 2, 1.0, 0.0).astype(np.float32)

        # Interaction signals
        name_addr_interact = name_sim * addr_sim
        exact_name_addr_interact = exact_name * addr_sim
        country_name_interact = country_match * name_sim

        b_df = pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_id": tgt_ids,
            "strategy_count": strat_cnts,
            "is_multi_strategy": is_multi_strat,
            "exact_name_match": exact_name,
            "exact_address_match": exact_addr,
            "country_match": country_match,
            "name_similarity": name_sim,
            "address_similarity": addr_sim,
            "name_token_similarity": name_tok_sim,
            "address_token_similarity": addr_tok_sim,
            "name_token_overlap": name_tok_over,
            "address_token_overlap": addr_tok_over,
            "name_length_diff": name_len_diff,
            "address_length_diff": addr_len_diff,
            "name_addr_interact": name_addr_interact,
            "exact_name_addr_interact": exact_name_addr_interact,
            "country_name_interact": country_name_interact
        })
        feature_batches.append(b_df)

    return pd.concat(feature_batches, ignore_index=True)


# -----------------------------------------------------------------------------
# Disambiguation & Candidate Group Scoring
# -----------------------------------------------------------------------------

def score_and_rank_candidates(
    df_features: pd.DataFrame,
    config: Optional[RankerConfig] = None
) -> pd.DataFrame:
    """
    Score candidate pairs and calculate per-S1 candidate rank, score margin, and confidence.
    """
    if config is None:
        config = RankerConfig()
        
    df_out = df_features.copy()
    if df_out.empty:
        df_out["match_score"] = 0.0
        df_out["score_margin"] = 0.0
        df_out["rank"] = 1
        return df_out

    total_pos_weight = (
        config.w_exact_name + config.w_exact_addr + config.w_name_sim +
        config.w_addr_sim + config.w_name_tok_sim + config.w_addr_tok_sim +
        config.w_multi_strategy + config.w_name_addr_interact + config.w_exact_name_addr_interact
    )

    raw_score = (
        config.w_exact_name * df_out["exact_name_match"] +
        config.w_exact_addr * df_out["exact_address_match"] +
        config.w_name_sim * df_out["name_similarity"] +
        config.w_addr_sim * df_out["address_similarity"] +
        config.w_name_tok_sim * df_out["name_token_similarity"] +
        config.w_addr_tok_sim * df_out["address_token_similarity"] +
        config.w_multi_strategy * df_out["is_multi_strategy"] +
        config.w_name_addr_interact * df_out["name_addr_interact"] +
        config.w_exact_name_addr_interact * df_out["exact_name_addr_interact"]
    )

    weighted_signal = raw_score / total_pos_weight if total_pos_weight > 0 else raw_score
    len_penalty = config.len_diff_penalty_factor * (df_out["name_length_diff"] + df_out["address_length_diff"])
    score_after_penalty = np.maximum(0.0, weighted_signal - len_penalty)
    country_multiplier = np.where(df_out["country_match"] == 0.0, config.country_mismatch_penalty, 1.0)
    
    df_out["match_score"] = np.round(np.clip(score_after_penalty * country_multiplier, 0.0, 1.0), 4)

    # Calculate per-S1 rank and score margin over 2nd best candidate
    df_out = df_out.sort_values(by=["source1_entity_id", "match_score"], ascending=[True, False])
    df_out["rank"] = df_out.groupby("source1_entity_id").cumcount() + 1
    
    # Calculate top1 and top2 score per S1 group
    group_scores = df_out.groupby("source1_entity_id")["match_score"].apply(list).to_dict()
    
    margins = []
    cand_counts = []
    for s1_id, score_list in zip(df_out["source1_entity_id"], df_out["match_score"]):
        scores_for_s1 = group_scores.get(s1_id, [score_list])
        cand_counts.append(len(scores_for_s1))
        if len(scores_for_s1) >= 2:
            margin = scores_for_s1[0] - scores_for_s1[1]
        else:
            margin = scores_for_s1[0]  # Sole candidate margin is its own score
        margins.append(round(margin, 4))
        
    df_out["score_margin"] = margins
    df_out["candidate_count"] = cand_counts
    
    return df_out.reset_index(drop=True)


def select_matches_with_policy(
    df_scored: pd.DataFrame,
    config: Optional[RankerConfig] = None
) -> pd.DataFrame:
    """
    Select final predicted matches applying configurable decision policies.
    """
    if config is None:
        config = RankerConfig()
        
    if df_scored.empty or "match_score" not in df_scored.columns:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_score"])

    cutoff = config.min_score_threshold
    margin_cutoff = config.min_margin_threshold

    if config.selection_policy == "margin_gated_top1":
        # Select rank-1 candidate if score >= threshold AND (score_margin >= margin_cutoff OR score >= 0.85)
        rank1 = df_scored[df_scored["rank"] == 1].copy()
        selected = rank1[
            (rank1["match_score"] >= cutoff) & 
            ((rank1["score_margin"] >= margin_cutoff) | (rank1["match_score"] >= 0.85))
        ]
    elif config.selection_policy == "high_precision_rank1_rank2":
        # Select rank-1 if score >= cutoff; select rank-2 if score >= cutoff + 0.15 AND is_multi_strategy==1 AND country_match==1
        rank1 = df_scored[(df_scored["rank"] == 1) & (df_scored["match_score"] >= cutoff)]
        rank2 = df_scored[
            (df_scored["rank"] == 2) & 
            (df_scored["match_score"] >= cutoff + 0.15) & 
            (df_scored["is_multi_strategy"] == 1.0) & 
            (df_scored["country_match"] == 1.0)
        ]
        selected = pd.concat([rank1, rank2], ignore_index=True)
    else:  # 'simple_top1'
        rank1 = df_scored[df_scored["rank"] == 1].copy()
        selected = rank1[rank1["match_score"] >= cutoff]

    return selected[["source1_entity_id", "candidate_entity_id", "match_score"]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Validation Metric Evaluator (Macro F0.5)
# -----------------------------------------------------------------------------

def evaluate_validation_predictions(
    df_predictions: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    total_val_s1_count: int
) -> Dict[str, Any]:
    """
    Calculate Macro F0.5, Precision, Recall, TP, FP, FN, predicted match count, and no-match FPs.
    
    Macro F0.5 Formula:
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    if df_predictions.empty:
        pred_dict = {}
    else:
        pred_dict = df_predictions.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    tp = 0
    fp = 0
    fn = 0
    no_match_fp = 0

    all_eval_s1 = set(ground_truth.keys()).union(set(pred_dict.keys()))
    
    for s1_id in all_eval_s1:
        true_set = ground_truth.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        if not true_set:
            if pred_set:
                fp += len(pred_set)
                no_match_fp += len(pred_set)
            continue

        true_pos = len(pred_set.intersection(true_set))
        false_pos = len(pred_set - true_set)
        false_neg = len(true_set - pred_set)

        tp += true_pos
        fp += false_pos
        fn += false_neg

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (0.25 * precision + recall) > 0:
        f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    else:
        f0_5 = 0.0

    return {
        "macro_f0_5": round(f0_5, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_match_count": len(df_predictions),
        "no_match_false_positives": no_match_fp,
        "total_eval_s1_entities": total_val_s1_count
    }


# -----------------------------------------------------------------------------
# Streaming Inference Runner for Full Test Set Execution
# -----------------------------------------------------------------------------

def stream_full_test_inference(
    s1_df: pd.DataFrame,
    target_dfs: List[pd.DataFrame],
    candidate_pairs_df: pd.DataFrame,
    config: Optional[RankerConfig] = None,
    chunk_size: int = 500_000
) -> pd.DataFrame:
    """
    Memory-efficient streaming runner for full test set inference.
    Processes candidate pairs in chunks of 500K to generate predictions without RAM exhaustion.
    """
    if config is None:
        config = RankerConfig()

    n_total = len(candidate_pairs_df)
    results = []

    for start in range(0, n_total, chunk_size):
        chunk = candidate_pairs_df.iloc[start:start+chunk_size]
        features = extract_vectorized_features(s1_df, target_dfs, chunk, batch_size=250_000)
        scored = score_and_rank_candidates(features, config)
        selected = select_matches_with_policy(scored, config)
        results.append(selected)

    if not results:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_score"])

    final_df = pd.concat(results, ignore_index=True)
    # Deduplicate in case a candidate pair crossed chunk boundaries
    final_df = final_df.sort_values(by=["source1_entity_id", "match_score"], ascending=[True, False])
    final_df = final_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"], keep="first")
    return final_df.reset_index(drop=True)
