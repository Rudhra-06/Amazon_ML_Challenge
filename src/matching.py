"""
Matching module for Business Entity Resolution.

Provides memory-efficient candidate-pair feature extraction, deterministic scoring,
and threshold-based match selection for entity resolution candidate pairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Fast Lightweight String Similarity Helpers
# -----------------------------------------------------------------------------

def jaccard_token_similarity(s1: str, s2: str) -> float:
    """Calculate word token Jaccard similarity between two normalized strings."""
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return float(intersection / union) if union > 0 else 0.0


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """
    Calculate character n-gram Jaccard similarity between two strings.
    Robust to minor typos, spelling errors, and suffix variations.
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
        
    s1_pad = f" {s1} "
    s2_pad = f" {s2} "
    
    if len(s1_pad) < n or len(s2_pad) < n:
        return 1.0 if s1 == s2 else 0.0
        
    grams1 = set(s1_pad[i:i+n] for i in range(len(s1_pad) - n + 1))
    grams2 = set(s2_pad[i:i+n] for i in range(len(s2_pad) - n + 1))
    
    intersection = len(grams1.intersection(grams2))
    union = len(grams1.union(grams2))
    return float(intersection / union) if union > 0 else 0.0


def calculate_pair_features_single(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
    strategy_count: int = 1
) -> Dict[str, Any]:
    """
    Compute fine-grained matching features for a single (S1, Target) entity pair.
    """
    exact_name = 1.0 if (name1 and name1 == name2) else 0.0
    exact_addr = 1.0 if (addr1 and addr1 == addr2) else 0.0
    country_match = 1.0 if (country1 == country2 and country1 != "unknown") else (0.5 if (country1 == "unknown" or country2 == "unknown") else 0.0)
    
    name_sim = char_ngram_jaccard(name1, name2, n=3)
    addr_sim = char_ngram_jaccard(addr1, addr2, n=3)
    
    name_tok_sim = jaccard_token_similarity(name1, name2)
    addr_tok_sim = jaccard_token_similarity(addr1, addr2)
    
    name_len_diff = abs(len(name1) - len(name2))
    addr_len_diff = abs(len(addr1) - len(addr2))
    
    is_multi_strategy = 1.0 if strategy_count >= 2 else 0.0

    return {
        "exact_name_match": exact_name,
        "exact_address_match": exact_addr,
        "country_match": country_match,
        "name_similarity": round(name_sim, 4),
        "address_similarity": round(addr_sim, 4),
        "name_token_similarity": round(name_tok_sim, 4),
        "address_token_similarity": round(addr_tok_sim, 4),
        "name_length_diff": name_len_diff,
        "address_length_diff": addr_len_diff,
        "strategy_count": strategy_count,
        "is_multi_strategy": is_multi_strategy
    }


# -----------------------------------------------------------------------------
# Configuration Dataclass for Scoring Engine
# -----------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Configuration weights and penalties for deterministic candidate scoring."""
    w_exact_name: float = 0.35
    w_exact_addr: float = 0.25
    w_name_sim: float = 0.20
    w_addr_sim: float = 0.15
    w_name_tok_sim: float = 0.10
    w_addr_tok_sim: float = 0.05
    w_multi_strategy: float = 0.10
    
    # Penalties
    country_mismatch_penalty: float = 0.50  # Multiplied if countries mismatch
    len_diff_penalty_factor: float = 0.005  # Per character difference penalty
    min_score_threshold: float = 0.50


# -----------------------------------------------------------------------------
# Feature Extraction Engine
# -----------------------------------------------------------------------------

def convert_candidate_input(
    candidate_input: Union[Dict[str, Set[str]], List[Tuple[str, str]], Dict[Tuple[str, str], int], pd.DataFrame]
) -> pd.DataFrame:
    """
    Standardize candidate inputs into a pair DataFrame with columns:
    ['source1_entity_id', 'candidate_entity_id', 'strategy_count']
    """
    if isinstance(candidate_input, pd.DataFrame):
        df_pairs = candidate_input.copy()
        col_map = {}
        for c in df_pairs.columns:
            if c in ("s1_id", "source1_id", "s1"):
                col_map[c] = "source1_entity_id"
            elif c in ("target_id", "candidate_id", "target", "s2_s3_id"):
                col_map[c] = "candidate_entity_id"
        df_pairs.rename(columns=col_map, inplace=True)
        if "strategy_count" not in df_pairs.columns:
            df_pairs["strategy_count"] = 1
        return df_pairs[["source1_entity_id", "candidate_entity_id", "strategy_count"]]
        
    pairs = []
    if isinstance(candidate_input, dict):
        first_key = next(iter(candidate_input)) if candidate_input else None
        if isinstance(first_key, tuple):
            # Dict[(s1_id, cand_id), count]
            for (s1_id, cand_id), count in candidate_input.items():
                pairs.append((str(s1_id), str(cand_id), int(count)))
        else:
            # Dict[s1_id, Set[cand_id]]
            for s1_id, cand_set in candidate_input.items():
                for cand_id in cand_set:
                    pairs.append((str(s1_id), str(cand_id), 1))
    elif isinstance(candidate_input, list):
        for item in candidate_input:
            if len(item) == 2:
                pairs.append((str(item[0]), str(item[1]), 1))
            elif len(item) == 3:
                pairs.append((str(item[0]), str(item[1]), int(item[2])))
                
    return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id", "strategy_count"])


def build_candidate_features(
    s1_df: pd.DataFrame,
    target_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    candidate_input: Union[Dict[str, Set[str]], List[Tuple[str, str]], Dict[Tuple[str, str], int], pd.DataFrame],
    batch_size: int = 100_000
) -> pd.DataFrame:
    """
    Memory-efficient feature extraction for candidate S1-Target pairs.
    
    Args:
        s1_df: Source 1 preprocessed DataFrame.
        target_dfs: Source 2 / Source 3 preprocessed DataFrames.
        candidate_input: Candidate pairs generated by blocking layer.
        batch_size: Processing batch size to control peak RAM usage.
        
    Returns:
        pd.DataFrame containing pair entity IDs and extracted feature columns.
    """
    df_pairs = convert_candidate_input(candidate_input)
    if df_pairs.empty:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "strategy_count",
            "exact_name_match", "exact_address_match", "country_match",
            "name_similarity", "address_similarity", "name_token_similarity",
            "address_token_similarity", "name_length_diff", "address_length_diff",
            "is_multi_strategy"
        ])

    # Combine target DataFrames if list
    if isinstance(target_dfs, list):
        target_combined = pd.concat(target_dfs, ignore_index=True)
    else:
        target_combined = target_dfs

    # Ensure normalized columns exist
    def get_cols(df: pd.DataFrame):
        name_c = "norm_name" if "norm_name" in df.columns else "business_name"
        addr_c = "norm_address" if "norm_address" in df.columns else "business_address"
        country_c = "norm_country" if "norm_country" in df.columns else "country"
        return name_c, addr_c, country_c

    s1_n, s1_a, s1_c = get_cols(s1_df)
    tgt_n, tgt_a, tgt_c = get_cols(target_combined)

    # Build fast lookup dicts mapping entity_id -> (name, address, country)
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

    feature_rows = []
    n_pairs = len(df_pairs)

    for start in range(0, n_pairs, batch_size):
        batch = df_pairs.iloc[start:start+batch_size]
        for row in batch.itertuples(index=False):
            s1_id, cand_id, strat_cnt = row[0], row[1], row[2]
            
            s1_info = s1_lookup.get(s1_id, ("", "", ""))
            tgt_info = tgt_lookup.get(cand_id, ("", "", ""))
            
            feat = calculate_pair_features_single(
                name1=s1_info[0], addr1=s1_info[1], country1=s1_info[2],
                name2=tgt_info[0], addr2=tgt_info[1], country2=tgt_info[2],
                strategy_count=strat_cnt
            )
            feat["source1_entity_id"] = s1_id
            feat["candidate_entity_id"] = cand_id
            feature_rows.append(feat)

    df_features = pd.DataFrame(feature_rows)
    return df_features


# -----------------------------------------------------------------------------
# Deterministic Scoring & Match Selection Engine
# -----------------------------------------------------------------------------

def score_candidates(
    df_features: pd.DataFrame,
    config: Optional[ScoringConfig] = None
) -> pd.DataFrame:
    """
    Calculate deterministic match scores for candidate pairs.
    
    Args:
        df_features: DataFrame of candidate pair features output by `build_candidate_features`.
        config: Configurable ScoringConfig instance with weights and penalties.
        
    Returns:
        pd.DataFrame with added `match_score` column.
    """
    if config is None:
        config = ScoringConfig()
        
    df_out = df_features.copy()
    if df_out.empty:
        df_out["match_score"] = 0.0
        return df_out

    # Sum of configured positive signal weights for normalization
    total_pos_weight = (
        config.w_exact_name +
        config.w_exact_addr +
        config.w_name_sim +
        config.w_addr_sim +
        config.w_name_tok_sim +
        config.w_addr_tok_sim +
        config.w_multi_strategy
    )
    
    # Base additive signal score normalized by total positive weight
    raw_score = (
        config.w_exact_name * df_out["exact_name_match"] +
        config.w_exact_addr * df_out["exact_address_match"] +
        config.w_name_sim * df_out["name_similarity"] +
        config.w_addr_sim * df_out["address_similarity"] +
        config.w_name_tok_sim * df_out["name_token_similarity"] +
        config.w_addr_tok_sim * df_out["address_token_similarity"] +
        config.w_multi_strategy * df_out["is_multi_strategy"]
    )
    
    weighted_signal = raw_score / total_pos_weight if total_pos_weight > 0 else raw_score
    
    # Length difference penalty
    len_penalty = config.len_diff_penalty_factor * (df_out["name_length_diff"] + df_out["address_length_diff"])
    score_after_penalty = np.maximum(0.0, weighted_signal - len_penalty)
    
    # Country mismatch multiplicative penalty: if country_match == 0.0, apply penalty factor
    country_multiplier = np.where(df_out["country_match"] == 0.0, config.country_mismatch_penalty, 1.0)
    
    final_score = score_after_penalty * country_multiplier
    
    # Normalize score between 0.0 and 1.0 cleanly
    df_out["match_score"] = np.round(np.clip(final_score, 0.0, 1.0), 4)
    return df_out


def select_matches(
    df_scored: pd.DataFrame,
    threshold: Optional[float] = None,
    mode: str = "top_1_per_s1",
    config: Optional[ScoringConfig] = None
) -> pd.DataFrame:
    """
    Select final predicted matches based on score thresholding and selection mode.
    
    Args:
        df_scored: DataFrame containing candidate scores (from `score_candidates`).
        threshold: Score threshold for match selection (defaults to `config.min_score_threshold`).
        mode: Selection policy:
              - 'top_1_per_s1': Select highest-scoring candidate per Source 1 record if score >= threshold.
              - 'all_above_threshold': Select all candidates with score >= threshold.
        config: ScoringConfig instance.
        
    Returns:
        pd.DataFrame containing columns ['source1_entity_id', 'candidate_entity_id', 'match_score'].
    """
    if config is None:
        config = ScoringConfig()
        
    cutoff = threshold if threshold is not None else config.min_score_threshold
    
    if df_scored.empty or "match_score" not in df_scored.columns:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_score"])

    # Filter above threshold
    filtered = df_scored[df_scored["match_score"] >= cutoff].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_score"])

    if mode == "top_1_per_s1":
        # Sort by s1_id and match_score descending
        filtered = filtered.sort_values(by=["source1_entity_id", "match_score"], ascending=[True, False])
        filtered = filtered.drop_duplicates(subset=["source1_entity_id"], keep="first")
    elif mode == "all_above_threshold":
        filtered = filtered.sort_values(by=["source1_entity_id", "match_score"], ascending=[True, False])
        
    return filtered[["source1_entity_id", "candidate_entity_id", "match_score"]].reset_index(drop=True)
