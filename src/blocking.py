"""
Blocking module for Business Entity Resolution.

Supports several independently testable blocking strategies and evaluation functions
to measure blocking recall, candidate statistics, and reduction ratio.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional

from src.preprocessing import (
    normalize_country,
    normalize_business_name,
    normalize_business_address,
    extract_name_prefix,
    extract_first_k_tokens,
    extract_address_street_key
)


# -----------------------------------------------------------------------------
# Base Blocking Strategy & Inverted Index Engine
# -----------------------------------------------------------------------------

class BaseBlockingStrategy(ABC):
    """
    Abstract base class for independent blocking strategies.
    
    Each blocking strategy implements `extract_keys(row)` which returns
    one or more string blocking keys for an entity record.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        
    @abstractmethod
    def extract_keys(self, row: pd.Series) -> List[str]:
        """Extract blocking keys for a single row series or dict."""
        pass

    def build_index(self, target_dfs: List[pd.DataFrame]) -> None:
        """
        Build inverted index key -> list of target entity IDs (S2 / S3 records).
        Memory-conscious iteration over target dataframes.
        """
        self.inverted_index.clear()
        
        for df in target_dfs:
            # Ensure normalized columns exist
            name_col = "norm_name" if "norm_name" in df.columns else "business_name"
            addr_col = "norm_address" if "norm_address" in df.columns else "business_address"
            country_col = "norm_country" if "norm_country" in df.columns else "country"
            id_col = "entity_id"
            
            for row in df[[id_col, name_col, addr_col, country_col]].itertuples(index=False):
                ent_id, norm_name, norm_addr, norm_country = row
                
                # Build pseudo-series for key extraction
                item = {
                    "entity_id": ent_id,
                    "norm_name": norm_name if isinstance(norm_name, str) else "",
                    "norm_address": norm_addr if isinstance(norm_addr, str) else "",
                    "norm_country": norm_country if isinstance(norm_country, str) else ""
                }
                
                keys = self.extract_keys(item)
                for key in keys:
                    if key:
                        self.inverted_index[key].append(ent_id)

    def generate_candidates(self, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Generate candidate target entity IDs for each Source 1 record.
        
        Returns:
            Dict[str, Set[str]]: Mapping from S1 entity_id to set of candidate S2/S3 entity_ids.
        """
        candidates: Dict[str, Set[str]] = {}
        
        name_col = "norm_name" if "norm_name" in s1_df.columns else "business_name"
        addr_col = "norm_address" if "norm_address" in s1_df.columns else "business_address"
        country_col = "norm_country" if "norm_country" in s1_df.columns else "country"
        id_col = "entity_id"
        
        for row in s1_df[[id_col, name_col, addr_col, country_col]].itertuples(index=False):
            ent_id, norm_name, norm_addr, norm_country = row
            
            item = {
                "entity_id": ent_id,
                "norm_name": norm_name if isinstance(norm_name, str) else "",
                "norm_address": norm_addr if isinstance(norm_addr, str) else "",
                "norm_country": norm_country if isinstance(norm_country, str) else ""
            }
            
            keys = self.extract_keys(item)
            cand_set: Set[str] = set()
            for key in keys:
                if key in self.inverted_index:
                    cand_set.update(self.inverted_index[key])
            
            candidates[ent_id] = cand_set
            
        return candidates


# -----------------------------------------------------------------------------
# Concrete Blocking Strategies
# -----------------------------------------------------------------------------

class ExactNameCountryBlocking(BaseBlockingStrategy):
    """
    Strategy 1: Exact normalized business name + country.
    Blocks records that share identical normalized business names within the same country.
    """
    def __init__(self):
        super().__init__("Exact Normalized Name + Country")

    def extract_keys(self, row: Any) -> List[str]:
        name = row["norm_name"]
        country = row["norm_country"]
        if not name:
            return []
        return [f"name:{name}|c:{country}"]


class ExactAddressCountryBlocking(BaseBlockingStrategy):
    """
    Strategy 2: Exact normalized address + country.
    Blocks records that share identical normalized addresses within the same country.
    """
    def __init__(self):
        super().__init__("Exact Normalized Address + Country")

    def extract_keys(self, row: Any) -> List[str]:
        addr = row["norm_address"]
        country = row["norm_country"]
        if not addr:
            return []
        return [f"addr:{addr}|c:{country}"]


class NamePrefixCountryBlocking(BaseBlockingStrategy):
    """
    Strategy 3: Normalized name prefix / token keys + country.
    Blocks records sharing the first N characters (e.g. 4) of normalized name + country.
    """
    def __init__(self, prefix_length: int = 4):
        super().__init__(f"Name Prefix ({prefix_length} chars) + Country")
        self.prefix_length = prefix_length

    def extract_keys(self, row: Any) -> List[str]:
        name = row["norm_name"]
        country = row["norm_country"]
        if not name:
            return []
        prefix = extract_name_prefix(name, length=self.prefix_length)
        if not prefix:
            return []
        return [f"npref:{prefix}|c:{country}"]


class AddressDerivedCountryBlocking(BaseBlockingStrategy):
    """
    Strategy 4: Address-derived keys + country.
    Blocks records sharing street number + street token key within the same country.
    """
    def __init__(self):
        super().__init__("Address-Derived Key + Country")

    def extract_keys(self, row: Any) -> List[str]:
        addr = row["norm_address"]
        country = row["norm_country"]
        if not addr:
            return []
        street_key = extract_address_street_key(addr)
        if not street_key:
            return []
        return [f"addr_key:{street_key}|c:{country}"]


class CombinedSignalBlocking(BaseBlockingStrategy):
    """
    Strategy 5: Combination of name/address signals.
    Creates composite blocking keys combining name prefix and address key within country.
    """
    def __init__(self, prefix_length: int = 3):
        super().__init__("Combined Name Prefix + Address Key + Country")
        self.prefix_length = prefix_length

    def extract_keys(self, row: Any) -> List[str]:
        name = row["norm_name"]
        addr = row["norm_address"]
        country = row["norm_country"]
        
        prefix = extract_name_prefix(name, length=self.prefix_length) if name else "noname"
        addr_key = extract_address_street_key(addr) if addr else "noaddr"
        
        if prefix == "noname" and addr_key == "noaddr":
            return []
            
        return [f"comb:{prefix}_{addr_key}|c:{country}"]


# -----------------------------------------------------------------------------
# Evaluation & Metric Functions
# -----------------------------------------------------------------------------

def parse_ground_truth(gt_input: Any) -> Dict[str, Set[str]]:
    """
    Parse ground truth into a standardized dictionary:
    `s1_id -> set of true_match_ids (from S2 / S3)`.
    
    Accepts DataFrame, dict, or iterable of pairs.
    """
    if isinstance(gt_input, dict):
        # Already a dict mapping s1_id -> set/list of match_ids
        return {k: set(v) if not isinstance(v, set) else v for k, v in gt_input.items()}
        
    gt_dict = defaultdict(set)
    
    if isinstance(gt_input, pd.DataFrame):
        cols = gt_input.columns
        if "s1_id" in cols and "match_id" in cols:
            for s1_id, match_id in zip(gt_input["s1_id"], gt_input["match_id"]):
                if pd.notna(s1_id) and pd.notna(match_id):
                    gt_dict[str(s1_id)].add(str(match_id))
        elif "entity_id" in cols and "target_id" in cols:
            for s1_id, match_id in zip(gt_input["entity_id"], gt_input["target_id"]):
                if pd.notna(s1_id) and pd.notna(match_id):
                    gt_dict[str(s1_id)].add(str(match_id))
        else:
            # Fallback to first two columns
            s1_col, match_col = cols[0], cols[1]
            for s1_id, match_id in zip(gt_input[s1_col], gt_input[match_col]):
                if pd.notna(s1_id) and pd.notna(match_id):
                    gt_dict[str(s1_id)].add(str(match_id))
                    
    return dict(gt_dict)


def calculate_candidate_stats(candidates: Dict[str, Set[str]]) -> Tuple[np.ndarray, float, float, int]:
    """
    Calculate candidate counts statistics per S1 entity.
    
    Returns:
        counts (np.ndarray): Array of candidate counts per S1 entity.
        avg_candidates (float): Average candidates per S1.
        median_candidates (float): Median candidates per S1.
        max_candidates (int): Maximum candidates per S1.
    """
    if not candidates:
        return np.array([0]), 0.0, 0.0, 0
        
    counts = np.array([len(cands) for cands in candidates.values()], dtype=np.int64)
    avg_cand = float(np.mean(counts))
    median_cand = float(np.median(counts))
    max_cand = int(np.max(counts))
    
    return counts, avg_cand, median_cand, max_cand


def calculate_reduction_ratio(
    candidates: Dict[str, Set[str]],
    num_s1: int,
    num_target_total: int
) -> float:
    """
    Calculate reduction ratio relative to full Cartesian comparison space:
    RR = 1 - (Total Candidate Pair Comparisons / (num_s1 * num_target_total))
    """
    if num_s1 <= 0 or num_target_total <= 0:
        return 1.0
        
    total_comparisons = sum(len(cands) for cands in candidates.values())
    full_space = float(num_s1) * float(num_target_total)
    
    reduction_ratio = 1.0 - (total_comparisons / full_space)
    return max(0.0, min(1.0, reduction_ratio))


def evaluate_blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]]
) -> Tuple[int, int, float]:
    """
    Evaluate strict blocking recall.
    Answers: "For each validation Source 1 entity, did the blocking stage include ALL of its true S2/S3 matches?"
    
    Returns:
        num_validation_s1 (int): Number of validation S1 entities with true matches.
        num_fully_recovered (int): Number of S1 entities with ALL true matches present in candidate set.
        blocking_recall (float): Proportion of validation S1 entities fully recovered.
    """
    valid_gt = {s1_id: true_matches for s1_id, true_matches in ground_truth.items() if len(true_matches) > 0}
    num_validation_s1 = len(valid_gt)
    
    if num_validation_s1 == 0:
        return 0, 0, 0.0
        
    fully_recovered_count = 0
    
    for s1_id, true_matches in valid_gt.items():
        s1_cands = candidates.get(s1_id, set())
        if true_matches.issubset(s1_cands):
            fully_recovered_count += 1
            
    recall = fully_recovered_count / float(num_validation_s1)
    return num_validation_s1, fully_recovered_count, recall


def evaluate_blocking_strategy(
    strategy: BaseBlockingStrategy,
    s1_df: pd.DataFrame,
    target_dfs: List[pd.DataFrame],
    ground_truth: Any
) -> Dict[str, Any]:
    """
    Independently evaluate a single blocking strategy on validation data.
    
    Args:
        strategy: Instance of BaseBlockingStrategy.
        s1_df: Source 1 DataFrame.
        target_dfs: List of target DataFrames (e.g. [s2_df, s3_df]).
        ground_truth: Ground truth mapping or DataFrame.
        
    Returns:
        Dict reporting:
        - strategy_name
        - num_validation_s1
        - num_fully_recovered
        - blocking_recall
        - avg_candidates
        - median_candidates
        - max_candidates
        - reduction_ratio
    """
    # 1. Build strategy index
    strategy.build_index(target_dfs)
    
    # 2. Generate candidates for S1
    candidates = strategy.generate_candidates(s1_df)
    
    # 3. Parse GT and calculate recall
    gt_dict = parse_ground_truth(ground_truth)
    num_val_s1, num_recovered, recall = evaluate_blocking_recall(candidates, gt_dict)
    
    # 4. Calculate candidate statistics
    counts, avg_cand, median_cand, max_cand = calculate_candidate_stats(candidates)
    
    # 5. Calculate reduction ratio
    num_s1 = len(s1_df)
    num_target_total = sum(len(df) for df in target_dfs)
    rr = calculate_reduction_ratio(candidates, num_s1, num_target_total)
    
    report = {
        "strategy_name": strategy.name,
        "num_validation_s1": num_val_s1,
        "num_fully_recovered": num_recovered,
        "blocking_recall": round(recall, 6),
        "avg_candidates": round(avg_cand, 2),
        "median_candidates": round(median_cand, 2),
        "max_candidates": max_cand,
        "reduction_ratio": round(rr, 6)
    }
    
    return report
