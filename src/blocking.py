"""
Blocking module for Business Entity Resolution.

Supports several independently testable blocking strategies and evaluation functions
to measure blocking recall, candidate statistics, and reduction ratio.
Memory-optimized using contiguous target entity ID arrays and compact int32 inverted indices.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import gc
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
    
    Memory-optimized:
    - Target entity IDs are stored once in a contiguous 1D NumPy array (`target_entity_ids`).
    - The inverted index maps blocking keys to compact int32 NumPy arrays (`inverted_index`).
    """
    
    def __init__(self, name: str, max_candidates_per_key: Optional[int] = None):
        self.name = name
        self.max_candidates_per_key = max_candidates_per_key
        self.target_entity_ids: np.ndarray = np.array([], dtype=object)
        self.inverted_index: Dict[str, np.ndarray] = {}
        
    @abstractmethod
    def extract_keys(self, row: pd.Series) -> List[str]:
        """Extract blocking keys for a single row series or dict."""
        pass

    def build_index(self, target_dfs: List[pd.DataFrame]) -> None:
        """
        Build memory-compact inverted index key -> np.ndarray(int32).
        Stores target entity IDs in a single contiguous 1D array to minimize RAM overhead.
        """
        self.inverted_index.clear()
        
        all_ids_list = []
        key_builder = defaultdict(list)
        
        offset = 0
        for df in target_dfs:
            id_col = "entity_id"
            name_col = "norm_name" if "norm_name" in df.columns else "business_name"
            addr_col = "norm_address" if "norm_address" in df.columns else "business_address"
            country_col = "norm_country" if "norm_country" in df.columns else "country"
            
            ids = df[id_col].fillna("").astype(str).to_numpy()
            names = df[name_col].fillna("").astype(str).to_numpy()
            addrs = df[addr_col].fillna("").astype(str).to_numpy()
            countries = df[country_col].fillna("").astype(str).to_numpy()
            
            all_ids_list.append(ids)
            
            for idx, (ent_id, norm_name, norm_addr, norm_country) in enumerate(zip(ids, names, addrs, countries)):
                item = {
                    "entity_id": ent_id,
                    "norm_name": norm_name,
                    "norm_address": norm_addr,
                    "norm_country": norm_country
                }
                keys = self.extract_keys(item)
                global_idx = offset + idx
                for k in keys:
                    if k:
                        key_builder[k].append(global_idx)
                        
            offset += len(df)
            
        self.target_entity_ids = np.concatenate(all_ids_list) if all_ids_list else np.array([], dtype=object)
        
        # Convert integer index lists to compact C-contiguous int32 arrays
        self.inverted_index = {}
        for k, v in key_builder.items():
            if self.max_candidates_per_key and len(v) > self.max_candidates_per_key:
                v = v[:self.max_candidates_per_key]
            self.inverted_index[k] = np.array(v, dtype=np.int32)
            
        # Explicit garbage collection of temporary lists
        del key_builder
        del all_ids_list
        gc.collect()

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
        
        ids = s1_df[id_col].fillna("").astype(str).to_numpy()
        names = s1_df[name_col].fillna("").astype(str).to_numpy()
        addrs = s1_df[addr_col].fillna("").astype(str).to_numpy()
        countries = s1_df[country_col].fillna("").astype(str).to_numpy()
        
        target_ids = self.target_entity_ids
        
        for ent_id, norm_name, norm_addr, norm_country in zip(ids, names, addrs, countries):
            item = {
                "entity_id": ent_id,
                "norm_name": norm_name,
                "norm_address": norm_addr,
                "norm_country": norm_country
            }
            
            keys = self.extract_keys(item)
            cand_idx_set: Set[int] = set()
            for key in keys:
                if key in self.inverted_index:
                    cand_idx_set.update(self.inverted_index[key])
            
            if cand_idx_set:
                cand_ids = {target_ids[idx] for idx in cand_idx_set}
            else:
                cand_ids = set()
                
            candidates[ent_id] = cand_ids
            
        return candidates


# -----------------------------------------------------------------------------
# Concrete Blocking Strategies
# -----------------------------------------------------------------------------

class ExactNameCountryBlocking(BaseBlockingStrategy):
    """
    Strategy 1: Exact normalized business name + country.
    Blocks records that share identical normalized business names within the same country.
    """
    def __init__(self, max_candidates_per_key: Optional[int] = None):
        super().__init__("Exact Normalized Name + Country", max_candidates_per_key=max_candidates_per_key)

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
    def __init__(self, max_candidates_per_key: Optional[int] = None):
        super().__init__("Exact Normalized Address + Country", max_candidates_per_key=max_candidates_per_key)

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
    def __init__(self, prefix_length: int = 4, max_candidates_per_key: Optional[int] = None):
        super().__init__(f"Name Prefix ({prefix_length} chars) + Country", max_candidates_per_key=max_candidates_per_key)
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
    def __init__(self, max_candidates_per_key: Optional[int] = None):
        super().__init__("Address-Derived Key + Country", max_candidates_per_key=max_candidates_per_key)

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
    def __init__(self, prefix_length: int = 3, max_candidates_per_key: Optional[int] = None):
        super().__init__("Combined Name Prefix + Address Key + Country", max_candidates_per_key=max_candidates_per_key)
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
    """
    strategy.build_index(target_dfs)
    candidates = strategy.generate_candidates(s1_df)
    
    gt_dict = parse_ground_truth(ground_truth)
    num_val_s1, num_recovered, recall = evaluate_blocking_recall(candidates, gt_dict)
    
    counts, avg_cand, median_cand, max_cand = calculate_candidate_stats(candidates)
    
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
