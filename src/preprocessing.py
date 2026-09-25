"""
Preprocessing module for Business Entity Resolution.

Contains reusable, deterministic functions for normalizing business names,
addresses, and country codes/labels conservatively without destroying critical matching information.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional
import pandas as pd


# -----------------------------------------------------------------------------
# Common Regex & Keyword Maps
# -----------------------------------------------------------------------------

def remove_accents(text: str) -> str:
    """Strip accents and diacritics using Unicode NFKD decomposition."""
    if not text or not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


# Legal entity suffixes (sorted by length descending for greedy matching)
LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "incorporated", "corporation",
    "limited company", "gesellschaft mit beschraenkter haftung",
    "gesellschaft mit beschrankter haftung", "gmbh", "limited",
    "llc", "inc", "ltd", "corp", "co", "company", "plc", "sa", "sas",
    "bv", "nv", "ag", "ab", "oy", "sl", "srl", "spa", "z oo", "sp z o o",
    "cia", "s.a.", "s.a", "s.r.l.", "s.r.l", "l.l.c.", "l.l.c", "inc.",
    "ltd.", "corp.", "co.", "pvt.", "private"
]

# Address token standardization dictionary (conservative mapping)
ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "parkway": "pkwy",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "highway": "hwy",
    "court": "ct",
    "place": "pl",
    "square": "sq",
    "expressway": "expy",
    "extension": "ext",
    "route": "rt",
    "post office box": "pobox",
    "po box": "pobox",
    "p.o. box": "pobox",
    "p.o box": "pobox",
    "po. box": "pobox",
    "pobox": "pobox",
}

# Compile regexes for performance
RE_PUNCT = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\d+")
RE_AMPERSAND = re.compile(r"&")


# -----------------------------------------------------------------------------
# Normalization Functions
# -----------------------------------------------------------------------------

def normalize_country(country: str) -> str:
    """
    Normalize country string label.
    Treats country strictly as an open string label without assuming fixed schemas.
    """
    if not country or pd.isna(country) or not isinstance(country, str):
        return "unknown"
    cleaned = remove_accents(country.strip().lower())
    cleaned = RE_PUNCT.sub(" ", cleaned)
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip()
    return cleaned if cleaned else "unknown"


def clean_text_basic(text: str) -> str:
    """
    Basic conservative text normalization:
    1. Strip accents
    2. Lowercase
    3. Replace '&' with ' and '
    4. Replace punctuation with space
    5. Collapse whitespace
    """
    if not text or pd.isna(text) or not isinstance(text, str):
        return ""
    text = remove_accents(text.strip().lower())
    text = RE_AMPERSAND.sub(" and ", text)
    text = RE_PUNCT.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def strip_legal_suffixes(normalized_name: str) -> str:
    """
    Remove standard legal suffixes from a normalized business name string.
    Only strips suffixes appearing at the end of the string to avoid destroying core names.
    """
    if not normalized_name:
        return ""
    
    tokens = normalized_name.split()
    if not tokens:
        return ""
        
    # Check multi-word and single-word legal suffixes at tail
    changed = True
    while changed and tokens:
        changed = False
        joined = " ".join(tokens)
        for suffix in LEGAL_SUFFIXES:
            if joined == suffix:
                # If name is JUST legal suffix, retain token to avoid empty string
                return tokens[0]
            if joined.endswith(" " + suffix):
                suffix_len = len(suffix.split())
                tokens = tokens[:-suffix_len]
                changed = True
                break
                
    res = " ".join(tokens).strip()
    return res if res else normalized_name


def normalize_business_name(name: str, strip_legal: bool = False) -> str:
    """
    Normalize business name conservatively.
    
    Args:
        name: Raw business name string.
        strip_legal: Whether to strip common legal entity suffixes.
        
    Returns:
        Cleaned, normalized business name string.
    """
    cleaned = clean_text_basic(name)
    if strip_legal and cleaned:
        cleaned = strip_legal_suffixes(cleaned)
    return cleaned


def normalize_business_address(address: str) -> str:
    """
    Normalize business address conservatively:
    1. Basic clean (accents, lowercase, punctuation, whitespace)
    2. Standardize common street/unit abbreviations word-by-word
    """
    cleaned = clean_text_basic(address)
    if not cleaned:
        return ""
        
    tokens = cleaned.split()
    norm_tokens = []
    for t in tokens:
        norm_tokens.append(ADDRESS_ABBREVIATIONS.get(t, t))
        
    return " ".join(norm_tokens)


# -----------------------------------------------------------------------------
# Key Extraction Helpers for Blocking
# -----------------------------------------------------------------------------

def extract_name_prefix(normalized_name: str, length: int = 4) -> str:
    """Extract first N characters of normalized business name (ignoring whitespace)."""
    compact = normalized_name.replace(" ", "")
    return compact[:length] if compact else ""


def extract_first_k_tokens(text: str, k: int = 2) -> str:
    """Extract first k words from normalized string joined by space."""
    tokens = text.split()
    return " ".join(tokens[:k]) if tokens else ""


def extract_name_tokens(normalized_name: str, min_len: int = 3) -> List[str]:
    """Extract tokens of minimum length from normalized business name."""
    if not normalized_name:
        return []
    return [t for t in normalized_name.split() if len(t) >= min_len]


def extract_address_digits(normalized_address: str) -> str:
    """Extract all digits (e.g., street/building numbers) from address concatenated."""
    digits = RE_DIGITS.findall(normalized_address)
    return "".join(digits) if digits else ""


def extract_address_street_key(normalized_address: str) -> str:
    """
    Derive address key combining first digit sequence (building #) 
    and first non-numeric word (street name).
    """
    if not normalized_address:
        return ""
        
    tokens = normalized_address.split()
    digit_part = ""
    street_part = ""
    
    for t in tokens:
        if not digit_part and t.isdigit():
            digit_part = t
        elif not street_part and t.isalpha() and len(t) >= 2:
            street_part = t
            
        if digit_part and street_part:
            break
            
    if digit_part or street_part:
        return f"{digit_part}_{street_part}"
    return extract_first_k_tokens(normalized_address, k=2)


# -----------------------------------------------------------------------------
# Pandas Dataframe Preprocessing Helper
# -----------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    addr_col: str = "business_address",
    country_col: str = "country",
    strip_legal: bool = False
) -> pd.DataFrame:
    """
    Apply conservative normalization across DataFrame columns.
    Creates new normalized columns:
    - 'norm_name'
    - 'norm_address'
    - 'norm_country'
    
    Args:
        df: Input DataFrame containing entity records.
        name_col: Column name for business name.
        addr_col: Column name for business address.
        country_col: Column name for country.
        strip_legal: Whether to strip legal suffixes for 'norm_name'.
        
    Returns:
        DataFrame with added normalized columns.
    """
    df_out = df.copy()
    
    # Fill missing values gracefully
    raw_names = df_out[name_col].fillna("").astype(str)
    raw_addrs = df_out[addr_col].fillna("").astype(str)
    raw_countries = df_out[country_col].fillna("").astype(str)
    
    df_out["norm_country"] = raw_countries.apply(normalize_country)
    df_out["norm_name"] = raw_names.apply(lambda s: normalize_business_name(s, strip_legal=strip_legal))
    df_out["norm_address"] = raw_addrs.apply(normalize_business_address)
    
    return df_out
