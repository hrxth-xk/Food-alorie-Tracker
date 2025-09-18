# src/api_client.py
import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, Dict, List, Tuple

# explicit load from repo root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

from rapidfuzz import process, fuzz

USDA_API_KEY = os.getenv("USDA_API_KEY")
MASTER_CSV = os.getenv("MASTER_CSV", "data/processed/master_food.csv")
CACHE_PATH = os.getenv("API_CACHE_PATH", "data/cache.json")
CACHE_MAX_ENTRIES = 1000
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours


def _ensure_cache_dir():
    d = os.path.dirname(CACHE_PATH) or "."
    os.makedirs(d, exist_ok=True)


def _load_cache() -> Dict[str, Dict]:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict]):
    _ensure_cache_dir()
    # simple trimming
    if len(cache) > CACHE_MAX_ENTRIES:
        keys = sorted(cache.keys(), key=lambda k: cache[k].get("ts", 0))
        for k in keys[: len(keys) // 2]:
            cache.pop(k, None)
    try:
        with open(CACHE_PATH, "w", encoding="utf8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def _cache_get(key: str) -> Optional[List[Dict]]:
    cache = _load_cache()
    ent = cache.get(key)
    if not ent:
        return None
    ts = ent.get("ts", 0)
    if (time.time() - ts) > CACHE_TTL_SECONDS:
        # expired
        cache.pop(key, None)
        _save_cache(cache)
        return None
    return ent.get("results")


def _cache_set(key: str, results: List[Dict]):
    cache = _load_cache()
    cache[key] = {"ts": int(time.time()), "results": results}
    _save_cache(cache)


def load_local_master() -> pd.DataFrame:
    if not os.path.exists(MASTER_CSV):
        return pd.DataFrame(columns=["name", "calories", "protein", "fat", "carbs", "source"])
    return pd.read_csv(MASTER_CSV)


def search_local_substring(query: str, limit: int = 20) -> List[Dict]:
    df = load_local_master()
    if df.empty:
        return []
    q = query.strip().lower()
    mask = df["name"].astype(str).str.lower().str.contains(q, na=False)
    return df[mask].head(limit).to_dict(orient="records")


def search_local_fuzzy(query: str, limit: int = 20, score_cutoff: int = 60) -> List[Dict]:
    """
    Fuzzy match local master names using rapidfuzz.
    Returns matching rows ordered by score.
    """
    df = load_local_master()
    if df.empty:
        return []
    names = df["name"].astype(str).tolist()
    # get top matches (name, score, index)
    matches: List[Tuple[str, int, int]] = process.extract(
        query, names, scorer=fuzz.WRatio, limit=limit
    )
    results = []
    for name, score, idx in matches:
        if score < score_cutoff:
            continue
        row = df.iloc[idx].to_dict()
        row["_match_score"] = score
        results.append(row)
    return results


def usda_search(query: str, limit: int = 10) -> Optional[Dict]:
    if not USDA_API_KEY:
        return None
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key": USDA_API_KEY, "query": query, "pageSize": limit}
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def normalize_usda_item(item: Dict) -> Dict:
    name = item.get("description") or item.get("lowercaseDescription") or "Unknown"
    nutrients = {}
    for n in item.get("foodNutrients", []):
        k = (n.get("nutrientName") or "").lower()
        nutrients[k] = n.get("value")
    calories = (
        nutrients.get("energy")
        or nutrients.get("energy (kcal)")
        or nutrients.get("kilocalories")
        or nutrients.get("calories")
    )
    protein = nutrients.get("protein")
    fat = nutrients.get("total lipid (fat)") or nutrients.get("fat")
    carbs = nutrients.get("carbohydrate, by difference") or nutrients.get("carbohydrate")
    return {
        "name": name,
        "calories": calories,
        "protein": protein,
        "fat": fat,
        "carbs": carbs,
        "source": "usda",
        "fdcId": item.get("fdcId"),
    }


def search(query: str, use_api_first: bool = True, limit: int = 20) -> List[Dict]:
    """
    Search workflow:
    1) check cache
    2) if use_api_first and API key -> USDA (cache results)
    3) fuzzy local search (cache)
    4) substring local fallback (cache)
    """
    if not query or not query.strip():
        return []

    key = query.strip().lower()
    # 1) cache
    cached = _cache_get(key)
    if cached:
        return cached[:limit]

    # 2) USDA
    if use_api_first and USDA_API_KEY:
        js = usda_search(query, limit=limit)
        if js and js.get("foods"):
            items = [normalize_usda_item(it) for it in js["foods"]]
            _cache_set(key, items)
            return items

    # 3) fuzzy local
    fuzzy = search_local_fuzzy(query, limit=limit)
    if fuzzy:
        _cache_set(key, fuzzy)
        return fuzzy

    # 4) substring local fallback
    local = search_local_substring(query, limit=limit)
    if local:
        _cache_set(key, local)
    return local



# --- add to src/api_client.py (or src/app.py) ---

import re
from typing import List, Dict

_normalize_re = re.compile(r"[^\w\s]", flags=re.UNICODE)
_parenthetical_re = re.compile(r"\(.*?\)")

def _normalize_name(name: str) -> str:
    """Lower, remove parentheses and punctuation, collapse whitespace."""
    if not name:
        return ""
    s = str(name).lower()
    s = _parenthetical_re.sub("", s)        # remove "(...)" pieces
    s = _normalize_re.sub(" ", s)           # remove punctuation
    s = " ".join(s.split())                 # normalize whitespace
    return s.strip()

def _completeness_score(item: Dict) -> int:
    """Simple score: number of main nutrient fields present + id + prefer local source."""
    score = 0
    for k in ("calories", "protein", "carbs", "fat"):
        try:
            if item.get(k) is not None and str(item.get(k)).strip() != "":
                score += 1
        except Exception:
            pass
    # stable id bonus
    if item.get("fdcId") or item.get("id") or item.get("local_id"):
        score += 1
    # optional: prefer local DB entries (bump score)
    src = (item.get("source") or "").lower()
    if src == "local":
        score += 1
    return score

def dedupe_results(results: List[Dict], keep_all_collapsed: bool = False) -> List[Dict]:
    """
    Collapse obvious duplicate results by normalized name.
    - results: list of dicts from search()
    - keep_all_collapsed: if True, each returned item will include a key "_collapsed" listing other items collapsed into it.
    """
    if not results:
        return results

    groups = {}
    order = []  # first-seen normalized name order
    for item in results:
        name = item.get("name") or item.get("description") or item.get("food") or ""
        key = _normalize_name(name)
        if not key:
            # fallback to raw name or an id if name empty
            key = f"__id_{item.get('fdcId') or item.get('id') or hash(name)}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    deduped = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            # single item
            item = dict(group[0])
            if keep_all_collapsed:
                item["_collapsed"] = []
            deduped.append(item)
            continue

        # choose best candidate by completeness score, tie-break by source preference, then by length of name (shorter preferred)
        scored = []
        for it in group:
            sc = _completeness_score(it)
            # small tiebreaker to prefer shorter names (likely more canonical)
            name_len = len(str(it.get("name") or ""))
            scored.append((sc, -name_len, it))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)  # highest score, shortest name first
    best = dict(scored[0][2])
    if keep_all_collapsed:
        # keep other items in a compressed form for optional UI expansion
        collapsed = [ { "name": i.get("name"), "source": i.get("source"), "calories": i.get("calories") } for (_,_,i) in scored[1:] ]
        best["_collapsed"] = collapsed
    deduped.append(best)

    return deduped
