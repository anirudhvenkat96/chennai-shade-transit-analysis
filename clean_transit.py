"""
Transit Dataset Cleaner — Chennai Shade Analysis
Cleans metro and MRTS datasets; bus stops are kept as-is.
Run in the activated chennai-shade conda environment.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
import unicodedata
import fiona

fiona.drvsupport.supported_drivers["KML"] = "rw"

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BBOX      = (79.9, 12.8, 80.4, 13.3)           # minx miny maxx maxy
DEDUP_M   = 50                                   # metres — merge radius
UTM_CRS   = "EPSG:32644"                         # UTM zone 44N for Chennai

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

log_lines = []

def log(s=""):
    print(s, flush=True)
    log_lines.append(str(s))

def save_log():
    path = os.path.join(OUT_DIR, "cleaning_log.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Log saved → {path}", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_tamil(text: str) -> bool:
    """True if the string contains at least one Tamil Unicode character."""
    for ch in text:
        if "\u0B80" <= ch <= "\u0BFF":
            return True
    return False

def project(gdf):
    """Return gdf projected to UTM 44N; set CRS=4326 if missing."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(UTM_CRS)

def centroid_of_group(coords):
    """Return (x, y) centroid of a list of (x, y) tuples."""
    arr = np.array(coords)
    return arr.mean(axis=0)

# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self):
        from collections import defaultdict
        d = defaultdict(list)
        for i in range(len(self.parent)):
            d[self.find(i)].append(i)
        return list(d.values())

# ── Metro cleaner ─────────────────────────────────────────────────────────────

# Patterns that identify entrance/exit nodes and depot nodes.
# Matched as case-insensitive substrings (after stripping the name).
_ENTRANCE_EXIT_PATTERNS = [
    "entrance", "exit", "depot",
    " north", " south", " east", " west",
    " a1", " a2", " a3", " a4",
    " a", " b", " c", " d",
]

def is_non_station(name: str) -> bool:
    """Return True if this name looks like an entrance, exit, or depot node."""
    nl = name.lower()
    for pat in _ENTRANCE_EXIT_PATTERNS:
        if nl.endswith(pat):
            return True
    # Also catch standalone mid-name occurrences for entrance/exit/depot
    for word in ("entrance", "exit", "depot"):
        if word in nl:
            return True
    return False

def clean_metro():
    log("=" * 60)
    log("  METRO — CLEANING (PASS 2)")
    log("=" * 60)

    # Read the original raw file (not the previous cleaned version) so this
    # function is idempotent and always starts from source truth.
    path = os.path.join(DATA_DIR, "chennai-metro-staions.geojson")
    gdf  = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    raw = len(gdf)
    log(f"\n  Raw features         : {raw}")

    # ── Step 1: Drop null / NaN names ────────────────────────────────────
    name_valid = (
        gdf["name"].notna()
        & (gdf["name"].astype(str).str.strip() != "")
        & (~gdf["name"].astype(str).str.lower().isin(["nan", "none", ""]))
    )
    gdf = gdf[name_valid].copy().reset_index(drop=True)
    log(f"  After null-name drop : {len(gdf)}  (removed {raw - len(gdf)})")

    # ── Step 2: Drop entrance / exit / depot nodes ────────────────────────
    names_raw = [str(n).strip() for n in gdf["name"].tolist()]
    keep_mask = [not is_non_station(n) for n in names_raw]
    removed_nodes = keep_mask.count(False)
    if removed_nodes:
        log(f"\n  Entrance/exit/depot nodes removed ({removed_nodes}):")
        for n, keep in zip(names_raw, keep_mask):
            if not keep:
                log(f"    - {n}")
    gdf = gdf[keep_mask].copy().reset_index(drop=True)
    log(f"  After node filter    : {len(gdf)}")

    # ── Step 3: Merge nearby nodes (100 m) — Tamil+English and same-name ──
    proj   = project(gdf)
    coords = np.array([[g.x, g.y] for g in proj.geometry])
    names  = [str(n).strip() for n in gdf["name"].tolist()]

    uf   = UnionFind(len(gdf))
    tree = cKDTree(coords)

    # Merge any pair within 100 m (catches Tamil/English pairs and any
    # remaining same-station duplicates)
    MERGE_M = 100
    for i, j in tree.query_pairs(r=MERGE_M):
        uf.union(i, j)

    groups = uf.groups()
    log(f"  Groups after 100 m merge: {len(groups)}")

    # For each group pick the representative closest to the centroid,
    # preferring English-named members.
    kept_indices = []
    for group in groups:
        group_coords = [coords[k] for k in group]
        cx, cy = centroid_of_group(group_coords)

        english_members = [k for k in group if not is_tamil(names[k])]
        candidates = english_members if english_members else group

        best = min(
            candidates,
            key=lambda k: (coords[k][0] - cx) ** 2 + (coords[k][1] - cy) ** 2,
        )
        kept_indices.append(best)

    kept_indices.sort()
    cleaned = gdf.iloc[kept_indices].copy().reset_index(drop=True)
    after = len(cleaned)

    # ── Step 4: Print final list; flag if still above 55 ─────────────────
    log(f"\n  Final station count  : {after}")
    log(f"\n  Station list:")
    for i, row in cleaned[["name"]].iterrows():
        log(f"    {i+1:>3}. {row['name']}")

    if after > 55:
        log(f"\n  WARNING: {after} stations is above the expected max of 55.")
        log(f"  Manual review recommended for the list above.")

    # Save
    out_path = os.path.join(DATA_DIR, "chennai-metro-clean.geojson")
    cleaned.to_file(out_path, driver="GeoJSON")
    log(f"\n  Saved → {out_path}")
    return raw, after

# ── MRTS cleaner ──────────────────────────────────────────────────────────────

def clean_mrts():
    log("")
    log("=" * 60)
    log("  MRTS — CLEANING")
    log("=" * 60)

    path = os.path.join(DATA_DIR, "chennai-mrts-stops.kml")
    gdf  = gpd.read_file(path, driver="KML")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    before = len(gdf)
    log(f"\n  Raw features   : {before}")

    # ── Step 1: Remove stops outside Chennai bbox ──────────────────────────
    from shapely.geometry import box as shapely_box
    bbox_poly = shapely_box(*BBOX)
    in_bbox   = gdf[gdf.geometry.within(bbox_poly)].copy()
    removed_bbox = before - len(in_bbox)
    log(f"\n  [1] Removed outside bbox [{BBOX}]: {removed_bbox}")
    log(f"      Remaining: {len(in_bbox)}")

    # ── Step 2: Remove unnamed stops ──────────────────────────────────────
    name_col = "Name"
    valid_name = (
        in_bbox[name_col].notna()
        & (in_bbox[name_col].astype(str).str.strip() != "")
        & (in_bbox[name_col].astype(str).str.lower() != "nan")
        & (in_bbox[name_col].astype(str).str.lower() != "none")
    )
    named    = in_bbox[valid_name].copy()
    removed_unnamed = len(in_bbox) - len(named)
    log(f"\n  [2] Removed unnamed stops: {removed_unnamed}")
    log(f"      Remaining: {len(named)}")

    # ── Step 3: Remove exact duplicate geometries (deduplicate by name) ───
    # All duplicate pairs in MRTS are 0.0m apart (identical coords).
    # drop_duplicates on name (case-insensitive) keeps the first occurrence.
    named["_name_lower"] = named[name_col].str.strip().str.lower()
    deduped = named.drop_duplicates(subset="_name_lower").drop(columns="_name_lower")
    deduped = deduped.reset_index(drop=True)
    removed_dupes = len(named) - len(deduped)
    log(f"\n  [3] Removed exact duplicate geometries: {removed_dupes}")
    log(f"      Remaining: {len(deduped)}")

    after = len(deduped)
    log(f"\n  Station list after cleaning:")
    for _, row in deduped[[name_col]].iterrows():
        log(f"    {row[name_col]}")

    # Save
    out_path = os.path.join(DATA_DIR, "chennai-mrts-clean.geojson")
    deduped.to_file(out_path, driver="GeoJSON")
    log(f"\n  Saved → {out_path}")
    return before, after

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  CHENNAI TRANSIT — DATASET CLEANING")
    log("=" * 60)

    metro_before, metro_after = clean_metro()
    mrts_before,  mrts_after  = clean_mrts()

    # Bus stops: no cleaning needed
    bus_path = os.path.join(DATA_DIR, "chennai-bus-stops.geojson")
    bus_gdf  = gpd.read_file(bus_path)
    bus_total = len(bus_gdf)

    log("")
    log("=" * 60)
    log("  FINAL SUMMARY")
    log("=" * 60)
    log(f"  Bus stops : {bus_total} stops   (kept as-is)")
    log(f"  Metro     : {metro_before} stations before  →  {metro_after} stations after")
    log(f"  MRTS      : {mrts_before} stops before      →  {mrts_after} stops after")
    log("")
    total = bus_total + metro_after + mrts_after
    log(f"  Combined total ready for full analysis: {total} stops")
    log(f"    ({bus_total} bus + {metro_after} metro + {mrts_after} MRTS)")

    save_log()

if __name__ == "__main__":
    main()
