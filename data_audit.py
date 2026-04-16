"""
Data Quality Audit — Chennai Transit Datasets + NDVI Raster
Outputs: outputs/data_audit.txt + printed summary
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
from scipy.spatial import cKDTree

# ── Config ──────────────────────────────────────────────────────────────────
BBOX = (79.9, 12.8, 80.4, 13.3)   # minx, miny, maxx, maxy
BBOX_POLY = box(*BBOX)
DUPLICATE_DIST_M = 50              # metres

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
AUDIT_PATH = os.path.join(OUT_DIR, "data_audit.txt")

DATASETS = {
    "bus_stops": os.path.join(DATA_DIR, "chennai-bus-stops.geojson"),
    "metro":     os.path.join(DATA_DIR, "chennai-metro-staions.geojson"),
    "mrts":      os.path.join(DATA_DIR, "chennai-mrts-stops.kml"),
}
NDVI_PATH = os.path.join(DATA_DIR, "chennai_ndvi_2023.tif")

EXPECTED_COUNTS = {
    "metro": (40, 55),   # ~45 stations
    "mrts":  (15, 30),   # ~20 stations
}

lines = []   # accumulates the full report

def log(s=""):
    print(s)
    lines.append(s)

def section(title):
    log()
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)

# ── Helper: find near-duplicates (same name, within 50 m) ───────────────────
def find_near_duplicates(gdf, dist_m=50):
    """Return list of (idx_a, idx_b, dist_m, name) for suspicious pairs."""
    if gdf.crs is None or gdf.crs.is_geographic:
        proj = gdf.to_crs(epsg=32644)   # UTM 44N covers Chennai
    else:
        proj = gdf.copy()

    coords = np.array([[g.x, g.y] for g in proj.geometry])
    tree   = cKDTree(coords)
    pairs  = tree.query_pairs(r=dist_m)

    dupes = []
    name_col = _name_col(gdf)
    for i, j in pairs:
        if name_col:
            ni = str(gdf.iloc[i][name_col]).strip().lower()
            nj = str(gdf.iloc[j][name_col]).strip().lower()
            if ni and nj and ni != "nan" and ni == nj:
                d = coords[i] - coords[j]
                dist = float(np.linalg.norm(d))
                dupes.append((gdf.index[i], gdf.index[j], round(dist, 1), gdf.iloc[i][name_col]))
    return dupes

def _name_col(gdf):
    for c in ["name", "Name", "NAME", "stop_name", "station_name"]:
        if c in gdf.columns:
            return c
    return None

def _name_series(gdf):
    c = _name_col(gdf)
    if c:
        return gdf[c].astype(str)
    return None

# ── Audit each transit dataset ───────────────────────────────────────────────
def audit_transit(label, path):
    section(f"TRANSIT DATASET: {label.upper()}")

    # ── Load ────────────────────────────────────────────────────────────────
    if not os.path.exists(path):
        log(f"  ERROR: file not found — {path}")
        return

    if path.endswith(".kml"):
        import fiona
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(path, driver="KML")
    else:
        gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    log(f"  CRS            : {gdf.crs}")
    log(f"  Geometry types : {gdf.geom_type.value_counts().to_dict()}")

    # Drop Z coords if present (simplifies bbox check)
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: g if g is None else (
            type(g)([c[:2] for c in g.coords]) if hasattr(g, "coords") else g
        )
    )

    # ── 1. Total features ───────────────────────────────────────────────────
    total = len(gdf)
    log(f"\n  [1] Total features    : {total}")

    # Expected count flag
    if label in EXPECTED_COUNTS:
        lo, hi = EXPECTED_COUNTS[label]
        flag = "OK" if lo <= total <= hi else f"WARNING — expected {lo}–{hi}"
        log(f"      Expected range    : {lo}–{hi}  →  {flag}")

    # ── 2. Named vs unnamed ─────────────────────────────────────────────────
    ns = _name_series(gdf)
    if ns is not None:
        valid_names = ns[(ns.str.strip() != "") & (ns.str.lower() != "nan") & (ns.str.lower() != "none")]
        named   = len(valid_names)
        unnamed = total - named
        log(f"\n  [2] Named features    : {named}")
        log(f"      Unnamed / blank   : {unnamed}")
        if unnamed:
            log(f"      Unnamed indices   : {list(gdf[~gdf.index.isin(valid_names.index)].index[:10])}")
    else:
        log(f"\n  [2] No recognisable name column found. Columns: {list(gdf.columns)}")

    # ── 3. Duplicate geometries ─────────────────────────────────────────────
    dupes = find_near_duplicates(gdf, dist_m=DUPLICATE_DIST_M)
    log(f"\n  [3] Near-duplicate pairs (same name, <{DUPLICATE_DIST_M} m): {len(dupes)}")
    if dupes:
        for a, b, d, name in dupes[:10]:
            log(f"      idx {a} & {b}  dist={d} m  name='{name}'")
        if len(dupes) > 10:
            log(f"      ... and {len(dupes)-10} more")

    # ── 4. Outside Chennai bbox ─────────────────────────────────────────────
    in_bbox  = gdf[gdf.geometry.within(BBOX_POLY)]
    out_bbox = gdf[~gdf.geometry.within(BBOX_POLY)]
    log(f"\n  [4] Within Chennai bbox {BBOX}  : {len(in_bbox)}")
    log(f"      Outside bbox              : {len(out_bbox)}")
    if len(out_bbox):
        nc = _name_col(gdf)
        for idx, row in out_bbox.head(5).iterrows():
            nm = row[nc] if nc else "—"
            g  = row.geometry
            log(f"      idx {idx}  name='{nm}'  coords=({g.x:.4f}, {g.y:.4f})")

    # ── 5. Sample 5 rows ────────────────────────────────────────────────────
    log(f"\n  [5] Sample (5 rows):")
    sample_cols = [c for c in gdf.columns if c != "geometry"][:6]
    sample = gdf[sample_cols + ["geometry"]].head(5)
    for idx, row in sample.iterrows():
        g = row.geometry
        coord_str = f"({g.x:.4f}, {g.y:.4f})" if hasattr(g, "x") else str(g)
        vals = "  |  ".join(f"{c}={str(row[c])[:30]}" for c in sample_cols)
        log(f"      [{idx}] {coord_str}  {vals}")

    return {
        "label": label, "total": total,
        "named": named if ns is not None else None,
        "unnamed": unnamed if ns is not None else None,
        "dupes": len(dupes),
        "outside_bbox": len(out_bbox),
    }

# ── Audit NDVI raster ────────────────────────────────────────────────────────
def audit_ndvi(path):
    section("NDVI RASTER: chennai_ndvi_2023.tif")

    if not os.path.exists(path):
        log(f"  ERROR: file not found — {path}")
        return

    with rasterio.open(path) as src:
        log(f"  CRS            : {src.crs}")
        log(f"  Driver         : {src.driver}")
        log(f"  Dimensions     : {src.width} x {src.height} px")
        log(f"  Band count     : {src.count}")
        log(f"  Dtype          : {src.dtypes[0]}")
        log(f"  NoData value   : {src.nodata}")

        bounds = src.bounds
        log(f"\n  [1] Raster extent:")
        log(f"      left={bounds.left:.4f}  right={bounds.right:.4f}")
        log(f"      bottom={bounds.bottom:.4f}  top={bounds.top:.4f}")

        # Compare to Chennai bbox
        minx, miny, maxx, maxy = BBOX
        covers_x = bounds.left <= minx and bounds.right >= maxx
        covers_y = bounds.bottom <= miny and bounds.top >= maxy
        log(f"\n  [2] Covers Chennai bbox {BBOX}:")
        log(f"      X (lon): {'YES' if covers_x else 'PARTIAL/NO'}  "
            f"(raster {bounds.left:.4f}–{bounds.right:.4f}, need {minx}–{maxx})")
        log(f"      Y (lat): {'YES' if covers_y else 'PARTIAL/NO'}  "
            f"(raster {bounds.bottom:.4f}–{bounds.top:.4f}, need {miny}–{maxy})")

        # Read band 1
        data = src.read(1).astype(float)
        nodata = src.nodata

        if nodata is not None:
            mask = data == nodata
        else:
            mask = np.zeros_like(data, dtype=bool)

        valid = data[~mask]
        nodata_pct = 100.0 * mask.sum() / data.size

        log(f"\n  [3] NDVI statistics (band 1, valid pixels only):")
        log(f"      NoData pixels : {mask.sum():,} / {data.size:,}  ({nodata_pct:.2f}%)")
        log(f"      Min NDVI      : {valid.min():.4f}")
        log(f"      Max NDVI      : {valid.max():.4f}")
        log(f"      Mean NDVI     : {valid.mean():.4f}")
        log(f"      Std dev       : {valid.std():.4f}")

        # Anomaly check
        anomalous_low  = (valid < -1.0).sum()
        anomalous_high = (valid >  1.0).sum()
        log(f"\n  [4] Anomalous values:")
        log(f"      Values < -1.0 : {anomalous_low}")
        log(f"      Values >  1.0 : {anomalous_high}")
        if anomalous_low + anomalous_high == 0:
            log(f"      All values within valid NDVI range [-1, 1]  OK")
        else:
            log(f"      WARNING — anomalous values detected, may need masking")

        # Distribution buckets
        log(f"\n  [5] NDVI distribution:")
        buckets = [(-1.0, 0.0, "water/bare"),
                   ( 0.0, 0.2, "sparse/urban"),
                   ( 0.2, 0.4, "low vegetation"),
                   ( 0.4, 0.6, "moderate vegetation"),
                   ( 0.6, 1.0, "dense vegetation")]
        for lo, hi, label in buckets:
            n = ((valid >= lo) & (valid < hi)).sum()
            pct = 100.0 * n / len(valid)
            log(f"      [{lo:+.1f}, {hi:+.1f})  {label:<22} : {n:>8,}  ({pct:.1f}%)")

    return {
        "covers_bbox": covers_x and covers_y,
        "nodata_pct": nodata_pct,
        "ndvi_min": float(valid.min()),
        "ndvi_max": float(valid.max()),
        "ndvi_mean": float(valid.mean()),
        "anomalous": anomalous_low + anomalous_high,
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("  CHENNAI TRANSIT + NDVI — DATA QUALITY AUDIT")
    log(f"  Chennai bounding box: {BBOX}")
    log("=" * 60)

    results = {}
    for label, path in DATASETS.items():
        r = audit_transit(label, path)
        if r:
            results[label] = r

    ndvi = audit_ndvi(NDVI_PATH)

    # ── Final summary ────────────────────────────────────────────────────────
    section("SUMMARY")
    header = f"  {'Dataset':<15} {'Total':>7} {'Named':>7} {'Unnamed':>9} {'Dupes':>7} {'OutBbox':>9}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for label, r in results.items():
        named_s   = str(r["named"])   if r["named"]   is not None else "—"
        unnamed_s = str(r["unnamed"]) if r["unnamed"] is not None else "—"
        log(f"  {label:<15} {r['total']:>7} {named_s:>7} {unnamed_s:>9} "
            f"{r['dupes']:>7} {r['outside_bbox']:>9}")

    if ndvi:
        log(f"\n  NDVI raster:")
        log(f"    Covers Chennai bbox : {'YES' if ndvi['covers_bbox'] else 'PARTIAL/NO'}")
        log(f"    NoData              : {ndvi['nodata_pct']:.2f}%")
        log(f"    NDVI range          : {ndvi['ndvi_min']:.3f} – {ndvi['ndvi_max']:.3f}  "
            f"(mean {ndvi['ndvi_mean']:.3f})")
        log(f"    Anomalous values    : {ndvi['anomalous']}")

    # ── Write report ────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n  Report saved → {AUDIT_PATH}")

if __name__ == "__main__":
    main()
