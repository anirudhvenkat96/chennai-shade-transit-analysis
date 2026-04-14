import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterstats
import osmnx as ox
from shapely.geometry import mapping

warnings.filterwarnings('ignore')

os.makedirs('outputs/networks', exist_ok=True)

# ── Load transit data (same logic as proof_of_concept.py) ─────────────────
print("Loading transit data...")

def extract_name(gdf, prefix):
    for col in ("name", "Name"):
        if col in gdf.columns:
            return gdf[col].fillna(
                gdf.geometry.apply(lambda g: f"{prefix}_{g.y:.4f}_{g.x:.4f}")
            )
    return gdf.geometry.apply(lambda g: f"{prefix}_{g.y:.4f}_{g.x:.4f}")

def is_ascii(s):
    try:
        s.encode('ascii')
        return True
    except (UnicodeEncodeError, AttributeError):
        return False

bus_stops = gpd.read_file('data/chennai-bus-stops.geojson')
metro     = gpd.read_file('data/chennai-metro-staions.geojson')
mrts      = gpd.read_file('data/chennai-mrts-stops.kml')

for gdf, ttype, prefix in [(bus_stops, 'bus', 'bus'), (metro, 'metro', 'metro'), (mrts, 'mrts', 'mrts')]:
    gdf['transit_type'] = ttype
    gdf['stop_name']    = extract_name(gdf, prefix)

bus_stops = bus_stops[['geometry', 'transit_type', 'stop_name']].to_crs('EPSG:4326')
metro     = metro    [['geometry', 'transit_type', 'stop_name']].to_crs('EPSG:4326')
mrts      = mrts     [['geometry', 'transit_type', 'stop_name']].to_crs('EPSG:4326')

all_stops = gpd.GeoDataFrame(
    pd.concat([bus_stops, metro, mrts], ignore_index=True), crs='EPSG:4326'
).cx[79.9:80.4, 12.8:13.3].reset_index(drop=True)
all_stops['stop_id'] = all_stops.index

metro_en = all_stops[(all_stops.transit_type == 'metro') & all_stops.stop_name.apply(is_ascii)]
mrts_en  = all_stops[(all_stops.transit_type == 'mrts')  & all_stops.stop_name.apply(is_ascii)]
bus_en   = all_stops[(all_stops.transit_type == 'bus')   & all_stops.stop_name.apply(is_ascii)]

selected = pd.concat([
    metro_en.iloc[:4],
    mrts_en.iloc[:3],
    bus_en.iloc[:3],
]).reset_index(drop=True)

print(f"Processing {len(selected)} stations:\n")

# ── Process each station ───────────────────────────────────────────────────
station_records = []
network_files_created = []
total = len(selected)

for i, (_, stop) in enumerate(selected.iterrows(), start=1):
    stop_id   = int(stop.stop_id)
    name      = stop.stop_name
    ttype     = stop.transit_type
    pt        = stop.geometry
    net_fname = f"{stop_id}_{ttype}.geojson"
    net_path  = f"outputs/networks/{net_fname}"

    print(f"[{i}/{total}] {name} ({ttype})  id={stop_id}")

    try:
        G     = ox.graph_from_point((pt.y, pt.x), dist=600, network_type='walk')
        edges = ox.graph_to_gdfs(G, nodes=False)

        edges_proj     = edges.to_crs('EPSG:32644')
        edges_buffered = edges_proj.buffer(8).to_crs('EPSG:4326')

        stats = rasterstats.zonal_stats(
            edges_buffered,
            'data/chennai_ndvi_2023.tif',
            stats=['mean'],
            nodata=-9999
        )

        ndvi_values = [s['mean'] if s['mean'] is not None else 0.0 for s in stats]
        shaded      = sum(1 for v in ndvi_values if v > 0.3)
        seg_total   = len(ndvi_values)
        shade_score = round((shaded / seg_total * 100) if seg_total else 0.0, 1)
        mean_ndvi   = round(float(np.mean(ndvi_values)) if ndvi_values else 0.0, 4)

        edges = edges.copy().reset_index(drop=True)
        edges['mean_ndvi']       = [round(v, 4) for v in ndvi_values]
        edges['is_shaded']       = [v > 0.3 for v in ndvi_values]
        edges['segment_colour']  = ['#2d7a3a' if v > 0.3 else '#cc3300' for v in ndvi_values]

        # ── Write network GeoJSON ──────────────────────────────────────────
        features = []
        for _, seg in edges.iterrows():
            try:
                geom = mapping(seg.geometry)
            except Exception:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "segment_colour": seg['segment_colour'],
                    "mean_ndvi":      seg['mean_ndvi'],
                    "is_shaded":      bool(seg['is_shaded']),
                }
            })

        with open(net_path, 'w', encoding='utf-8') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f,
                      separators=(',', ':'), ensure_ascii=False)

        network_files_created.append(net_fname)
        print(f"  shade={shade_score:.1f}%  segments={seg_total}  → {net_path}")

    except Exception as e:
        print(f"  WARNING: Failed — {e}")
        shade_score = None
        mean_ndvi   = None
        net_fname   = None

    station_records.append({
        "stop_id":      stop_id,
        "stop_name":    name,
        "transit_type": ttype,
        "shade_score":  shade_score,
        "mean_ndvi":    mean_ndvi,
        "network_file": net_fname,
        "longitude":    round(pt.x, 6),
        "latitude":     round(pt.y, 6),
    })

    time.sleep(1)

# ── Write stations.geojson ─────────────────────────────────────────────────
station_features = []
for r in station_records:
    station_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [r['longitude'], r['latitude']]
        },
        "properties": {
            "stop_id":      r['stop_id'],
            "stop_name":    r['stop_name'],
            "transit_type": r['transit_type'],
            "shade_score":  r['shade_score'],
            "mean_ndvi":    r['mean_ndvi'],
            "network_file": r['network_file'],
            "longitude":    r['longitude'],
            "latitude":     r['latitude'],
        }
    })

stations_path = 'outputs/stations.geojson'
with open(stations_path, 'w', encoding='utf-8') as f:
    json.dump({"type": "FeatureCollection", "features": station_features}, f,
              indent=2, ensure_ascii=False)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("NETWORK FILES CREATED")
print("-"*72)
for fname in network_files_created:
    path = f"outputs/networks/{fname}"
    size = os.path.getsize(path)
    print(f"  {fname:<35}  {size/1024:6.1f} KB")

print(f"\n  {len(network_files_created)}/10 network files written to outputs/networks/")

print("\n" + "="*72)
print("stations.geojson SUMMARY")
print("-"*72)
print(f"  {'ID':<6} {'NAME':<32} {'TYPE':<8} {'SHADE':>7}  {'NDVI':>6}  NETWORK FILE")
print(f"  {'-'*6} {'-'*32} {'-'*8} {'-'*7}  {'-'*6}  {'-'*30}")
for r in station_records:
    shade_str = f"{r['shade_score']:.1f}%" if r['shade_score'] is not None else "N/A"
    ndvi_str  = f"{r['mean_ndvi']:.4f}"    if r['mean_ndvi']   is not None else "N/A"
    nf        = r['network_file'] or "—"
    print(f"  {r['stop_id']:<6} {r['stop_name']:<32} {r['transit_type']:<8} "
          f"{shade_str:>7}  {ndvi_str:>6}  {nf}")

print("="*72)
print(f"\nDone. outputs/stations.geojson written ({len(station_features)} features).")
