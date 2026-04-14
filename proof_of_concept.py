import subprocess, sys

# ── Fix Windows console encoding ───────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Ensure folium is installed ─────────────────────────────────────────────
try:
    import folium
except ImportError:
    print("Installing folium...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "folium"])
    import folium

import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterstats
import osmnx as ox
from shapely.geometry import Point
import json

warnings.filterwarnings('ignore')

# ── Output directories ─────────────────────────────────────────────────────
os.makedirs('outputs/maps', exist_ok=True)

TRANSIT_COLORS = {
    "metro": "blue",
    "mrts":  "purple",
    "bus":   "orange",
}

# ── Load transit data ──────────────────────────────────────────────────────
print("Loading transit data...")

def extract_name(gdf, prefix):
    for col in ("name", "Name"):
        if col in gdf.columns:
            return gdf[col].fillna(
                gdf.geometry.apply(lambda g: f"{prefix}_{g.y:.4f}_{g.x:.4f}")
            )
    return gdf.geometry.apply(lambda g: f"{prefix}_{g.y:.4f}_{g.x:.4f}")

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

print(f"Loaded {len(all_stops)} stops  "
      f"(bus: {(all_stops.transit_type=='bus').sum()}, "
      f"metro: {(all_stops.transit_type=='metro').sum()}, "
      f"mrts: {(all_stops.transit_type=='mrts').sum()})")

# ── Pick first 10 stops — 4 metro, 3 mrts, 3 bus — English names only ─────
def is_ascii(s):
    try:
        s.encode('ascii')
        return True
    except (UnicodeEncodeError, AttributeError):
        return False

metro_en = all_stops[(all_stops.transit_type == 'metro') & all_stops.stop_name.apply(is_ascii)]
mrts_en  = all_stops[(all_stops.transit_type == 'mrts')  & all_stops.stop_name.apply(is_ascii)]
bus_en   = all_stops[(all_stops.transit_type == 'bus')   & all_stops.stop_name.apply(is_ascii)]

selected = pd.concat([
    metro_en.iloc[:4],
    mrts_en.iloc[:3],
    bus_en.iloc[:3],
]).reset_index(drop=True)
print(f"\nSelected stops:")
for _, row in selected.iterrows():
    print(f"  [{row.stop_id}] {row.stop_name} ({row.transit_type})")

# ── Shade analysis ─────────────────────────────────────────────────────────
def analyse_stop(stop):
    pt = stop.geometry
    G  = ox.graph_from_point((pt.y, pt.x), dist=600, network_type='walk')
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
    total       = len(ndvi_values)
    shade_score = (shaded / total * 100) if total else 0.0
    mean_ndvi   = float(np.mean(ndvi_values)) if ndvi_values else 0.0

    edges = edges.copy()
    edges['ndvi']  = ndvi_values
    edges['color'] = ['#2d7a3a' if v > 0.3 else '#cc3300' for v in ndvi_values]

    return edges, shade_score, mean_ndvi, shaded, total

# ── Build Folium map ───────────────────────────────────────────────────────
def build_map(stop, edges, shade_score):
    pt    = stop.geometry
    name  = stop.stop_name
    ttype = stop.transit_type

    m = folium.Map(location=[pt.y, pt.x], zoom_start=15, tiles='OpenStreetMap')

    for _, row in edges.iterrows():
        try:
            coords = [(c[1], c[0]) for c in row.geometry.coords]
        except Exception:
            continue
        folium.PolyLine(coords, color=row['color'], weight=3, opacity=0.85).add_to(m)

    folium.CircleMarker(
        location=[pt.y, pt.x],
        radius=9,
        color='white',
        weight=2,
        fill=True,
        fill_color=TRANSIT_COLORS.get(ttype, 'gray'),
        fill_opacity=1.0,
        popup=folium.Popup(
            f"<b>{name}</b><br>Type: {ttype}<br>Shade score: {shade_score:.1f}%",
            max_width=200
        )
    ).add_to(m)

    title_html = f"""
    <div style="
        position: fixed; top: 12px; right: 12px; z-index: 1000;
        background: rgba(255,255,255,0.92); padding: 8px 14px;
        border-radius: 6px; border: 1px solid #ccc;
        font-family: Arial, sans-serif; font-size: 13px; line-height: 1.5;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
        <b>{name}</b><br>
        {ttype.upper()} &nbsp;|&nbsp; Shade: <b>{shade_score:.1f}%</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    return m

# ── Main loop ──────────────────────────────────────────────────────────────
results = []
total = len(selected)

for i, (_, stop) in enumerate(selected.iterrows(), start=1):
    name  = stop.stop_name
    ttype = stop.transit_type
    print(f"\n[{i}/{total}] Processing {name} ({ttype})...")
    print(f"  Coords: {stop.geometry.x:.4f}, {stop.geometry.y:.4f}")

    try:
        print(f"  Fetching street network...")
        edges, shade_score, mean_ndvi, shaded, seg_total = analyse_stop(stop)
        print(f"  Shade score: {shade_score:.1f}%  ({shaded}/{seg_total} segments)")
    except Exception as e:
        print(f"  WARNING: Failed — {e}")
        results.append({
            "stop_name": name, "transit_type": ttype,
            "shade_score": None, "mean_ndvi": None,
            "lon": stop.geometry.x, "lat": stop.geometry.y,
        })
        time.sleep(1)
        continue

    m = build_map(stop, edges, shade_score)
    slug     = name.lower().replace(' ', '_').replace('/', '_').replace('.', '')[:40]
    filename = f"outputs/maps/{slug}_{ttype}.html"
    m.save(filename)
    print(f"  Map saved: {filename}")

    results.append({
        "stop_name":    name,
        "transit_type": ttype,
        "shade_score":  round(shade_score, 1),
        "mean_ndvi":    round(mean_ndvi, 3),
        "lon":          stop.geometry.x,
        "lat":          stop.geometry.y,
    })

    time.sleep(1)

# ── Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"{'STATION':<30} {'TYPE':<8} {'SHADE SCORE':>12}")
print("-"*62)

valid   = [r for r in results if r['shade_score'] is not None]
invalid = [r for r in results if r['shade_score'] is None]

for r in sorted(valid, key=lambda x: x['shade_score']):
    print(f"{r['stop_name']:<30} {r['transit_type']:<8} {r['shade_score']:>10.1f}%")
for r in invalid:
    print(f"{r['stop_name']:<30} {r['transit_type']:<8} {'N/A':>12}")

print("="*62)

# ── Save GeoJSON ───────────────────────────────────────────────────────────
features = [
    {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [r['lon'], r['lat']]},
        "properties": {
            "stop_name":    r['stop_name'],
            "transit_type": r['transit_type'],
            "shade_score":  r['shade_score'],
            "mean_ndvi":    r['mean_ndvi'],
        }
    }
    for r in results if r['lon'] is not None
]

with open('outputs/poc_results.geojson', 'w', encoding='utf-8') as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to outputs/poc_results.geojson")
print(f"Maps saved to    outputs/maps/")
