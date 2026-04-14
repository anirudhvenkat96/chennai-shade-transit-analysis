import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from shapely.geometry import Point
import osmnx as ox
import rasterstats
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────

print("Loading transit data...")
bus_stops = gpd.read_file('data/chennai-bus-stops.geojson')
metro = gpd.read_file('data/chennai-metro-staions.geojson')
mrts = gpd.read_file('data/chennai-mrts-stops.kml')

# ── Helper to extract name field ───────────────────────────────────────────

def extract_name(gdf, fallback_prefix):
    """Try common name fields, fall back to coordinates"""
    if 'name' in gdf.columns:
        return gdf['name'].fillna(
            gdf.geometry.apply(
                lambda g: f"{fallback_prefix}_{g.y:.4f}_{g.x:.4f}"
            )
        )
    elif 'Name' in gdf.columns:
        return gdf['Name'].fillna(
            gdf.geometry.apply(
                lambda g: f"{fallback_prefix}_{g.y:.4f}_{g.x:.4f}"
            )
        )
    else:
        return gdf.geometry.apply(
            lambda g: f"{fallback_prefix}_{g.y:.4f}_{g.x:.4f}"
        )

# ── Standardise and combine all transit stops ──────────────────────────────

bus_stops['transit_type'] = 'bus'
bus_stops['stop_name'] = extract_name(bus_stops, 'bus')

metro['transit_type'] = 'metro'
metro['stop_name'] = extract_name(metro, 'metro')

mrts['transit_type'] = 'mrts'
mrts['stop_name'] = extract_name(mrts, 'mrts')

# Keep only relevant columns
bus_stops  = bus_stops[['geometry', 'transit_type', 'stop_name']]
metro      = metro[['geometry', 'transit_type', 'stop_name']]
mrts       = mrts[['geometry', 'transit_type', 'stop_name']]

# Align CRS
bus_stops  = bus_stops.to_crs('EPSG:4326')
metro      = metro.to_crs('EPSG:4326')
mrts       = mrts.to_crs('EPSG:4326')

# Combine
all_stops = gpd.GeoDataFrame(
    pd.concat([bus_stops, metro, mrts], ignore_index=True),
    crs='EPSG:4326'
)

# Clip to Chennai bounding box
all_stops = all_stops.cx[79.9:80.4, 12.8:13.3]

# Add unique ID
all_stops = all_stops.reset_index(drop=True)
all_stops['stop_id'] = all_stops.index

print(f"Combined transit stops: {len(all_stops)}")
print(f"  Bus:   {len(all_stops[all_stops.transit_type == 'bus'])}")
print(f"  Metro: {len(all_stops[all_stops.transit_type == 'metro'])}")
print(f"  MRTS:  {len(all_stops[all_stops.transit_type == 'mrts'])}")

# ── Test on ONE stop first ─────────────────────────────────────────────────

# Find metro or bus stop closest to Anna Nagar, Chennai
anna_nagar = Point(80.2093, 13.0850)
metro_bus = all_stops[all_stops.transit_type.isin(['metro', 'bus'])].copy()
metro_bus_proj = metro_bus.to_crs('EPSG:32644')
anna_nagar_proj = gpd.GeoSeries([anna_nagar], crs='EPSG:4326').to_crs('EPSG:32644')[0]
metro_bus_proj['dist_to_anna_nagar'] = metro_bus_proj.geometry.distance(anna_nagar_proj)
closest_idx = metro_bus_proj['dist_to_anna_nagar'].idxmin()
test_stop = all_stops.loc[closest_idx]
test_point = test_stop.geometry

print(f"\nTest stop:")
print(f"  ID:          {test_stop.stop_id}")
print(f"  Name:        {test_stop.stop_name}")
print(f"  Type:        {test_stop.transit_type}")
print(f"  Coordinates: {test_point.x:.4f}, {test_point.y:.4f}")

# Fetch walkable street network within 600m
print("\nFetching street network (20-30 seconds)...")
G = ox.graph_from_point(
    (test_point.y, test_point.x),
    dist=600,
    network_type='walk'
)

edges = ox.graph_to_gdfs(G, nodes=False)
print(f"Street segments found: {len(edges)}")

# Buffer streets by 8m to capture overhead canopy
edges_proj     = edges.to_crs('EPSG:32644')
edges_buffered = edges_proj.buffer(8)
edges_buffered_wgs = edges_buffered.to_crs('EPSG:4326')

# Sample NDVI along buffered street segments
print("Sampling NDVI along streets...")
stats = rasterstats.zonal_stats(
    edges_buffered_wgs,
    'data/chennai_ndvi_2023.tif',
    stats=['mean'],
    nodata=-9999
)

# Calculate shade score
ndvi_values = [s['mean'] for s in stats if s['mean'] is not None]
shaded      = sum(1 for v in ndvi_values if v > 0.3)
shade_score = (shaded / len(ndvi_values)) * 100 if ndvi_values else 0

print(f"\nResults for '{test_stop.stop_name}':")
print(f"  Shade score:      {shade_score:.1f}%")
print(f"  Shaded segments:  {shaded} of {len(ndvi_values)}")
print(f"  Mean NDVI:        {np.mean(ndvi_values):.3f}")
print("\nTest complete — ready to run full analysis!")