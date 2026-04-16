"""
One-shot rename pass for chennai-metro-clean.geojson.
Replaces Tamil names and standardises two English names.
"""

import os
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "chennai-metro-clean.geojson")

RENAMES = {
    # Tamil → English
    "சைதாப்பேட்டை":        "Saidapet",
    "எல் ஐ சி":             "LIC Colony",
    "அரசினர் தோட்டம்":     "Government Estate",
    "தேனாம்பேட்டை":        "Teynampet",
    "நந்தனம்":              "Nandanam",
    "சின்னமலை":             "Shenoy Nagar",
    "கிண்டி":               "Guindy",
    "ஏ ஜி டி.எம்.எஸ்":     "AG-DMS",
    "ஆயிரம் விளக்கு":      "Thousand Lights",
    "தியாகராயா கல்லூரி":   "Thyagaraya College",
    "விம்கோ நகர்":          "Wimco Nagar",
    "சுங்கச்சாவடி":         "Customs",
    "புது வண்ணாரப்பேட்டை": "Puduvannarapettai",
    # English standardisation
    "Central Metro":         "Chennai Metro Central",
    "Thiruvottriyur":        "Thiruvottiyur",
}

gdf = gpd.read_file(PATH)
before_names = set(gdf["name"].astype(str))

applied = []
not_found = []

for old, new in RENAMES.items():
    mask = gdf["name"].astype(str).str.strip() == old.strip()
    hits = mask.sum()
    if hits:
        gdf.loc[mask, "name"] = new
        applied.append((old, new, hits))
    else:
        not_found.append(old)

gdf.to_file(PATH, driver="GeoJSON")

# ── Report ────────────────────────────────────────────────────────────────────
print(f"Applied renames ({len(applied)}):")
for old, new, n in applied:
    print(f"  {old}  →  {new}  ({n} row{'s' if n>1 else ''})")

if not_found:
    print(f"\nNot found in file ({len(not_found)}) — may already be correct or absent:")
    for name in not_found:
        print(f"  {name}")

print(f"\nFinal station list ({len(gdf)} stations):")
for i, name in enumerate(sorted(gdf["name"].astype(str).tolist()), 1):
    print(f"  {i:>3}. {name}")
