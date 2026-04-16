import geopandas as gpd, warnings, fiona, sys
sys.stdout = open(r"C:\Users\aniru\chennai-shade-transit-analysis\outputs\_probe_out.txt", "w", encoding="utf-8")
warnings.filterwarnings("ignore")
fiona.drvsupport.supported_drivers["KML"] = "rw"

metro = gpd.read_file(r"C:\Users\aniru\chennai-shade-transit-analysis\data\chennai-metro-staions.geojson")
print("METRO columns:", list(metro.columns))
print("\nMETRO name / alt_name (first 30):")
for _, r in metro[["name","alt_name"]].head(30).iterrows():
    print(" ", repr(r["name"]), "|", repr(r["alt_name"]))

# check how many have non-null alt_name
notnull = metro["alt_name"].notna() & (metro["alt_name"].astype(str).str.strip() != "") & (metro["alt_name"].astype(str).str.lower() != "nan")
print(f"\nnon-null alt_name: {notnull.sum()} / {len(metro)}")
print("\nalt_name sample where present:")
for _, r in metro[notnull][["name","alt_name"]].head(20).iterrows():
    print(" ", repr(r["name"]), "->", repr(r["alt_name"]))

mrts = gpd.read_file(r"C:\Users\aniru\chennai-shade-transit-analysis\data\chennai-mrts-stops.kml", driver="KML")
print("\nMRTS columns:", list(mrts.columns))
print("MRTS Name sample (first 20):")
for _, r in mrts[["Name"]].head(20).iterrows():
    print(" ", repr(r["Name"]))
print(f"\nMRTS total: {len(mrts)}")
print("Index 117 (unnamed):", mrts.iloc[117]["Name"] if len(mrts) > 117 else "N/A")
