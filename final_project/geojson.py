import geopandas as gpd
import folium

# 1. Read the GeoJSON file downloaded from SLA
gdf = gpd.read_file("MasterPlan2019PlanningAreaBoundaryNoSea.geojson")

# 2. Check the initial coordinate reference system (CRS)
print("Original CRS:", gdf.crs)

# 3. Reproject to WGS84 for web mapping
gdf_wgs84 = gdf.to_crs(epsg=4326)

# 4. Preview the first few rows and column names
print(gdf_wgs84.head())

# 1. Initialize a Folium map centered on Singapore
singapore_map = folium.Map(
    location=[1.3521, 103.8198],
    zoom_start=11,
    tiles="CartoDB positron"
)

# 2. Add the subzones to the map
# (Replace 'SUBZONE_N' with the actual column name for the subzone name from gdf.head())
folium.GeoJson(
    gdf_wgs84,
    name="SLA Subzones",
    style_function=lambda feature: {
        "fillColor": "#3388ff",
        "color": "#000000",
        "weight": 1,
        "fillOpacity": 0.2,
    },
    tooltip=folium.GeoJsonTooltip(
        # Displays the first attribute column on hover
        fields=[gdf_wgs84.columns[1]],
        aliases=["Name:"]
    )
).add_to(singapore_map)

# 3. Save to an HTML file to open in your browser, or display directly in a notebook
singapore_map.save("subzones_map.html")
print("Map exported to subzones_map.html!")
