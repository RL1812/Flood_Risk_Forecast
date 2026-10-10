import os
import geopandas as gpd
from rasterstats import zonal_stats

# Data files live in final_project/, one level above helpers/
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Load GeoJSON
gdf = gpd.read_file(os.path.join(PROJECT_DIR, "MasterPlan2019PlanningAreaBoundaryNoSea.geojson"))
gdf_wgs84 = gdf.to_crs(epsg=4326)

# 2. Extract zonal statistics directly from DEM
stats = zonal_stats(
    gdf_wgs84,
    os.path.join(PROJECT_DIR, "output_hh.tif"),
    stats=["mean", "min", "std"],
    nodata=-9999
)

# 3. Append features as DataFrame columns
gdf_wgs84["elev_mean"] = [
    round(s["mean"], 2) if s["mean"] is not None else 0.0 for s in stats]
gdf_wgs84["elev_min"] = [
    round(s["min"], 2) if s["min"] is not None else 0.0 for s in stats]
gdf_wgs84["elev_std"] = [
    round(s["std"], 2) if s["std"] is not None else 0.0 for s in stats]

# 4. Save to an enriched GeoJSON/SQLite file
gdf_wgs84.to_file(os.path.join(PROJECT_DIR, "enriched_planning_areas.geojson"), driver="GeoJSON")
print("Topographic stats successfully merged into boundaries!")
