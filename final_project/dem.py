import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open('output_hh.tif') as src:
    # 1. Inspect Profile & Metadata
    print("=== Core Profile ===")
    for key, value in src.profile.items():
        print(f"{key}: {value}")

    # 2. Read and inspect actual elevation data (Band 1)
    elevation = src.read(1)
    
    # Filter out NoData values if defined
    if src.nodata is not None:
        valid_mask = (elevation != src.nodata) & (~np.isnan(elevation))
    else:
        valid_mask = ~np.isnan(elevation)

    valid_elevation = elevation[valid_mask]

    print("\n=== Elevation Information ===")
    print(f"Raster Grid Shape: {elevation.shape} (Height: {src.height}px, Width: {src.width}px)")
    print(f"Total Pixels: {elevation.size:,}")
    print(f"Valid Elevation Pixels: {valid_elevation.size:,}")
    print(f"Min Elevation: {valid_elevation.min():.2f} m")
    print(f"Max Elevation: {valid_elevation.max():.2f} m")
    print(f"Mean Elevation: {valid_elevation.mean():.2f} m")
    print(f"Median Elevation: {np.median(valid_elevation):.2f} m")
    print(f"Std Deviation: {valid_elevation.std():.2f} m")
    print("\nSample Elevation Values (top-left 5x5 grid):\n", elevation[:5, :5])

    # 3. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    image_hidden = ax.imshow(elevation, cmap='terrain')

    # Overlay using rasterio's geo-referenced show
    show((src, 1), ax=ax, cmap='terrain',
         title="Digital Elevation Model (DEM) - Ground Level (m)")

    # Add colorbar legend
    cbar = fig.colorbar(image_hidden, ax=ax,
                        orientation='vertical', fraction=0.035, pad=0.04)
    cbar.set_label('Elevation above sea level (m)', rotation=90)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("dem_visualization.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved successfully as 'dem_visualization.png'!")
