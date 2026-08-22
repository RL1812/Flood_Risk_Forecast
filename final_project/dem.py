import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
with rasterio.open('output_hh.tif') as src:
    #print(src.crs)
    print("=== Core Profile ===")
    for key, value in src.profile.items():
        print(f"{key}: {value}")
    fig, ax = plt.subplots(figsize=(10, 8))
    # 2. Render Band 1 using a terrain colormap
    # (src, 1) specifies the first band and applies the spatial coordinates automatically
    image_hidden = ax.imshow(src.read(1), cmap='terrain')

    # Overlay using rasterio's geo-referenced show
    show((src, 1), ax=ax, cmap='terrain',
         title="Digital Elevation Model (DEM) - Ground Level (m)")

    # 3. Add a colorbar legend
    cbar = fig.colorbar(image_hidden, ax=ax,
                        orientation='vertical', fraction=0.035, pad=0.04)
    cbar.set_label('Elevation above sea level (m)', rotation=90)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("dem_visualization.png", dpi=300, bbox_inches="tight")
    print("Plot saved successfully!")
