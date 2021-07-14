from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.wkt import loads

sns.set(font = "Helvetica Neue", style = "white")

# file locations
results_path = "~/Downloads/avgIFR.csv"
try: 
    pwd = Path(__file__).parent 
except NameError:
    pwd = Path.cwd()

# titles, labels, etc.
title = "mean IFR by study location"

# load IFR data and join against GADM geometries
ifr = pd.read_csv(results_path)\
    .set_index("location_id")\
    .drop(columns = ["country", "location", 'location_id.1', 'country.1', 'location.1'])
gdf = pd.read_csv(pwd / "IFR_geometries.csv")\
    .set_index("location_id")\
    .assign(geometry = lambda _:_["geometry"].apply(loads))\
    .drop(columns = ["Unnamed: 0"])\
    .pipe(gpd.GeoDataFrame)\
    .join(ifr)\
    .assign(log_ifr = lambda _: np.log(_["Mean"]))

# set colorscheme for choropleth
sm = mpl.cm.ScalarMappable(
    norm = mpl.colors.Normalize(gdf.log_ifr.min(), gdf.log_ifr.max()), 
    cmap = "YlOrRd" # see https://matplotlib.org/stable/tutorials/colors/colormaps.html for other colorschemes
)

# set up plotting
fig, ax = plt.subplots()
plt.gca().axis("off")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "2.5%", pad = 0.5)

# plot background map (sans Antarctica)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world[world.continent != "Antarctica"].plot(
    ax = ax, 
    facecolor = "dimgrey", 
    edgecolor = "white", 
    linewidth = 0.70, 
    zorder = -100, 
    alpha = 0.5
)

# plot log(IFR) for all locations to ensure the colorbar captures the full range of values
gdf.plot(
    ax = ax, 
    column = "log_ifr", 
    legend = True, 
    legend_kwds = {'label': "log(IFR)", 'orientation': "vertical"}, 
    cax = cax, 
    cmap = sm.cmap,
    zorder = 0,
    linewidth = 0
)

# plot white border for non-point locations
gdf[~gdf.display_as_pt].plot(
    ax = ax, 
    facecolor = "none", 
    edgecolor = "white", 
    linewidth = 0.5
)

# plot point locations with exaggerated markers
gdf[gdf.display_as_pt].representative_point().plot(
    ax = ax, 
    color = [sm.to_rgba(_) for _  in gdf[gdf.display_as_pt].log_ifr], 
    marker = 'o', 
    markersize = 50, 
    linewidth = 0.5, 
    edgecolor = "white"
)

# show map
plt.suptitle(f"\n{title}")
plt.subplots_adjust(left = 0)
plt.show()
