# choropleth generation

The choropleth generation workflow takes in results from the Bayesian model and uses administrative boundary delineations from the [Database of Global Administrative Areas](https://gadm.org/index.html) to display the results visually.

## install dependencies

a. create a virtual environment and activate it

```
python3 -mvenv venv 
source ./venv/bin/activate
```

b. install libraries 

```
pip3 install -r requirements.txt
```

## run workflow 

a. unzip the `IFR_geometries.csv.zip` file in the same directory as the `choropleth.py` file.

b. in `choropleth.py`, update the `results_path` variable to the path of the model results CSV, and then run `python3 choropleth.py` 


# update geometries (optional)
if we need to add locations (e.g. for sero-only studies), we'll need to download additional delineations and wire them up as show in `assemble_gdf.py`.
