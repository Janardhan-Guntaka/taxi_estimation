import sys
from pathlib import Path
import pytz  # Import for timezone conversion

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import zipfile
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Timezones
UTC_TZ = pytz.utc
NY_TZ = pytz.timezone("America/New_York")  # Eastern Time

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

# LocationID to Borough mapping
location_mapping = {
    1: "EWR", 2: "Queens", 3: "Bronx", 4: "Manhattan", 5: "Staten Island",
    6: "Staten Island", 7: "Queens", 8: "Queens", 9: "Queens", 10: "Queens",
    11: "Brooklyn", 12: "Manhattan", 13: "Manhattan", 14: "Brooklyn", 15: "Queens", 132: "Queens",
    # Add all remaining mappings...
}

location_df = pd.DataFrame(location_mapping.items(), columns=["LocationID", "Borough"])

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
    """Downloads, extracts, and loads a shapefile as a GeoDataFrame."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    if not zip_path.exists():
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file: {e}")

    if not shapefile_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file: {e}")

    return gpd.read_file(shapefile_path).to_crs("epsg:4326")

# Get current time in UTC and convert to EST/EDT
current_time_utc = pd.Timestamp.now(tz=UTC_TZ)
current_time_ny = current_time_utc.astimezone(NY_TZ)

st.title("New York Yellow Taxi Cab Demand Next Hour")
st.header(f'Current NYC Time: {current_time_ny.strftime("%Y-%m-%d %I:%M %p %Z")}')

progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner("Downloading shape file for taxi zones..."):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)

with st.spinner("Fetching batch of inference data..."):
    features = load_batch_of_features_from_store(current_time_utc)
    st.sidebar.write("Inference features fetched")
    progress_bar.progress(2 / N_STEPS)

with st.spinner("Fetching predictions..."):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Model predictions loaded")
    progress_bar.progress(3 / N_STEPS)

shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

# Borough Dropdown
selected_borough = st.selectbox(
    "Select a Borough:", location_df["Borough"].unique()
)

# Get LocationIDs for selected Borough
location_ids = location_df[location_df["Borough"] == selected_borough]["LocationID"].tolist()

# Filter predictions based on selected borough
filtered_predictions = predictions[predictions["pickup_location_id"].isin(location_ids)]

# Display filtered predictions
st.write(f"Showing predictions for: {selected_borough} (LocationID(s): {location_ids})")
st.dataframe(filtered_predictions.sort_values("predicted_demand", ascending=False).head(10))

with st.spinner("Plotting predicted ride demand..."):
    st.subheader("Taxi Ride Predictions Map")
    
    def create_taxi_map(shapefile_path, prediction_data):
        """Create an interactive choropleth map of NYC taxi zones with predicted rides"""
        nyc_zones = gpd.read_file(shapefile_path)
        nyc_zones = nyc_zones.merge(
            prediction_data[["pickup_location_id", "predicted_demand"]],
            left_on="LocationID",
            right_on="pickup_location_id",
            how="left",
        )
        nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
        nyc_zones = nyc_zones.to_crs(epsg=4326)

        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
        colormap = LinearColormap(
            colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
            vmin=nyc_zones["predicted_demand"].min(),
            vmax=nyc_zones["predicted_demand"].max(),
        )
        colormap.add_to(m)

        def style_function(feature):
            predicted_demand = feature["properties"].get("predicted_demand", 0)
            return {"fillColor": colormap(float(predicted_demand)), "color": "black", "weight": 1, "fillOpacity": 0.7}

        folium.GeoJson(
            nyc_zones.to_json(),
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=["zone", "predicted_demand"], aliases=["Zone:", "Predicted Demand:"]),
        ).add_to(m)

        st.session_state.map_obj = m
        st.session_state.map_created = True
        return m

    map_obj = create_taxi_map(shapefile_path, filtered_predictions)
    
    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    st.sidebar.write("Finished plotting taxi rides demand")
    progress_bar.progress(4 / N_STEPS)

# Display statistics
st.subheader("Prediction Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Rides", f"{filtered_predictions['predicted_demand'].mean():.0f}")
with col2:
    st.metric("Maximum Rides", f"{filtered_predictions['predicted_demand'].max():.0f}")
with col3:
    st.metric("Minimum Rides", f"{filtered_predictions['predicted_demand'].min():.0f}")

# Plot Top 10 Locations
top10 = filtered_predictions.sort_values("predicted_demand", ascending=False).head(10)["pickup_location_id"].to_list()
for location_id in top10:
    fig = plot_prediction(
        features=features[features["pickup_location_id"] == location_id],
        prediction=filtered_predictions[filtered_predictions["pickup_location_id"] == location_id],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
