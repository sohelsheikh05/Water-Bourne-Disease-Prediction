import pandas as pd
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium
import statsmodels.api as sm
import branca.colormap as cm

# --- SARIMAX Model Training ---
def train_sarimax(df, target_col='target', order=(1,1,1), seasonal=(0,0,0,0)):
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float).squeeze()

    model = sm.tsa.statespace.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)
    return fitted_model

# --- Forecasting Function ---
def forecast(model, last_df, steps=30):
    pred = model.get_forecast(steps=steps)
    forecast_df = pred.summary_frame()
    forecast_df['index'] = pd.date_range(
        start=last_df['index'].iloc[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )
    return forecast_df

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Dynamic Forecast Map - Northeast India", layout="wide")
st.title("Dynamic Forecast Map - Northeast India")

# Load historical data
CSV_FILE = "forecast_results.csv"  # CSV must contain 'district', 'index', 'target'

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    st.error(f"CSV file not found: {CSV_FILE}")
    st.stop()

required_cols = {'district', 'index', 'target'}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

df['index'] = pd.to_datetime(df['index'])

# District coordinates
district_coords = {
    "West Jaintia Hills": [25.5463, 91.7300],
    "East Jaintia Hills": [25.6301, 92.8251],
    "Imphal West": [24.8170, 93.9368],
    "Bishnupur": [24.6032, 93.9494],
    "Imphal East": [24.8057, 94.0464],
    "Thoubal": [24.6137, 93.9241],
    "Kakching": [24.5185, 93.9886],
    "Churachandpur": [24.3306, 93.6719],
    "East Khasi Hills": [25.5788, 91.8933],
    "Ri Bhoi": [25.6716, 91.8261],
    "South West Khasi Hills": [25.3143, 91.5631]
}
df['coords'] = df['district'].map(district_coords)

# --- User Inputs ---
selected_date = st.date_input(
    "Select training cutoff date",
    value=df['index'].min(),
    min_value=df['index'].min(),
    max_value=df['index'].max()
)

train_df = df[df['index'] <= pd.to_datetime(selected_date)]
if train_df.empty:
    st.warning("No data available up to selected date.")
    st.stop()

steps = st.slider("Forecast steps (days)", min_value=7, max_value=60, value=30)

# --- Train SARIMAX per district & forecast ---
forecast_list = []

for district in train_df['district'].unique():
    ts = train_df[train_df['district'] == district].sort_values('index')
    
    try:
        model = train_sarimax(ts, target_col='target')
        forecast_df = forecast(model, ts, steps=steps)
        forecast_df['district'] = district
        forecast_df['target'] = 'target'
        forecast_list.append(forecast_df)
    except Exception as e:
        st.warning(f"Failed for {district}: {e}")

if not forecast_list:
    st.error("No forecasts generated.")
    st.stop()

all_forecasts = pd.concat(forecast_list, ignore_index=True)
all_forecasts['coords'] = all_forecasts['district'].map(district_coords)

# --- Save forecast results ---
all_forecasts.to_csv('forecasts.csv', index=False)

st.download_button(
    label="Download Forecast CSV",
    data=all_forecasts.to_csv(index=False),
    file_name='forecast_results.csv',
    mime='text/csv'
)

# --- Display Folium Map ---
m = folium.Map(location=[25.5, 92.5], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

vmin = all_forecasts['mean'].min()
vmax = all_forecasts['mean'].max()
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
colormap.caption = 'Predicted Cases'
colormap.add_to(m)

for _, row in all_forecasts.iterrows():
    lat, lon = row['coords']
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color=colormap(row['mean']),
        fill=True,
        fill_color=colormap(row['mean']),
        fill_opacity=0.7,
        popup=f"<b>District:</b> {row['district']}<br><b>Date:</b> {row['index'].date()}<br><b>Predicted Cases:</b> {row['mean']:.2f}"
    ).add_to(marker_cluster)

st_folium(m, width=800, height=600)
