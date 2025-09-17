from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import statsmodels.api as sm
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="District Disease Forecast API")

# ✅ CORS: allow frontend + API domain
origins = [
    "http://localhost:3000",  # local dev
    "https://water-bourne-disease-prediction-3.onrender.com",  # your backend domain
    "https://your-frontend-domain.com",  # when you deploy frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load district coordinates
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

# Request schema
class InputData(BaseModel):
    district: str
    steps: int
    target: str  # disease column to forecast (e.g., diarrhea_cases)

# Helpers
def prepare_ts(district: str) -> pd.DataFrame:
    df = pd.read_csv("data.csv")
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df[df["district"] == district]
    if df.empty:
        raise ValueError(f"No data found for district '{district}'")
    return df.set_index('date').sort_index()

def train_sarimax(df, target_col, order=(1,1,1), seasonal=(0,0,0,0)):
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float).squeeze()
    model = sm.tsa.statespace.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def forecast(model, last_date, steps=30):
    pred = model.get_forecast(steps=steps)
    forecast_df = pred.summary_frame()
    forecast_df['date'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    return forecast_df

# ✅ Predict endpoint
@app.post("/predict")
def predict_cases(data: InputData):
    if data.district not in district_coords:
        return {"error": f"District '{data.district}' is not supported."}
    try:
        ts = prepare_ts(data.district)
        if data.target not in ts.columns:
            return {"error": f"Target column '{data.target}' not found in data."}

        model = train_sarimax(ts, target_col=data.target)
        last_date = ts.index[-1]
        forecast_df = forecast(model, last_date, steps=data.steps)

        response = []
        for i in range(data.steps):
            response.append({
                "district": data.district,
                "target": data.target,
                "date": forecast_df['date'].iloc[i].strftime("%Y-%m-%d"),
                "mean": float(forecast_df['mean'].iloc[i]),
                "mean_ci_lower": float(forecast_df['mean_ci_lower'].iloc[i]),
                "mean_ci_upper": float(forecast_df['mean_ci_upper'].iloc[i]),
                "coords": district_coords[data.district]
            })

        return {"district": data.district, "target": data.target, "forecast": response}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
