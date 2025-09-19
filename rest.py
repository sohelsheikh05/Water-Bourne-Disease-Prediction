from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import statsmodels.api as sm
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

app = FastAPI(title="District Disease Forecast API")

# ✅ CORS: allow frontend + API domain
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://sih-25001-dashboard.vercel.app"  # deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # use ["*"] during dev if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# District coordinates
# -----------------------------
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

# -----------------------------
# Request schemas
# -----------------------------
class InputData(BaseModel):
    district: str
    steps: int
    target: str  # disease column to forecast

class StateInputData(BaseModel):
    state: str
    target: str
    date: str  # YYYY-MM-DD

class StateDiseaseRequest(BaseModel):
    state: str
    disease: str

class TotalCasesRequest(BaseModel):
    state: str
    disease: str
    date: str

# -----------------------------
# State → District mapping (FIXED)
# -----------------------------
state_districts = {
    "Meghalaya": ["West Jaintia Hills", "East Jaintia Hills", "East Khasi Hills", "Ri Bhoi", "South West Khasi Hills"],
    "Manipur": ["Imphal West", "Bishnupur", "Imphal East", "Thoubal", "Kakching", "Churachandpur"]
}

# -----------------------------
# Helpers
# -----------------------------
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

# -----------------------------
# Endpoints
# -----------------------------

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

@app.post("/state_district_cases")
def get_forecast_table(req: StateDiseaseRequest):
    try:
        state = req.state
        disease = req.disease

        if state not in state_districts:
            return {"error": f"State '{state}' not found."}

        df = pd.read_csv("forecast_results.csv")
        api_call_date = datetime.today().strftime("%Y-%m-%d")

        df_state = df[
            df["district"].isin(state_districts[state]) &
            (df["target"] == disease)
        ]

        df_state = df_state.groupby(["district", "target"], as_index=False).agg({
            "mean": "sum",
            "mean_se": "mean",
            "mean_ci_lower": "min",
            "mean_ci_upper": "max"
        })

        result = []
        for _, row in df_state.iterrows():
            coords = district_coords.get(row["district"], [26.2006, 92.9376])
            result.append({
                "date": api_call_date,
                "cases": row["mean"],
                "district": row["district"],
                "target": row["target"],
                "lat": coords[0],
                "lng": coords[1]
            })

        return {"state": state, "disease": disease, "forecast": result}

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_state")
def predict_state(data: StateInputData):
    if data.state not in state_districts:
        return {"error": f"State '{data.state}' is not supported."}

    try:
        prediction_date = datetime.strptime(data.date, "%Y-%m-%d")
        all_predictions = []
        districts = []

        for district in state_districts[data.state]:
            try:
                ts = prepare_ts(district)
                if data.target not in ts.columns:
                    continue

                model = train_sarimax(ts, target_col=data.target)
                last_date = ts.index[-1]
                steps_needed = (prediction_date - last_date).days

                if steps_needed <= 0:
                    all_predictions.append({
                        "district": district,
                        "error": f"Requested date {data.date} is not after last available data {last_date.date()}"
                    })
                    continue

                forecast_df = forecast(model, last_date, steps=steps_needed)
                row = forecast_df.iloc[steps_needed - 1]

                districts.append({
                    "district": district,
                    "coords": district_coords[district],
                    "mean": row["mean"],
                })
                all_predictions.append({
                    "district": district,
                    "target": data.target,
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "cases": float(row["mean"]),
                    "mean_ci_lower": float(row["mean_ci_lower"]),
                    "mean_ci_upper": float(row["mean_ci_upper"]),
                    "coords": district_coords[district],
                })

            except Exception as e:
                all_predictions.append({"district": district, "error": str(e)})

        return {
            "state": data.state,
            "target": data.target,
            "date": data.date,
            "districts": districts,
            "predictions": all_predictions
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/diseases")
def get_disease_names():
    try:
        df = pd.read_csv("forecast_results.csv")
        unique_diseases = sorted(set(df["target"]))
        return {"diseases": unique_diseases}
    except Exception as e:
        return {"error": str(e)}

@app.get("/states")
def get_states():
    return {"states": list(state_districts.keys())}

@app.post("/total_cases_by_state")
def total_cases_by_state(data: TotalCasesRequest):
    try:
        if data.state not in state_districts:
            return {"error": f"State '{data.state}' not found."}
        forecast_df = pd.read_csv("forecast_results1.csv")

        df_state = forecast_df[
            forecast_df["district"].isin(state_districts[data.state]) &
            (forecast_df["target"] == data.disease) &
            (forecast_df["date"] == data.date)
        ]

        total_cases = df_state["mean"].sum()

        return {
            "state": data.state,
            "disease": data.disease,
            "total_cases": total_cases
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Run server (Render requires PORT)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
