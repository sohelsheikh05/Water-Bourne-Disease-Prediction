from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import statsmodels.api as sm
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from datetime import datetime
app = FastAPI(title="District Disease Forecast API")

# ✅ CORS: allow frontend + API domain
origins = [
    "http://localhost:3000",  # local dev
    "https://water-bourne-disease-prediction-3.onrender.com",  # your backend domain
    "https://your-frontend-domain.com", 
    "http://127.0.0.0:3000"
      # when you deploy frontend
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
class StateInputData(BaseModel):
    state: str
    target: str        # e.g. "diarrhea_cases"
    date: str          # prediction date (YYYY-MM-DD)


# Suppose we map which districts belong to which state
state_districts = {
    "Meghalaya": ["West Jaintia Hills","East Jaintia Hills","Imphal West","Bishnupur","Imphal East","Thoubal","Kakching","Churachandpur"],
    "Manipur": ["East Khasi Hills","Ri Bhoi","South West Khasi Hills"]
}
NORTHEAST_COORDINATES = {
    "Meghalaya": {
        "West Jaintia Hills": {"lat": 25.5463, "lng": 91.7300},
        "East Jaintia Hills": {"lat": 25.6301, "lng": 92.8251},
        "Imphal West": {"lat": 24.8170, "lng": 93.9368},
        "Bishnupur": {"lat": 24.6032, "lng": 93.9494},
        "Imphal East": {"lat": 24.8057, "lng": 94.0464},
        "Thoubal": {"lat": 24.6137, "lng": 93.9241},
        "Kakching": {"lat": 24.5185, "lng": 93.9886},
        "Churachandpur": {"lat": 24.3306, "lng": 93.6719},
        "East Khasi Hills": {"lat": 25.5788, "lng": 91.8933},
        "Ri Bhoi": {"lat": 25.6716, "lng": 91.8261},
        "South West Khasi Hills": {"lat": 25.3143, "lng": 91.5631}
    },
    "Manipur": {
        "East Khasi Hills": {"lat": 25.5788, "lng": 91.8933},
        "Ri Bhoi": {"lat": 25.6716, "lng": 91.8261},
        "South West Khasi Hills": {"lat": 25.3143, "lng": 91.5631}
    }
}

# Request body model
class StateDiseaseRequest(BaseModel):
    state: str
    disease: str

# Endpoint




# Request body schema
class StateDiseaseRequest(BaseModel):
    state: str
    disease: str

# State → districts mapping
state_districts = {
    "Meghalaya": ["West Jaintia Hills","East Jaintia Hills","Imphal West","Bishnupur","Imphal East","Thoubal","Kakching","Churachandpur"],
    "Manipur": ["East Khasi Hills","Ri Bhoi","South West Khasi Hills"]
}

# District coordinates
NORTHEAST_COORDINATES = {
    "West Jaintia Hills": {"lat": 25.5463, "lng": 91.7300},
    "East Jaintia Hills": {"lat": 25.6301, "lng": 92.8251},
    "Imphal West": {"lat": 24.8170, "lng": 93.9368},
    "Bishnupur": {"lat": 24.6032, "lng": 93.9494},
    "Imphal East": {"lat": 24.8057, "lng": 94.0464},
    "Thoubal": {"lat": 24.6137, "lng": 93.9241},
    "Kakching": {"lat": 24.5185, "lng": 93.9886},
    "Churachandpur": {"lat": 24.3306, "lng": 93.6719},
    "East Khasi Hills": {"lat": 25.5788, "lng": 91.8933},
    "Ri Bhoi": {"lat": 25.6716, "lng": 91.8261},
    "South West Khasi Hills": {"lat": 25.3143, "lng": 91.5631}
}

@app.post("/state_district_cases")
def get_forecast_table(req: StateDiseaseRequest):
    try:
        state = req.state
        disease = req.disease

        if state not in state_districts:
            return {"error": f"State '{state}' not found."}
        print("state_districts[state]",state_districts[state])
        # Load forecast.csv
        df = pd.read_csv("forecast_results.csv")
        api_call_date = "2023-12-06"
        # Filter districts for this state
        df_state = df[
            df["district"].isin(state_districts[state]) &(df["target"] == disease) ]

        df_state = df_state.groupby(["district", "target"], as_index=False).agg({
            "mean": "sum",             # sum of cases if needed
            "mean_se": "mean",         # average standard error
            "mean_ci_lower": "min",    # take min lower bound
            "mean_ci_upper": "max"     # take max upper bound
        })

        # Current API call date
        
        print(df_state)
        result = []
        for _, row in df_state.iterrows():
            coords = NORTHEAST_COORDINATES.get(row["district"], {"lat": 26.2006, "lng": 92.9376})
            result.append({
                "date":api_call_date ,
                "cases": row["mean"],
                "district": row["district"],
                "target": row["target"],
                "lat": coords["lat"],
                "lng": coords["lng"],
               
            })

        return {"state": state, "disease": disease, "forecast": result}

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_state")
def predict_state(data: StateInputData):
    print(data)
    if data.state not in state_districts:
        return {"error": f"State '{data.state}' is not supported."}

    try:
        prediction_date = datetime.strptime(data.date, "%Y-%m-%d")
        all_predictions = []
        districts=[]

        for district in state_districts[data.state]:
            try:
               
                ts = prepare_ts(district)
                if data.target not in ts.columns:
                    continue  # skip district if target column not present

                model = train_sarimax(ts, target_col=data.target)
                last_date = ts.index[-1]
               
                steps_needed = (prediction_date - last_date).days

                if steps_needed <= 0:
                    # Requested date is not in the future

                    all_predictions.append({
                        "district": district,
                        "error": f"Requested date {data.date} is not after last available data {last_date.date()}"
                    })
                    continue
                
                forecast_df = forecast(model, last_date, steps=steps_needed)
                if forecast_df.empty or steps_needed > len(forecast_df):
                    all_predictions.append({
                        "district": district,
                        "error": "Forecast not available for the requested date"
                    })
                    continue
                forecast_df.to_csv("forecast_results1.csv", index=True)
                row = forecast_df.iloc[steps_needed - 1]
                  # pick that date
                districts.append({
                    "district":district,
                    "coords": NORTHEAST_COORDINATES[district],
                    "mean": row["mean"],
                })
                all_predictions.append({
                    "district": district,
                    "target": data.target,
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "cases": float(row["mean"]),
                    "mean_ci_lower": float(row["mean_ci_lower"]),
                    "mean_ci_upper": float(row["mean_ci_upper"]),
                    "coords": NORTHEAST_COORDINATES[district],
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
        # Load dataset
        df = pd.read_csv("forecast_results.csv")
        
        # Drop known non-disease columns
        print(df.columns)
        disease_cols = df["target"]
        
        # Ensure uniqueness
        unique_diseases = sorted(set(disease_cols))
        
        return {"diseases": unique_diseases}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/states")
def get_disease_names():
    try:
        # Load dataset
        df = pd.read_csv("forecast_results.csv")
        
        # Drop known non-disease columns
        print(df.columns)
        disease_cols = df["target"]
        
        # Ensure uniqueness
        unique_diseases = sorted(set(disease_cols))
        
        return {"diseases": unique_diseases}
    except Exception as e:
        return {"error": str(e)}

# Load forecast data
# Make sure your CSV has columns: date, mean, district, target


@app.post("/total_cases_by_state")
def total_cases_by_state(data):
    try:
        if data.state not in state_districts:
            return {"error": f"State '{data.state}' not found."}
        forecast_df = pd.read_csv("forecast_results1.csv")
        print(forecast_df.columns)
        
        df_state = forecast_df[
            forecast_df["district"].isin(state_districts[data.state]) &
            (forecast_df["target"] == data.disease) & (forecast_df["date"] == data.date)
        ]

        total_cases = df_state["mean"].sum()

        return {
            "state": data.state,
            "disease": data.disease,    
            "total_cases": total_cases
        }
    except Exception as e:
        return {"error": str(e)}    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
