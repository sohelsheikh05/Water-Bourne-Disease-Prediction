import joblib
import pandas as pd
from pymongo import MongoClient, errors

# -------------------------------
# 1. Connect to MongoDB
# -------------------------------
client = MongoClient("mongodb+srv://bookskart245:y1oW4ps1InkJCudg@cluster0.tbyfa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# create DB and collections (MongoDB creates them lazily on first insert)
db = client["forecast_db"]
input_collection = db["raw_data"]
output_collection = db["forecast_results"]

# -------------------------------
# 2. Ensure indexes (optional but good practice)
# -------------------------------
# For quick lookup
input_collection.create_index("timestamp")
output_collection.create_index("raw_id")

# -------------------------------
# 3. Load trained model
# -------------------------------
try:
    model = joblib.load("sarimax_model.pkl")
except FileNotFoundError:
    raise RuntimeError("❌ Trained model file 'model.pkl' not found. Train & save your model first.")

# -------------------------------
# 4. Start watching for new inserts
# -------------------------------
try:
    with input_collection.watch() as stream:
        print("👀 Watching MongoDB for new data...")
        for change in stream:
            if change["operationType"] == "insert":
                new_doc = change["fullDocument"]

                # Convert to DataFrame for prediction (exclude _id, keep only features)
                df = pd.DataFrame([new_doc])
                if "_id" in df.columns:
                    df = df.drop("_id", axis=1)

                # Run prediction
                prediction = model.predict(df)[0]

                # Save forecast result
                output_collection.insert_one({
                    "raw_id": new_doc["_id"],   # reference back to input
                    "prediction": float(prediction)
                })

                print(f"✅ Prediction saved for input {_id}")
except errors.PyMongoError as e:
    print(f"MongoDB error: {e}")
