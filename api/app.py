from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import json
import pandas as pd
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="API for predicting Airbnb rental prices in Mexico City")

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model components at startup
MODEL_DIR = Path("model_export")
model = joblib.load(MODEL_DIR / "airbnb_prices_predictor_model.joblib")
scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")

with open(MODEL_DIR / "model_metadata.json", "r") as f:
    metadata = json.load(f)

print("Model loaded successfully")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.post("/predict")
async def predict(
    accommodates: int,
    bedrooms: float,
    beds: float,
    bathrooms: float,
    neighbourhood: str = "Unknown",
    has_kitchen: int = 0,
    has_wifi: int = 0,
    has_hot_water: int = 0,
    has_hangers: int = 0,
    has_microwave: int = 0,
    has_hair_dryer: int = 0,
    has_shampoo: int = 0,
    has_tv: int = 0,
    has_smoke_alarm: int = 0,
):
    """Predict Airbnb listing price"""
    try:
        # Get neighbourhood encoding
        neighbourhood_code = metadata["neighbourhood_mapping"].get(neighbourhood, 0)
        
        # Count amenities
        amenities_count = sum([has_kitchen, has_wifi, has_hot_water, has_hangers, has_microwave, has_hair_dryer, has_shampoo, has_tv, has_smoke_alarm])
        
        # Build feature set
        features = {
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "beds": beds,
            "bathrooms": bathrooms,
            "neighbourhood_encoded": neighbourhood_code,
            "amenities_count": amenities_count + 15,
            "minimum_nights": 2,
            "maximum_nights": 365,
            "has_kitchen": has_kitchen,
            "has_wifi": has_wifi,
            "has_hot_water": has_hot_water,
            "has_hangers": has_hangers,
            "has_microwave": has_microwave,
            "has_hair_dryer": has_hair_dryer,
            "has_shampoo": has_shampoo,
            "has_tv": has_tv,
            "has_smoke_alarm": has_smoke_alarm,
        }
        
        # Fill remaining features with 0
        for feature in metadata["feature_names"]:
            if feature not in features:
                features[feature] = 0
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([features])[metadata["feature_names"]]
        
        # Scale numeric features
        input_scaled = input_df.copy()
        if metadata["numeric_features_scaled"]:
            input_scaled[metadata["numeric_features_scaled"]] = scaler.transform(input_df[metadata["numeric_features_scaled"]])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return {"predicted_price": round(float(prediction), 2)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)