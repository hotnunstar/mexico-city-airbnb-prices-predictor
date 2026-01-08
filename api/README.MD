# Airbnb Price Predictor API

Machine Learning model for predicting Airbnb rental prices in Mexico City.

## Prerequisites

⚠️ **Important**: Before running the application, the Jupyter notebook must be executed to train and export the model:

1. Make sure the api folder is in the same directory as `Airbnb_Prices_Predictor.ipynb`
2. Open the Jupyter notebook and run all cells
3. Required model files will be generated and saved in the correct directory
4. Only then proceed with running the application

## Running the Application

### Option 1: Using Python (No Docker)

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the application:

    ```bash
    python app.py
    ```

3. Open browser at: http://localhost:8000

### Option 2: Using Docker

1. Build the Docker image:

    ```bash
    docker build -t airbnb-api .
    ```

2. Run the container:

    ```bash
    docker run -p 8000:8000 airbnb-api
    ```

3. Open browser at: http://localhost:8000

## API Endpoints

- `GET /` - Web interface
- `POST /predict` - Prediction endpoint

## Testing

1. Open http://localhost:8000
2. Fill in the property details
3. Select amenities
4. Click "Predict Price"
5. View the predicted nightly price in USD

## Model Information

- **Algorithm**: Random Forest Regressor (Tuned)
- **Features**: 100+ features including property characteristics, location, and amenities
- **Performance**: See model training notebook for detailed metrics
- **Target**: Nightly rental price in USD
