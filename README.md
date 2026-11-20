# NYC Ride-Hailing Fare Prediction and Explainable Decision Support

**An interactive machine learning prototype for estimating base passenger fares for Uber and Lyft trips in New York City (NYC).**

**Live application:** https://nyc-fare-prediction.streamlit.app/

Developed by Emmanuel Kwame Okyere as part of an MSc Business Analytics (Financial Analytics) research project at the University of Ghana Business School.

## Project overview

Ride-hailing fares depend on trip characteristics such as distance, duration, location, and time. This project compares regression models for passenger fare prediction and integrates the selected LightGBM model with SHAP explanations in an interactive Streamlit decision-support application.

The intended users are pricing analysts and business decision-makers. The application allows them to examine an estimated fare and explore how changing trip characteristics affects model predictions. It is a **research prototype**, not an official Uber or Lyft fare quotation system.

## Data and prediction target

- **Source:** Public NYC Taxi and Limousine Commission (NYC TLC) High-Volume For-Hire Vehicle (HVFHV) trip records.
- **Period:** January-May 2026.
- **Platforms:** Uber and Lyft.
- **Target:** `base_passenger_fare`, measured in USD. This is not the same as the complete amount ultimately paid by a passenger.
- **Predictors used in the application:** trip distance (`trip_miles`), duration (`trip_time`), request month, request day of week, request hour, platform, pickup and drop-off Taxi Zone IDs, shared-ride request, wheelchair-accessible vehicle request, and Access-A-Ride flag.

The project involves data cleaning, exploratory analysis, feature engineering, model comparison, and interpretability. The fitted pipeline applies preprocessing to inputs before passing them to the LightGBM model. In the research modelling workflow, lower-cardinality categorical features are one-hot encoded, pickup and drop-off Taxi Zone IDs are target encoded, and distance and duration are retained as numerical features.

The large raw trip-record dataset and model-training notebooks are **not included** in this repository. The application and its saved inference pipeline are provided here.

## Models and evaluation

Five regression approaches were evaluated on the same held-out test set. Root Mean Squared Error (RMSE) was the main model-selection metric, supported by Mean Absolute Error (MAE) and R-squared (R²).

| Model | Training records | Test RMSE (USD) | Test MAE (USD) | Test R² |
| --- | ---: | ---: | ---: | ---: |
| **LightGBM** | **7,000,000** | **9.80** | **5.25** | **0.8506** |
| Random Forest | 5,000,000 | 9.93 | 5.40 | 0.8467 |
| XGBoost | 5,000,000 | 9.93 | 5.30 | 0.8467 |
| CatBoost | 5,000,000 | 10.05 | 5.37 | 0.8430 |
| Linear Regression | 84,745,747 | 10.61 | 6.13 | 0.8248 |

**LightGBM achieved the lowest test RMSE.** It was selected for the interactive prototype. Training sample sizes differ because of computational constraints; in an additional comparison using 5 million training records per ensemble model, LightGBM also achieved a lower RMSE than Random Forest and XGBoost.

These metrics describe the historical held-out test set. They do not guarantee accuracy for future periods or markets.

## Explainability with SHAP

SHapley Additive exPlanations (SHAP) were used to interpret the selected LightGBM model:

- At the overall model level, trip distance and trip duration were the most influential predictors, followed by pickup and drop-off locations.
- For a single trip scenario, the application presents a SHAP waterfall chart and a table of predictor contributions in USD.
- Contributions are grouped back to the user-facing predictors to make the explanation easier to read.

SHAP shows how model inputs contribute to a prediction relative to a baseline. It does **not** establish that those inputs causally determine a fare.

## Streamlit application

**Step 1 - Enter trip details.** Choose a request date/time (within the study period), Uber or Lyft, pickup/drop-off boroughs and Taxi Zones, distance, duration, and service-request flags.

**Step 2 - Review the result.** The app displays the estimated **base passenger fare in USD**, a trip summary, a SHAP waterfall explanation, and a table of feature contributions. The SHAP chart can be downloaded as a PNG.

**Explore scenarios.** Use **Modify Trip Inputs** to change trip characteristics and generate another estimate. These are manual scenarios, not live platform quotations.

## Repository contents

| File | Purpose |
| --- | --- |
| `fare_prediction_app.py` | Two-page Streamlit interface, inference, and SHAP explanation |
| `lgb_final_model_7m.joblib` | Saved LightGBM inference pipeline, including preprocessing |
| `taxi_zone_lookup_streamlit_final.csv` | Taxi Zone names and supported pickup/drop-off selections |
| `requirements.txt` | Python packages needed by the application |
| `README.md` | Project documentation |

## Run the application locally

Install a compatible Python environment, then run:

```bash
git clone https://github.com/ek-okyere/nyc-fare-prediction.git
cd nyc-fare-prediction
python -m venv .venv
```

Activate the virtual environment for your operating system, then:

```bash
python -m pip install -r requirements.txt
streamlit run fare_prediction_app.py
```

The application loads the serialized model and Taxi Zone lookup from the repository folder. Only load `.joblib` files from sources you trust. The local installation steps are provided for reproducibility; execution may depend on package and Python-version compatibility.

## Scope and limitations

- The model was developed using NYC TLC HVFHV data from January-May 2026, and the application date input is restricted to that period.
- Trip inputs are entered manually; the prototype is not connected to Uber, Lyft, live route services, or live pricing APIs.
- Predictions are estimates of `base_passenger_fare`, not binding quotes or guaranteed final passenger charges.
- Model explanations describe predictive associations, not causal effects.
- This repository contains the demonstration app and saved inference pipeline, rather than the full large-scale training workflow.
- Additional temporal validation, operational monitoring, and further testing would be needed before any production use.

## Technology

Python · pandas · NumPy · scikit-learn · LightGBM · SHAP · Streamlit · Matplotlib · joblib

## Author

**Emmanuel Kwame Okyere**  
MSc Business Analytics (Financial Analytics), University of Ghana Business School  
[LinkedIn](https://www.linkedin.com/in/ekokyere)
