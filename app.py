import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import datetime
from pathlib import Path

st.set_page_config(
    page_title="FloodGuard | Flood Analysis & Disaster Resource Allocation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 1. LOAD MODEL PIPELINE & DATASET
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "final_flood_dataset.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_model_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None

pipeline = load_model_pipeline()
df = load_dataset()

# Calculate dynamic boat threshold based on dataset if available, fallback to 58.84
BOAT_WATER_LEVEL_THRESHOLD = float(df["recorded_water_level_m"].quantile(0.75)) if df is not None else 58.84

# -----------------------------------------------------------------------------
# RESOURCE ALLOCATION HELPER FUNCTION
# -----------------------------------------------------------------------------
def display_resource_allocation(affected_population, water_level, predicted_risk, prefix_key=""):
    st.subheader("📦 Disaster Resource Inventory & Sufficiency Analysis")
    
    required_food = affected_population * 2 
    required_water = affected_population * 5 
    required_doctors = int(np.ceil(affected_population / 500)) if affected_population > 0 else 0
    required_nurses = int(np.ceil(affected_population / 100)) if affected_population > 0 else 0

    # Project Rule: Boats only for High/Very High risk AND water level >= 75th percentile (58.84m)
    if predicted_risk in ["High", "Very High"] and water_level >= BOAT_WATER_LEVEL_THRESHOLD:
        marooned = int(affected_population * 0.40)
        required_boats = int(np.ceil(marooned / 25)) # 25 capacity per boat
        boat_note = f"⚠️ Critical inundation ({water_level}m >= {BOAT_WATER_LEVEL_THRESHOLD:.2f}m). Full nautical deployment required."
    else:
        required_boats = 0
        if predicted_risk not in ["High", "Very High"]:
            boat_note = f"Risk is '{predicted_risk}'. Boat mobilization is unnecessary."
        else:
            boat_note = f"Water level ({water_level}m) is below the critical boating threshold ({BOAT_WATER_LEVEL_THRESHOLD:.2f}m). Road networks likely operable."

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown("**Enter Available Stock**")
        avail_food = st.number_input("Food Packets", min_value=0, value=int(required_food * 0.8), step=500, key=f"{prefix_key}_food")
        avail_water = st.number_input("Drinking Water (L)", min_value=0, value=int(required_water * 1.05), step=1000, key=f"{prefix_key}_water")
        avail_boats = st.number_input("Rescue Boats", min_value=0, value=max(0, required_boats - 2), step=2, key=f"{prefix_key}_boats")
        avail_doctors = st.number_input("Doctors", min_value=0, value=max(0, required_doctors - 1), step=2, key=f"{prefix_key}_docs")
        avail_nurses = st.number_input("Nurses", min_value=0, value=int(required_nurses * 0.9), step=5, key=f"{prefix_key}_nurses")

    with res_col2:
        st.markdown("**Allocation Assessment**")
        inventory_data = [
            {"Resource": "Food Packets (2/person)", "Required": required_food, "Available": avail_food, "Deficit": avail_food - required_food, "Status": "✅ Sufficient" if avail_food >= required_food else "⚠️ Not Sufficient"},
            {"Resource": "Drinking Water (5 L/person)", "Required": required_water, "Available": avail_water, "Deficit": avail_water - required_water, "Status": "✅ Sufficient" if avail_water >= required_water else "⚠️ Not Sufficient"},
            {"Resource": "Rescue Boats", "Required": required_boats, "Available": avail_boats, "Deficit": avail_boats - required_boats, "Status": "✅ Sufficient" if avail_boats >= required_boats else "⚠️ Not Sufficient"},
            {"Resource": "Doctors (1/500 affected)", "Required": required_doctors, "Available": avail_doctors, "Deficit": avail_doctors - required_doctors, "Status": "✅ Sufficient" if avail_doctors >= required_doctors else "⚠️ Not Sufficient"},
            {"Resource": "Nurses (1/100 affected)", "Required": required_nurses, "Available": avail_nurses, "Deficit": avail_nurses - required_nurses, "Status": "✅ Sufficient" if avail_nurses >= required_nurses else "⚠️ Not Sufficient"},
        ]
        st.dataframe(pd.DataFrame(inventory_data), hide_index=True, use_container_width=True)
        st.info(f"**Nautical Deployment Note**: {boat_note}")

# -----------------------------------------------------------------------------
# 2. APPLICATION NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🌊 FloodGuard")
st.sidebar.caption("Flood Analysis & Disaster Resource Allocation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["About", "Historical Analysis", "Prediction"])

if df is None or pipeline is None:
    st.error("Missing `model.pkl` or `final_flood_dataset.csv`. Please verify files are in the same directory.")
    st.stop()

risk_colors = {
    "Low": "rgba(40, 167, 69, 0.15)", "Moderate": "rgba(255, 193, 7, 0.2)",
    "High": "rgba(253, 126, 20, 0.2)", "Very High": "rgba(220, 53, 69, 0.25)",
}
border_colors = {
    "Low": "#28a745", "Moderate": "#ffc107",
    "High": "#fd7e14", "Very High": "#dc3545",
}
area_list = sorted(df["area"].unique().tolist())

# -----------------------------------------------------------------------------
# PAGE 1: ABOUT
# -----------------------------------------------------------------------------
if page == "About":
    st.title("📊 About the Project")
    st.subheader("Flood Analysis and Disaster Resource Allocation")
    
    st.write(
        "The main objective of this project is to analyze rainfall and water-level conditions, "
        "classify flood risk, and use the predicted risk and affected population to support practical disaster-resource allocation."
    )
    
    st.markdown("### 🎯 Project Motto")
    st.markdown("> *Predict the flood risk early, estimate the people affected, and allocate critical resources where they are actually needed.*")

    st.markdown("### ⚙️ What the system does")
    st.markdown("""
    * **Selects** an area and date and accepts 24h rainfall, water level, terrain type, nearest station and total population inputs for prediction.
    * **Retrieves/Processes** the corresponding rainfall, water-level, and geographical attributes.
    * **Classifies** flood risk using the trained Decision Tree model.
    * **Estimates** the affected population for resource planning.
    * **Calculates** food, drinking-water, boat, doctor, and nurse requirements.
    * **Compares** required resources with available resources and reports Sufficiency or Shortages.
    """)

    st.markdown("### 📚 Dataset Overview")
    # Adjusted column layout so the Period string is not cut off
    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Study Areas", f"{df['area'].nunique()}")
    col3.metric("Study Period", f"{df['date'].min():%Y} - {df['date'].max():%Y}")
    st.caption("The dataset contains rainfall, water level, geographical/terrain attributes, population, flood-risk score/category and affected population.")

    st.markdown("### 🤖 Models Selected")
    st.write("Three classification models were evaluated during the project. The accuracy comparison below:")
    
    st.markdown("""
    | Model | Holdout Accuracy |
    |---|---|
    | Logistic Regression | 89.05% |
    | **Decision Tree (Final)** | **96.67%** |
    | Random Forest | 95.50% |
    """)
    st.info("**Decision Tree holdout accuracy: 96.67%**\n\n*The **Decision Tree** was selected as the final model based on the accuracy.*")
    
    st.markdown("### ⚙️ Feature Processing")
    st.markdown("""
    * **Numerical features**: rainfall, 3-day rainfall, 7-day rainfall, water level, elevation, distance to station, month and day of year.
    * **Numerical preprocessing**: `StandardScaler`.
    * **Categorical preprocessing**: `One-Hot Encoding`.
    * **Target encoding**: `LabelEncoder`.
    * No feature-selection step was used.
    """)

    st.markdown("### 🧮 Flood-Risk Target & 🚑 Allocation Rules")
    st.write("The flood-risk categories in this project are analytical categories created from a weighted score based on rainfall and water-level scores. They are not official government flood-warning thresholds.")
    
    st.markdown("""
    | Resource | Project planning rule |
    |---|---|
    | **Food** | 2 packets per affected person |
    | **Drinking water** | 5 liters per affected person |
    | **Boats** | Required only for High/Very High risk AND water level at/above 75th percentile |
    | **Doctors** | 1 doctor per 500 affected people |
    | **Nurses** | 1 nurse per 100 affected people |
    """)
    st.caption(f"*These ratios are project assumptions for demonstration, not official emergency standards. Dataset-specific boat activation level used by this app: **{BOAT_WATER_LEVEL_THRESHOLD:.2f} m** (75th percentile of recorded water levels).*")

    st.markdown("## 📈 Project Charts")

    # Add equal space between the tab buttons and every chart.
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Risk Distribution",
        "Rainfall by Year",
        "Rainfall by Area",
        "Affected Population",
        "Risk by Season"
    ])

    with tab1:
        risk_counts = df["flood_risk"].value_counts().reindex(
            ["Low", "Moderate", "High", "Very High"], fill_value=0
        )
        st.bar_chart(risk_counts)

    with tab2:
        yearly = df.groupby("year")["rainfall_mm"].mean().round(2)
        st.line_chart(yearly)

    with tab3:
        area_rain = (
            df.groupby("area")["rainfall_mm"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        st.bar_chart(area_rain)

    with tab4:
        area_affected = (
            df.groupby("area")["affected_population"]
            .mean()
            .sort_values(ascending=False)
            .round(0)
        )
        st.bar_chart(area_affected)

    with tab5:
        risk_season = pd.crosstab(
            df["season"],
            df["flood_risk"]
        )

        chart_data = (
            risk_season
            .reset_index()
            .melt(
                id_vars="season",
                var_name="Flood Risk",
                value_name="Number of Records"
            )
        )

        st.vega_lite_chart(
            chart_data,
            {
                "mark": {
                    "type": "bar"
                },
                "encoding": {
                    "x": {
                        "field": "season",
                        "type": "nominal",
                        "axis": {
                            "labelAngle": 0
                        }
                    },
                    "xOffset": {
                        "field": "Flood Risk"
                    },
                    "y": {
                        "field": "Number of Records",
                        "type": "quantitative"
                    },
                    "color": {
                        "field": "Flood Risk",
                        "type": "nominal"
                    }
                },
                "width": "container",
                "height": 450,
                "config": {
                    "view": {
                        "stroke": None
                    }
                }
            },
            use_container_width=True
        )

    st.markdown("### ⚠️ Important Limitations")
    st.warning("""
    * The rainfall source is historical/reanalysis-style data, not a direct rain-gauge feed.
    * Water-level coverage depends on the available telemetry records and station matching used during dataset construction.
    * Population and affected-population values are project/synthetic assumptions where applicable.
    * The model predicts the project's analytical flood-risk labels; it does not replace official flood warnings.
    * For a real emergency system, live telemetry, official warning thresholds, verified population data, medical-capacity data and disaster-management rules should be integrated.
    """)

# -----------------------------------------------------------------------------
# PAGE 2: HISTORICAL ANALYSIS
# -----------------------------------------------------------------------------
elif page == "Historical Analysis":
    st.title("📊 Historical Analysis & Resource Assessment")

    col_top1, col_top2 = st.columns([1, 1])
    with col_top1:
        selected_area = st.selectbox("Select Target Area / Municipality", area_list)
    with col_top2:
        default_date = df['date'].max().date() 
        selected_date = st.date_input("Lookup Date", value=default_date)

    area_df = df[df["area"] == selected_area].sort_values("date").copy()
    area_df["date_only"] = area_df["date"].dt.date
    matched_row = area_df[area_df["date_only"] == selected_date]

    if matched_row.empty:
        doy = pd.to_datetime(selected_date).dayofyear
        area_df["doy_diff"] = (area_df["day_of_year"] - doy).abs()
        matched_row = area_df.sort_values("doy_diff").iloc[[0]]
        st.info(f"Exact date record absent. Retrieved historical proxy profile for day index {doy}.")
    else:
        matched_row = matched_row.iloc[[0]]

    curr = matched_row.iloc[0]

    st.markdown("""
    <style>
        [data-testid="stMetricLabel"] {
            font-size: 15px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.subheader("Area & Hydrological Conditions")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("24h Rainfall", f"{curr.get('rainfall_mm', 0):.1f} mm")
    m2.metric("Water Level", f"{curr.get('recorded_water_level_m', 0):.2f} m")
    m3.metric("Terrain Type", str(curr.get('terrain_type', 'N/A')))
    m4.metric("Nearest Station", str(curr.get('nearest_station', 'N/A')))
    m5.metric("Total Population", f"{int(curr.get('population', 0)):,}")

    num_feats = pipeline["numerical_features"]
    cat_feats = pipeline["categorical_features"]

    X_num = matched_row[num_feats]
    X_num_scaled = pipeline["scaler"].transform(X_num)
    X_cat = matched_row[cat_feats]
    X_cat_encoded = pipeline["encoder"].transform(X_cat)
    X_input = np.hstack([X_num_scaled, X_cat_encoded])

    prediction_idx = pipeline["model"].predict(X_input)[0]
    predicted_risk = pipeline["label_encoder"].inverse_transform([prediction_idx])[0]

    affected_population = int(curr.get("affected_population", curr.get("population", 0) * 0.15))

    st.markdown(
        f"""
        <div style="background-color:{risk_colors.get(predicted_risk, '#eee')};
                    border:2px solid {border_colors.get(predicted_risk, '#ccc')};
                    padding:16px; border-radius:10px; margin: 15px 0px;">
            <h3 style="margin:0; color:white;">Predicted Flood Risk: <b>{predicted_risk.upper()}</b></h3>
            <p style="margin:5px 0 0 0; font-size:16px; color:white;">
                Estimated Displacement / Affected Population: <b>{affected_population:,}</b>
            </p>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown("---")

    display_resource_allocation(affected_population, float(curr.get("recorded_water_level_m", 0)), predicted_risk, prefix_key="hist")

# -----------------------------------------------------------------------------
# PAGE 3: PREDICTION (CUSTOM INPUTS)
# -----------------------------------------------------------------------------
elif page == "Prediction":
    st.title("🔮 Custom Flood Prediction & Allocation")
    st.markdown("Input real-time telemetry to forecast flood severity and automatically calculate resource requirements.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pred_area = st.selectbox("Select Target Area", area_list)
            pred_date = st.date_input("Forecast Date", value=datetime.date.today())
        with col2:
            pred_rainfall = st.number_input("24h Rainfall (mm)", min_value=0.0, value=15.0, step=5.0)
            pred_water_level = st.number_input("Current Water Level (m)", min_value=0.0, value=40.0, step=0.5)
        
        submit_btn = st.form_submit_button("Predict Flood Risk")

    if submit_btn:
        area_base = df[df["area"] == pred_area].iloc[0]
        
        month = pred_date.month
        doy = pred_date.timetuple().tm_yday
        if month in [6, 7, 8, 9]: season = 'Monsoon'
        elif month in [10, 11]: season = 'Post-Monsoon'
        elif month in [12, 1, 2]: season = 'Winter'
        else: season = 'Summer'

        input_data = pd.DataFrame([{
            "rainfall_mm": pred_rainfall,
            "rainfall_3day_mm": pred_rainfall * 1.5, 
            "rainfall_7day_mm": pred_rainfall * 2.2, 
            "recorded_water_level_m": pred_water_level,
            "elevation_m": area_base["elevation_m"],
            "distance_to_station_km": area_base["distance_to_station_km"],
            "month": month,
            "day_of_year": doy,
            "season": season,
            "terrain_type": area_base["terrain_type"],
            "landform": area_base["landform"],
            "area_type": area_base["area_type"],
            "coastal_influence": area_base["coastal_influence"],
            "godavari_influence": area_base["godavari_influence"],
            "vegetation_landscape": area_base["vegetation_landscape"],
            "topographic_class": area_base["topographic_class"]
        }])

        X_num = input_data[pipeline["numerical_features"]]
        X_num_scaled = pipeline["scaler"].transform(X_num)
        
        X_cat = input_data[pipeline["categorical_features"]]
        X_cat_encoded = pipeline["encoder"].transform(X_cat)
        
        X_input = np.hstack([X_num_scaled, X_cat_encoded])

        prediction_idx = pipeline["model"].predict(X_input)[0]
        predicted_risk = pipeline["label_encoder"].inverse_transform([prediction_idx])[0]

        severity_ratios = {"Low": 0.05, "Moderate": 0.15, "High": 0.35, "Very High": 0.60}
        total_pop = area_base["population"]
        affected_population = int(total_pop * severity_ratios.get(predicted_risk, 0.15))

        st.markdown(
            f"""
            <div style="background-color:{risk_colors.get(predicted_risk, '#eee')};
                        border:2px solid {border_colors.get(predicted_risk, '#ccc')};
                        padding:16px; border-radius:10px; margin: 15px 0px;">
                <h3 style="margin:0; color:white;">Predicted Flood Risk: <b>{predicted_risk.upper()}</b></h3>
                <p style="margin:5px 0 0 0; font-size:16px;">
                    Estimated Displacement / Affected Population: <b>{affected_population:,}</b> 
                    <span style="font-size:14px; color:#555;">(out of {int(total_pop):,} total residents)</span>
                </p>
            </div>
            """, unsafe_allow_html=True,
        )
        st.markdown("---")

        display_resource_allocation(affected_population, pred_water_level, predicted_risk, prefix_key="pred")