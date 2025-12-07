# app/streamlit_app.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# ------------------- PATHS -------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_arbitrage_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "profit_model.pkl"


# ------------------- LOADERS -------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ------------------- HELPERS -------------------

CROP_ICONS = {
    "Tomato": "🍅",
    "Onion": "🧅",
    "Potato": "🥔",
    "Brinjal": "🍆",
    "Cabbage": "🥬",
    "Chilli": "🌶️",
    "Garlic": "🧄",
    "Ginger": "🫚",
}

STATE_COORDS = {
    "Maharashtra": (19.7515, 75.7139),
    "Karnataka": (15.3173, 75.7139),
    "Gujarat": (22.2587, 71.1924),
    "Delhi": (28.7041, 77.1025),
    "UP": (26.8467, 80.9462),
    "Rajasthan": (27.0238, 74.2179),
}


def get_crop_icon(name: str) -> str:
    return CROP_ICONS.get(name, "🌾")


def make_kpi_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="
            padding:12px 16px;
            border-radius:12px;
            background-color: #ffffff;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-bottom:8px;
        ">
            <div style="font-size:14px; color:#555;">{title}</div>
            <div style="font-size:26px; font-weight:700; color:#1b4332;">{value}</div>
            <div style="font-size:12px; color:#777;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def best_route_summary(df: pd.DataFrame):
    if df.empty:
        return None
    row = df.loc[df["Profit_per_kg"].idxmax()]
    return {
        "market": row["Market"],
        "city": row["City"],
        "commodity": row["Commodity"],
        "profit_per_kg": row["Profit_per_kg"],
        "profit_truck": row["Profit_per_truck"],
        "distance": row["Distance_km"],
        "state": row["State"],
    }


# ------------------- PAGES -------------------

def page_overview(filtered_df: pd.DataFrame, selected_commodity: str):
    # KPIs
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    make_kpi_card("Total Rows", f"{len(filtered_df):,}", "after applying filters")
    make_kpi_card("Avg Profit / kg (₹)", f"{filtered_df['Profit_per_kg'].mean():.2f}")
    make_kpi_card("Max Profit / kg (₹)", f"{filtered_df['Profit_per_kg'].max():.2f}")
    make_kpi_card(
        "Avg Opportunity Score",
        f"{filtered_df['Opportunity_Score'].mean():.1f}",
        "0–100 scale",
    )

    st.markdown("### 🧾 Filtered Data Preview")
    st.dataframe(filtered_df.head(30), use_container_width=True)

    # Quick commodity summary
    if selected_commodity != "All":
        st.markdown("---")
        st.markdown(f"### ℹ️ {get_crop_icon(selected_commodity)} {selected_commodity} summary")
        grp = filtered_df.groupby("Market")["Profit_per_kg"].mean().sort_values(ascending=False).head(5)
        st.write("Top mandis by average profit:")
        st.dataframe(grp.rename("Avg Profit / kg (₹)"))


def page_profit_analytics(filtered_df: pd.DataFrame):
    st.markdown("### 📈 Profit & Opportunity Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Profit per kg Distribution")
        fig = px.histogram(
            filtered_df,
            x="Profit_per_kg",
            nbins=40,
            title="Profit per kg Distribution",
            color_discrete_sequence=["#3B7D02"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Opportunity Score Distribution")
        fig3 = px.histogram(
            filtered_df,
            x="Opportunity_Score",
            nbins=40,
            title="Opportunity Score Distribution",
            color_discrete_sequence=["#F59E0B"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("#### Profit vs Distance (colored by Opportunity)")
        fig2 = px.scatter(
            filtered_df,
            x="Distance_km",
            y="Profit_per_kg",
            color="Opportunity_Score",
            color_continuous_scale="YlGn",
            hover_data=["Market", "City", "Commodity"],
            title="Profit vs Distance",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### State-wise Average Profit (Map)")
        state_group = (
            filtered_df.groupby("State")["Profit_per_kg"].mean().reset_index()
        )
        state_group["lat"] = state_group["State"].map(
            {k: v[0] for k, v in STATE_COORDS.items()}
        )
        state_group["lon"] = state_group["State"].map(
            {k: v[1] for k, v in STATE_COORDS.items()}
        )
        state_group = state_group.dropna(subset=["lat", "lon"])

        if not state_group.empty:
            fig_map = px.scatter_geo(
                state_group,
                lat="lat",
                lon="lon",
                size="Profit_per_kg",
                color="Profit_per_kg",
                color_continuous_scale="YlGn",
                hover_name="State",
                title="Average Profit per kg by State",
                scope="asia",
                center={"lat": 22.5, "lon": 79.0},
                projection="natural earth",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Not enough state data to draw the map for current filters.")


def page_route_recommendations(filtered_df: pd.DataFrame):
    st.markdown("### 🛣️ Route Recommendations")

    summary = best_route_summary(filtered_df)
    if summary is None:
        st.warning("No data available for current filters.")
        return

    st.markdown("#### 🔥 Best Route (Highest Profit per kg)")
    st.markdown(
        f"""
        <div style="
            padding:16px;
            border-radius:12px;
            background: linear-gradient(135deg,#15803d,#4ade80);
            color:white;
            margin-bottom:16px;
        ">
            <h3 style="margin:0 0 8px 0;">{summary['commodity']} from {summary['market']} → {summary['city']}</h3>
            <p style="margin:0 0 4px 0; font-size:15px;">
                State: {summary['state']} | Distance: {summary['distance']} km
            </p>
            <p style="margin:0; font-size:16px;">
                Profit per kg: <b>₹{summary['profit_per_kg']:.2f}</b> |
                Profit per 5000 kg truck: <b>₹{summary['profit_truck']:,.0f}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🥇 Top 10 Routes by Profit per kg")
    top = filtered_df.sort_values(by="Profit_per_kg", ascending=False).head(10)
    st.dataframe(
        top[
            [
                "State",
                "Market",
                "City",
                "Commodity",
                "Profit_per_kg",
                "Profit_per_truck",
                "Distance_km",
            ]
        ],
        use_container_width=True,
    )


def page_ml_prediction(model, filtered_df: pd.DataFrame):
    st.markdown("### 🤖 AI What-if Profit Prediction")

    if model is None:
        st.error("Model could not be loaded. Ensure profit_model.pkl exists.")
        return

    st.info(
        "Adjust mandi price, retail price, distance, and transport rate to simulate expected profit."
    )

    default_modal = float(filtered_df["Modal_Price"].median())
    default_retail = float(filtered_df["Retail_Price"].median())
    default_dist = float(filtered_df["Distance_km"].median())
    default_rate = float(filtered_df["Transport_rate_per_km"].median())

    c1, c2 = st.columns(2)
    with c1:
        modal = st.number_input(
            "Mandi Modal Price (₹/kg)", 0.0, 200.0, default_modal, step=0.5
        )
        distance = st.number_input(
            "Distance (km)", 0.0, 3000.0, default_dist, step=10.0
        )
    with c2:
        retail = st.number_input(
            "Retail Price (₹/kg)", 0.0, 300.0, default_retail, step=0.5
        )
        rate = st.number_input(
            "Transport Rate (₹ per km per ton)",
            0.0,
            100.0,
            default_rate,
            step=1.0,
        )

    if st.button("Predict Profit"):
        transport_cost = (distance * rate) / 1000.0
        spread = retail - modal
        features = np.array([[modal, retail, distance, rate, transport_cost, spread]])

        pred = float(model.predict(features)[0])
        truck_profit = pred * 5000

        make_kpi_card("Predicted Profit / kg (₹)", f"{pred:.2f}")
        make_kpi_card("Predicted Profit / Truck (₹)", f"{truck_profit:,.0f}")
        st.success(
            "Prediction generated using trained RandomForest model on historical data."
        )


def page_about():
    st.markdown("### ℹ️ About this Project")
    st.markdown(
        """
        **Mandi–Retail Price & Profit Dashboard** is a data science project that:

        - Simulates mandi (wholesale) and retail prices for key crops.
        - Computes transport cost, profit per kg, and profit per truck.
        - Calculates an **Opportunity Score (0–100)** based on profit and volatility.
        - Uses a **RandomForest Regression model** to predict profit per kg.
        - Provides a dashboard for:
            - Filtering by crop, mandi, and city  
            - Visualizing price spreads & profits  
            - Discovering top profitable trade routes  
            - Running "what-if" simulations using the ML model  

        This project is built end-to-end with:
        - **Python, Pandas, Scikit-learn**
        - **Streamlit & Plotly**
        - **GitHub for version control**
        """
    )


# ------------------- MAIN APP -------------------

def main():
    st.set_page_config(
        page_title="Mandi–Retail Price & Profit Dashboard",
        layout="wide",
    )

    df = load_data()
    model = load_model()

    # -------- Sidebar: Filters + Page selector --------
    st.sidebar.markdown("## 🔍 Filters")

    commodities = sorted(df["Commodity"].unique())
    selected_commodity = st.sidebar.selectbox("Select Commodity", ["All"] + commodities)

    filtered_df = df.copy()
    if selected_commodity != "All":
        filtered_df = filtered_df[filtered_df["Commodity"] == selected_commodity]

    markets = sorted(filtered_df["Market"].unique())
    selected_market = st.sidebar.selectbox("Select Mandi (Market)", ["All"] + markets)
    if selected_market != "All":
        filtered_df = filtered_df[filtered_df["Market"] == selected_market]

    cities = sorted(filtered_df["City"].unique())
    selected_city = st.sidebar.selectbox("Select City", ["All"] + cities)
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["City"] == selected_city]

    st.sidebar.markdown(f"Rows after filtering: **{len(filtered_df)}**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "📂 Go to page",
        ["Overview", "Profit Analytics", "Route Recommendations", "ML Prediction", "About"],
    )

    # -------- Header --------
    if selected_commodity == "All":
        icon = "🌾"
        crop_label = "All Crops"
    else:
        icon = get_crop_icon(selected_commodity)
        crop_label = selected_commodity

    st.markdown(
        f"""
        <div style="
            padding:18px 24px;
            background-color:#3B7D02;
            border-radius:14px;
            margin-bottom:18px;
        ">
            <h1 style="color:white; margin:0; text-align:center;">
                {icon} Mandi–Retail Price & Profit Dashboard
            </h1>
            <p style="color:#e5e5e5; margin:6px 0 0 0; text-align:center;">
                Current focus: <b>{crop_label}</b> | Analyze price spreads, profits, and trade opportunities across Indian mandis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning("No data for the selected filters. Please change filters in sidebar.")
        return

    # -------- Page routing --------
    if page == "Overview":
        page_overview(filtered_df, selected_commodity)
    elif page == "Profit Analytics":
        page_profit_analytics(filtered_df)
    elif page == "Route Recommendations":
        page_route_recommendations(filtered_df)
    elif page == "ML Prediction":
        page_ml_prediction(model, filtered_df)
    else:
        page_about()


if __name__ == "__main__":
    main()
