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


# ------------------- HELPER CONSTANTS -------------------

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

# For map – approximate state centres (not exact, but enough for viz)
STATE_COORDS = {
    "Maharashtra": (19.7515, 75.7139),
    "Karnataka": (15.3173, 75.7139),
    "Gujarat": (22.2587, 71.1924),
    "Delhi": (28.7041, 77.1025),
    "UP": (26.8467, 80.9462),
    "Rajasthan": (27.0238, 74.2179),
}

# Crop-based accent colors
CROP_COLORS = {
    "Tomato": "#e11d48",   # red
    "Onion": "#8b5cf6",    # purple
    "Potato": "#b45309",   # brown
    "Brinjal": "#7c3aed",
    "Cabbage": "#16a34a",
    "Chilli": "#dc2626",
    "Garlic": "#ca8a04",
    "Ginger": "#92400e",
}


def get_crop_icon(name: str) -> str:
    return CROP_ICONS.get(name, "🌾")


def get_palette(dark_mode: bool, commodity: str):
    """Return color palette based on dark/light + commodity accent."""
    accent = CROP_COLORS.get(commodity, "#15803d")

    if dark_mode:
        return {
            "bg": "#0f172a",
            "card_bg": "rgba(15,23,42,0.9)",
            "header_start": accent,
            "header_end": "#0f172a",
            "text": "#f9fafb",
            "muted": "#94a3b8",
            "accent": accent,
            "border": "rgba(148,163,184,0.4)",
        }
    else:
        return {
            "bg": "#f7f8f2",
            "card_bg": "rgba(255,255,255,0.85)",
            "header_start": accent,
            "header_end": "#3b7d02",
            "text": "#111827",
            "muted": "#6b7280",
            "accent": accent,
            "border": "rgba(156,163,175,0.4)",
        }


def kpi_card(title: str, value: str, subtitle: str, icon: str, palette: dict):
    st.markdown(
        f"""
        <div style="
            padding:14px 16px;
            border-radius:16px;
            background:{palette['card_bg']};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border:1px solid {palette['border']};
            box-shadow: 0 8px 18px rgba(15,23,42,0.15);
            margin-bottom:10px;
        ">
            <div style="font-size:14px; color:{palette['muted']}; display:flex; align-items:center; gap:6px;">
                <span style="font-size:18px;">{icon}</span> {title}
            </div>
            <div style="font-size:26px; font-weight:700; color:{palette['text']}; margin-top:4px;">
                {value}
            </div>
            <div style="font-size:12px; color:{palette['muted']}; margin-top:2px;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_divider(title: str, palette: dict, icon: str = "🌾"):
    st.markdown(
        f"""
        <div style="margin:18px 0 10px 0; display:flex; align-items:center; gap:12px;">
            <div style="flex:1; height:1px; background:linear-gradient(to right, transparent, {palette['accent']});"></div>
            <div style="font-weight:600; color:{palette['muted']}; white-space:nowrap; font-size:14px;">
                {icon} {title}
            </div>
            <div style="flex:1; height:1px; background:linear-gradient(to left, transparent, {palette['accent']});"></div>
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
        "opp": row.get("Opportunity_Score", np.nan),
    }


# ------------------- PAGES -------------------

def page_overview(filtered_df: pd.DataFrame, selected_commodity: str, palette: dict):
    section_divider("Overview", palette, "📊")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card(
            "Total Rows",
            f"{len(filtered_df):,}",
            "after applying filters",
            "📦",
            palette,
        )
    with col2:
        kpi_card(
            "Avg Profit / kg (₹)",
            f"{filtered_df['Profit_per_kg'].mean():.2f}",
            "mean of all filtered rows",
            "💰",
            palette,
        )
    with col3:
        kpi_card(
            "Max Profit / kg (₹)",
            f"{filtered_df['Profit_per_kg'].max():.2f}",
            "best trade in current view",
            "🚀",
            palette,
        )
    with col4:
        kpi_card(
            "Avg Opportunity Score",
            f"{filtered_df['Opportunity_Score'].mean():.1f}",
            "0–100, higher is better",
            "🎯",
            palette,
        )

    section_divider("Filtered Data Preview", palette, "🧾")
    st.dataframe(filtered_df.head(40), use_container_width=True)

    if selected_commodity != "All":
        section_divider(f"{get_crop_icon(selected_commodity)} {selected_commodity} Summary", palette)
        grp = (
            filtered_df.groupby("Market")["Profit_per_kg"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        st.write("Top mandis by average profit per kg:")
        st.dataframe(grp.rename("Avg Profit / kg (₹)"))


def page_profit_analytics(filtered_df: pd.DataFrame, palette: dict):
    section_divider("Profit & Opportunity Analytics", palette, "📈")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Profit per kg Distribution")
        fig1 = px.histogram(
            filtered_df,
            x="Profit_per_kg",
            nbins=30,
            color_discrete_sequence=[palette["accent"]],
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("#### Opportunity Score Distribution")
        fig2 = px.histogram(
            filtered_df,
            x="Opportunity_Score",
            nbins=30,
            color_discrete_sequence=["#1E88E5"],
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("#### Profit per kg vs Distance")
        fig3 = px.scatter(
            filtered_df,
            x="Distance_km",
            y="Profit_per_kg",
            color="Opportunity_Score",
            color_continuous_scale="Viridis",
            hover_data=["Market", "City", "Commodity"],
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### State-wise Average Profit (India Map)")

        # Group by state
        state_group = (
            filtered_df.groupby("State")["Profit_per_kg"]
            .mean()
            .reset_index()
        )

        # Map lat/lon
        state_group["lat"] = state_group["State"].map(
            {k: v[0] for k, v in STATE_COORDS.items()}
        )
        state_group["lon"] = state_group["State"].map(
            {k: v[1] for k, v in STATE_COORDS.items()}
        )
        state_group = state_group.dropna(subset=["lat", "lon"])

        if not state_group.empty:
            p_min = state_group["Profit_per_kg"].min()
            p_max = state_group["Profit_per_kg"].max()
            if abs(p_max - p_min) < 1e-6:
                state_group["bubble_size"] = 18
            else:
                state_group["bubble_size"] = 10 + 30 * (
                    (state_group["Profit_per_kg"] - p_min) / (p_max - p_min)
                )

            fig_map = px.scatter_geo(
                state_group,
                lat="lat",
                lon="lon",
                size="bubble_size",
                color="Profit_per_kg",
                hover_name="State",
                color_continuous_scale="YlGn",
                projection="natural earth",
                title="Average Profit per kg by State",
            )
            fig_map.update_layout(
                geo=dict(
                    scope="asia",
                    center=dict(lat=22.0, lon=79.0),
                    lonaxis_range=[68, 90],
                    lataxis_range=[8, 32],
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Not enough state data to draw the map for current filters.")


def page_route_recommendations(filtered_df: pd.DataFrame, palette: dict):
    section_divider("Route Recommendations", palette, "🛣️")

    summary = best_route_summary(filtered_df)
    if summary is None:
        st.warning("No data available for the current filters.")
        return

    opp_text = (
        f"Opportunity Score: {summary['opp']:.1f}/100"
        if not np.isnan(summary["opp"])
        else ""
    )

    st.markdown(
        f"""
        <div style="
            padding:18px 20px;
            border-radius:18px;
            background:linear-gradient(135deg,{palette['accent']},#22c55e);
            color:white;
            margin-bottom:18px;
            box-shadow:0 10px 25px rgba(15,23,42,0.35);
        ">
            <div style="font-size:14px; opacity:0.9; margin-bottom:4px;">🔥 Best Route (Highest Profit per kg)</div>
            <div style="font-size:22px; font-weight:700; margin-bottom:6px;">
                {summary['commodity']} from {summary['market']} → {summary['city']}
            </div>
            <div style="font-size:14px; margin-bottom:4px;">
                State: {summary['state']} | Distance: {summary['distance']} km
            </div>
            <div style="font-size:16px;">
                Profit per kg: <b>₹{summary['profit_per_kg']:.2f}</b> |
                Profit per 5000 kg truck: <b>₹{summary['profit_truck']:,.0f}</b>
            </div>
            <div style="font-size:13px; margin-top:4px; opacity:0.9;">
                {opp_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top 10 Routes by Profit per kg")
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


def page_ml_prediction(model, filtered_df: pd.DataFrame, palette: dict):
    section_divider("AI What-if Profit Prediction", palette, "🤖")

    if model is None:
        st.error("Model could not be loaded. Ensure profit_model.pkl exists.")
        return

    st.info(
        "Change mandi price, retail price, distance, and transport rate to simulate expected profit using the trained ML model."
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

        kpi_card("Predicted Profit / kg", f"₹{pred:.2f}", "Based on ML model", "📈", palette)
        kpi_card("Predicted Profit / Truck", f"₹{truck_profit:,.0f}", "For 5000 kg truck", "🚚", palette)
        st.success("Prediction generated successfully.")


def page_about(palette: dict):
    section_divider("About this Project", palette, "ℹ️")
    st.markdown(
        """
        **Mandi–Retail Price & Profit Dashboard** is an end-to-end data science project that:

        - Simulates mandi (wholesale) and retail prices for key crops.  
        - Computes transport cost, profit per kg, and profit per truck.  
        - Calculates an **Opportunity Score (0–100)** based on profit and volatility.  
        - Trains a **RandomForest Regression model** to predict profit per kg.  
        - Deploys everything into a multi-page Streamlit dashboard with interactive filters, maps, and ML-powered what-if analysis.

        Tech stack:
        - Python, Pandas, NumPy  
        - Scikit-learn  
        - Streamlit, Plotly  
        - Git & GitHub for version control  
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

    # ----- SIDEBAR FILTERS + PAGE NAV + DARK MODE -----
    st.sidebar.markdown("## 🔍 Filters")

    commodities = sorted(df["Commodity"].unique())
    selected_commodity = st.sidebar.selectbox(
        "Select Commodity", ["All"] + commodities
    )

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
    dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)

    page = st.sidebar.radio(
        "📂 Go to page",
        ["Overview", "Profit Analytics", "Route Recommendations", "ML Prediction", "About"],
    )

    # ----- PALETTE + HEADER -----
    crop_for_palette = selected_commodity if selected_commodity != "All" else "Default"
    palette = get_palette(dark_mode, crop_for_palette)

    # Background color hack
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {palette['bg']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

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
            border-radius:18px;
            margin-bottom:18px;
            background:linear-gradient(135deg,{palette['header_start']},{palette['header_end']});
            box-shadow:0 10px 30px rgba(15,23,42,0.4);
        ">
            <h1 style="color:white; margin:0; text-align:center;">
                {icon} Mandi–Retail Price & Profit Dashboard
            </h1>
            <p style="color:#e5e7eb; margin:8px 0 0 0; text-align:center;">
                Current focus: <b>{crop_label}</b> &nbsp;|&nbsp;
                Analyze price spreads, profits, and trade opportunities across Indian mandis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning("No data for the selected filters. Please adjust the filters in the sidebar.")
        return

    # ----- ROUTE TO PAGES -----
    if page == "Overview":
        page_overview(filtered_df, selected_commodity, palette)
    elif page == "Profit Analytics":
        page_profit_analytics(filtered_df, palette)
    elif page == "Route Recommendations":
        page_route_recommendations(filtered_df, palette)
    elif page == "ML Prediction":
        page_ml_prediction(model, filtered_df, palette)
    else:
        page_about(palette)


if __name__ == "__main__":
    main()
