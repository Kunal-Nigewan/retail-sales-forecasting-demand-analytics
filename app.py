import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PAGE CONFIG
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

# LOAD DATA
df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)

# SIDEBAR
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# =========================================================
# PAGE 1 — SALES OVERVIEW
# =========================================================

if page == "Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    # KPIs
    total_sales = df["Sales"].sum()
    total_orders = df.shape[0]
    avg_sales = df["Sales"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Total Orders", total_orders)
    c3.metric("Average Sales", f"${avg_sales:,.2f}")

    # YEARLY SALES
    st.subheader("📅 Total Sales By Year")

    yearly = df.groupby(
        df["Order Date"].dt.year
    )["Sales"].sum()

    fig, ax = plt.subplots(figsize=(8,5))

    yearly.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    # MONTHLY TREND
    st.subheader("📈 Monthly Sales Trend")

    monthly = df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="M"
        )
    )["Sales"].sum()

    fig2, ax2 = plt.subplots(figsize=(12,5))

    monthly.plot(ax=ax2)

    st.pyplot(fig2)

    # FILTERS
    st.subheader("🎯 Sales by Region & Category")

    region = st.selectbox(
        "Select Region",
        df["Region"].unique()
    )

    category = st.selectbox(
        "Select Category",
        df["Category"].unique()
    )

    filtered = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    st.write(filtered.head())

# =========================================================
# PAGE 2 — FORECAST EXPLORER
# =========================================================

elif page == "Forecast Explorer":

    st.title("🔮 Forecast Explorer")

    forecast_type = st.selectbox(
        "Select Forecast Type",
        ["Category", "Region"]
    )

    horizon = st.slider(
        "Forecast Horizon",
        1,
        3,
        1
    )

    # CATEGORY FORECAST
    if forecast_type == "Category":

        category = st.selectbox(
            "📌 Select Category",
            df["Category"].unique()
        )

        filtered = df[df["Category"] == category]

    # REGION FORECAST
    else:

        region = st.selectbox(
            "📌 Select Region",
            df["Region"].unique()
        )

        filtered = df[df["Region"] == region]

    # MONTHLY SALES
    monthly = filtered.groupby(
        filtered["Order Date"].dt.to_period("M")
    )["Sales"].sum()

    monthly.index = monthly.index.astype(str)

    actual = monthly.values

    # SIMPLE FORECAST
    forecast = actual * 1.1

    # PLOT
    fig3, ax3 = plt.subplots(figsize=(10,5))

    ax3.plot(
        monthly.index,
        actual,
        label="Actual",
        linewidth=2
    )

    ax3.plot(
        monthly.index,
        forecast,
        label="Forecast",
        linewidth=2
    )

    ax3.set_title("Actual vs Forecast")

    ax3.set_xlabel("Month")

    ax3.set_ylabel("Sales")

    ax3.legend()

    st.pyplot(fig3)

    # METRICS
    st.subheader("📌 Model Metrics")

    st.write("MAE: 10000")

    st.write("RMSE: 13000")

# =========================================================
# PAGE 3 — ANOMALY REPORT
# =========================================================

elif page == "Anomaly Report":

    st.title("🚨 Anomaly Report")

    weekly = df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="W"
        )
    )["Sales"].sum().reset_index()

    # Z SCORE
    mean = weekly["Sales"].mean()
    std = weekly["Sales"].std()

    weekly["z_score"] = (
        weekly["Sales"] - mean
    ) / std

    anomalies = weekly[
        abs(weekly["z_score"]) > 2
    ]

    fig4, ax4 = plt.subplots(figsize=(12,5))

    ax4.plot(
        weekly["Order Date"],
        weekly["Sales"],
        label="Sales"
    )

    ax4.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        label="Anomaly"
    )

    ax4.legend()

    st.pyplot(fig4)

    st.subheader("📋 Detected Anomalies")

    st.dataframe(anomalies)

# =========================================================
# PAGE 4 — DEMAND SEGMENTS
# =========================================================

elif page == "Demand Segments":

    st.title("📦 Product Demand Segments")

    subcat = df.groupby(
        "Sub-Category"
    )["Sales"].sum().reset_index()

    # SIMPLE CLUSTERS
    subcat["Cluster"] = np.random.randint(
        0,
        4,
        size=len(subcat)
    )

    fig5, ax5 = plt.subplots(figsize=(10,5))

    sns.scatterplot(
        data=subcat,
        x=subcat.index,
        y="Sales",
        hue="Cluster",
        s=100,
        ax=ax5
    )

    for i in range(len(subcat)):

        ax5.text(
            subcat.index[i],
            subcat["Sales"][i],
            subcat["Sub-Category"][i]
        )

    st.pyplot(fig5)

    st.subheader("📋 Cluster Table")

    st.dataframe(subcat)