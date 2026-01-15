import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ================== LOAD FILES ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans_rfm_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "rfm_scaler.pkl"))

# ================== LOAD DATA (CACHED) ==================
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

    # Data Cleaning
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # ✅ FIXED DATE PARSING (IMPORTANT)
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["InvoiceDate"])

    # Total Amount
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df

df = load_data()

# ================== SIDEBAR ==================
st.sidebar.title("🛒 Shopper Spectrum")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Clustering", "Recommendation", "Analytics"],
    label_visibility="collapsed"
)

# ================== HOME ==================
if menu == "Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation & Product Recommendation System")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", df["CustomerID"].nunique())
    col2.metric("Total Products", df["Description"].nunique())
    col3.metric("Total Revenue", f"£{df['TotalAmount'].sum():,.0f}")

    st.markdown("""
    ### 🚀 Features
    - RFM-based Customer Segmentation  
    - KMeans Clustering  
    - Item-based Collaborative Filtering  
    - Interactive Analytics Dashboard  

    Use the sidebar to explore modules.
    """)

# ================== CLUSTERING ==================
elif menu == "Clustering":
    st.title("👤 Customer Segmentation")
    st.markdown("Predict customer segment using **RFM analysis**")

    c1, c2, c3 = st.columns(3)

    with c1:
        recency = st.number_input("Recency (days)", min_value=0, value=300)
    with c2:
        frequency = st.number_input("Frequency", min_value=0, value=2)
    with c3:
        monetary = st.number_input("Monetary (£)", min_value=0.0, value=5000.0)

    if st.button("🎯 Predict Segment"):
        scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = int(kmeans.predict(scaled)[0])

        segment_map = {
            0: "💎 High-Value Customer",
            1: "🙂 Regular Customer",
            2: "🛍️ Occasional Shopper",
            3: "⚠️ At-Risk Customer"
        }

        st.success(f"**Segment:** {segment_map.get(cluster)}")

# ================== COLLABORATIVE FILTERING ==================
elif menu == "Recommendation":
    st.title("🛍 Product Recommender")
    st.markdown("Item-based **Collaborative Filtering** using cosine similarity")

    @st.cache_data
    def build_similarity_matrix(data):
        user_item = data.pivot_table(
            index="CustomerID",
            columns="Description",
            values="Quantity",
            aggfunc="sum"
        ).fillna(0)

        similarity = cosine_similarity(user_item.T)
        return pd.DataFrame(
            similarity,
            index=user_item.columns,
            columns=user_item.columns
        )

    similarity_df = build_similarity_matrix(df)

    product = st.text_input(
        "Enter Product Name",
        placeholder="GREEN VINTAGE SPOT BEAKER"
    )

    if st.button("🔮 Recommend Products"):
        if product not in similarity_df.columns:
            st.error("Product not found in dataset")
        else:
            recommendations = (
                similarity_df[product]
                .sort_values(ascending=False)
                .iloc[1:6]
                .index
                .tolist()
            )

            st.markdown("### ✅ Recommended Products")
            for r in recommendations:
                st.info(r)

# ================== ANALYTICS DASHBOARD ==================
elif menu == "Analytics":
    st.title("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    # Top Countries
    with col1:
        country_sales = (
            df.groupby("Country")["TotalAmount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig1 = px.bar(
            country_sales,
            x="Country",
            y="TotalAmount",
            title="Top 10 Countries by Revenue"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Monthly Trend
    with col2:
        df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        monthly_sales = df.groupby("Month")["TotalAmount"].sum().reset_index()

        fig2 = px.line(
            monthly_sales,
            x="Month",
            y="TotalAmount",
            title="Monthly Revenue Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)
