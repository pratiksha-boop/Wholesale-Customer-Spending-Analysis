import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wholesale Customer Spending Analysis",
    layout="wide"
)

st.title("Wholesale Customer Spending Analysis Dashboard")

# ---------------- PATH HANDLING ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "model", "model.pkl"),  # preferred
    os.path.join(BASE_DIR, "model.pkl")             # fallback
]

DATA_PATH = os.path.join(BASE_DIR, "data", "project9_wholesale dataset.csv")

# ---------------- LOAD MODEL ----------------
model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        model = joblib.load(path)
        break

if model is None:
    st.error("❌ model.pkl not found. Ensure it is committed to the repository.")
    st.stop()

scaler = model["scaler"]
pca = model["pca"]
kmeans = model["kmeans"]
feature_columns = model["feature_columns"]

# ---------------- LOAD DATA ----------------
if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found. Please check the data folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🧩 KMeans Clustering",
    "📉 PCA Analysis",
    "🧮 Predict Customer Cluster"
])

# ---------------- TAB 1: DATA OVERVIEW ----------------
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("**Dataset Shape**")
    st.write(df.shape)

    st.write("**Statistical Summary**")
    st.dataframe(df.describe(), use_container_width=True)

# ---------------- TAB 2: KMEANS CLUSTERING ----------------
with tab2:
    st.subheader("3D KMeans Clustering")

    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    X = scaled_df[feature_columns]
    X_pca = pca.transform(X)
    clusters = kmeans.predict(X_pca)

    cluster_df = pd.DataFrame(
        X_pca, columns=["PC1", "PC2", "PC3"]
    )
    cluster_df["Cluster"] = clusters.astype(str)

    fig = px.scatter_3d(
        cluster_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        title="3D KMeans Customer Segmentation",
        opacity=0.8
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3: PCA ANALYSIS ----------------
with tab3:
    st.subheader("3D PCA Visualization")

    fig_pca = px.scatter_3d(
        cluster_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        title="Principal Component Analysis (3D)"
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.write("**Explained Variance Ratio**")
    for i, var in enumerate(pca.explained_variance_ratio_, start=1):
        st.write(f"PC{i}: {var:.4f}")

# ---------------- TAB 4: USER INPUT ----------------
with tab4:
    st.subheader("Predict Customer Cluster")

    user_input = {}
    for col in df.columns:
        user_input[col] = st.number_input(
            label=col,
            min_value=0.0,
            value=float(df[col].median())
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Cluster"):
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(
            scaled_input, columns=df.columns
        )

        input_X = scaled_input_df[feature_columns]
        input_pca = pca.transform(input_X)
        cluster = kmeans.predict(input_pca)

        st.success(f"✅ Predicted Customer Cluster: **{cluster[0]}**")
