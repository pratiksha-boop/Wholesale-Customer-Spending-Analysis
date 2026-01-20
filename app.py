# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- LOAD MODEL BUNDLE ----------------
model = joblib.load("model/model.pkl")

scaler = model["scaler"]
pca = model["pca"]
kmeans = model["kmeans"]
feature_columns = model["feature_columns"]

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/project9_wholesale dataset.csv")

st.set_page_config(page_title="Wholesale Customer Spending Analysis", layout="wide")
st.title("Wholesale Customer Spending Analysis Dashboard")

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
    st.dataframe(df.head())

    st.write("Shape:", df.shape)
    st.write("Statistical Summary")
    st.dataframe(df.describe())

# ---------------- TAB 2: KMEANS CLUSTERING ----------------
with tab2:
    st.subheader("3D KMeans Clustering")

    # Scale full dataset
    scaled_data = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    X = scaled_df[feature_columns]
    X_pca = pca.transform(X)
    clusters = kmeans.predict(X_pca)

    cluster_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    cluster_df["Cluster"] = clusters.astype(str)

    fig = px.scatter_3d(
        cluster_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        title="3D KMeans Clusters",
        opacity=0.8
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3: PCA ----------------
with tab3:
    st.subheader("3D PCA Visualization")

    fig_pca = px.scatter_3d(
        cluster_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        title="PCA Components (3D)"
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.write("Explained Variance Ratio")
    st.write(pca.explained_variance_ratio_)

# ---------------- TAB 4: USER INPUT ----------------
with tab4:
    st.subheader("Predict Customer Cluster")

    user_input = {}
    for col in df.columns:
        user_input[col] = st.number_input(
            col,
            min_value=0.0,
            value=float(df[col].median())
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Cluster"):
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=df.columns)

        input_X = scaled_input_df[feature_columns]
        input_pca = pca.transform(input_X)
        cluster = kmeans.predict(input_pca)

        st.success(f"Predicted Cluster: {cluster[0]}")
