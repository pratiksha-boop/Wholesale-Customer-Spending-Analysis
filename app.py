import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Wholesale Customer Spending Analysis", layout="wide")
st.title("Wholesale Customer Spending Analysis Dashboard")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "model", "model.pkl"),
    os.path.join(BASE_DIR, "model.pkl")
]

model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        model = joblib.load(path)
        break

if model is None:
    st.error("❌ model.pkl not found.")
    st.stop()

scaler = model["scaler"]
pca = model["pca"]
kmeans = model["kmeans"]
feature_columns = model["feature_columns"]
df = model["training_data"]  # ✅ FROM MODEL

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🧩 KMeans Clustering",
    "📉 PCA Analysis",
    "🧮 Predict Customer Cluster"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)
    st.write(df.describe())

# ---------------- TAB 2 ----------------
with tab2:
    scaled_data = scaler.transform(df)
    X = pd.DataFrame(scaled_data, columns=df.columns)[feature_columns]
    X_pca = pca.transform(X)
    clusters = kmeans.predict(X_pca)

    cluster_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    cluster_df["Cluster"] = clusters.astype(str)

    fig = px.scatter_3d(cluster_df, x="PC1", y="PC2", z="PC3", color="Cluster")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    fig_pca = px.scatter_3d(cluster_df, x="PC1", y="PC2", z="PC3", color="Cluster")
    st.plotly_chart(fig_pca, use_container_width=True)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ---------------- TAB 4 ----------------
with tab4:
    user_input = {
        col: st.number_input(col, min_value=0.0, value=float(df[col].median()))
        for col in df.columns
    }

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Cluster"):
        scaled_input = scaler.transform(input_df)
        X_input = pd.DataFrame(scaled_input, columns=df.columns)[feature_columns]
        input_pca = pca.transform(X_input)
        cluster = kmeans.predict(input_pca)
        st.success(f"Predicted Cluster: {cluster[0]}")
