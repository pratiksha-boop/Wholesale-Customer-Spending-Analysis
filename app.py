import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="wide"
)

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()

df = artifacts["data"]
df_kmeans = artifacts["data_kmeans"]
preprocessor = artifacts["preprocessor"]
kmeans = artifacts["kmeans"]
hierarchical = artifacts["hierarchical"]
X_pca = artifacts["X_pca"]

cluster_profile_mean = artifacts["cluster_profile_mean"]
cluster_profile_median = artifacts["cluster_profile_median"]
cluster_counts = artifacts["cluster_counts"]

# -----------------------------
# Title
# -----------------------------
st.title("Wholesale Customer Segmentation – Unsupervised Learning")

tabs = st.tabs([
    "Dataset Overview",
    "K-Means Clustering (3D)",
    "Cluster Analysis",
    "Predict Customer Cluster"
])

# -----------------------------
# TAB 1: Dataset Overview
# -----------------------------
with tabs[0]:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# -----------------------------
# TAB 2: K-Means 3D Visualization
# -----------------------------
with tabs[1]:
    st.subheader("3D K-Means Clusters (PCA Reduced)")

    pca_df = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2", "PC3"]
    )
    pca_df["Cluster"] = df_kmeans["KMeans_Cluster"].astype(str)

    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        opacity=0.8
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 3: Cluster Analysis
# -----------------------------
with tabs[2]:
    st.subheader("Cluster Size Distribution")
    st.bar_chart(cluster_counts)

    st.subheader("Mean Spending per Cluster")
    st.dataframe(cluster_profile_mean)

    st.subheader("Median Spending per Cluster")
    st.dataframe(cluster_profile_median)

# -----------------------------
# TAB 4: Predict Cluster
# -----------------------------
with tabs[3]:
    st.subheader("Predict K-Means Cluster for a New Customer")

    col1, col2 = st.columns(2)

    with col1:
        fresh = st.number_input("Fresh", min_value=0.0)
        milk = st.number_input("Milk", min_value=0.0)
        grocery = st.number_input("Grocery", min_value=0.0)
        frozen = st.number_input("Frozen", min_value=0.0)

    with col2:
        detergents = st.number_input("Detergents_Paper", min_value=0.0)
        delicassen = st.number_input("Delicassen", min_value=0.0)
        channel = st.selectbox("Channel", [1, 2])
        region = st.selectbox("Region", [1, 2, 3])

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([{
            "Fresh": fresh,
            "Milk": milk,
            "Grocery": grocery,
            "Frozen": frozen,
            "Detergents_Paper": detergents,
            "Delicassen": delicassen,
            "Channel": channel,
            "Region": region
        }])

        processed_input = preprocessor.transform(input_df)
        cluster = kmeans.predict(processed_input)[0]

        st.success(f"Predicted K-Means Cluster: {cluster}")
        st.write("Average Spending Pattern for this Cluster:")
        st.dataframe(cluster_profile_mean.loc[[cluster]])
