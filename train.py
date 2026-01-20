# train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/project9_wholesale dataset.csv")

# ---------------- IQR OUTLIER TREATMENT ----------------
df_iqr = df.copy()

for col in df_iqr.columns:
    Q1 = df_iqr[col].quantile(0.25)
    Q3 = df_iqr[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_iqr[col] = np.clip(df_iqr[col], lower_bound, upper_bound)

# ---------------- FEATURE SCALING ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_iqr)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# ---------------- FEATURES ----------------
X = scaled_df.drop(columns=["Channel"])

# ---------------- PCA ----------------
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# ---------------- KMEANS ----------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
kmeans.fit(X_pca)

# ---------------- SAVE SINGLE MODEL FILE -----------
model_bundle = {
    "scaler": scaler,
    "pca": pca,
    "kmeans": kmeans,
    "feature_columns": X.columns.tolist()
}

joblib.dump(model_bundle, "model/model.pkl")

print("Training completed with IQR outlier treatment")
print("Single model.pkl saved successfully")