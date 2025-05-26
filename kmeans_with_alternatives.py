
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="K-Means Clustering with Alternatives", layout="wide")

st.title("🔍 K-Means Clustering with Elbow Method, PCA & Alternatives")
st.markdown("Upload an Excel file with an 'Alternative' column and numerical features to group named entities into clusters.")

uploaded_file = st.file_uploader("📤 Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    if 'Alternative' not in df.columns:
        st.error("❌ 'Alternative' column not found. Please include it in your Excel.")
    else:
        alternatives = df['Alternative']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect("Select columns for clustering", numeric_cols, default=numeric_cols)

        if selected_cols:
            data = df[selected_cols].dropna()

            # --- ELBOW METHOD ---
            st.subheader("📈 Elbow Method")
            sse = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
                sse.append(kmeans.inertia_)
            fig, ax = plt.subplots()
            ax.plot(K_range, sse, 'o-')
            ax.set_title("Elbow Method")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Inertia (SSE)")
            st.pyplot(fig)
            st.markdown("💡 **Tip:** Choose the K where the curve starts to bend.")

            # --- USER SELECTS K ---
            k_val = st.slider("🎯 Select number of clusters (K)", min_value=2, max_value=10, value=3)

            # --- K-MEANS ---
            model = KMeans(n_clusters=k_val, random_state=42)
            df['Cluster'] = model.fit_predict(data)

            st.subheader("🧾 Clustered Data")
            st.dataframe(df)

            # --- PCA ---
            st.subheader("📉 PCA: Principal Component Analysis (2D View)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data)
            df['PC1'] = pca_result[:, 0]
            df['PC2'] = pca_result[:, 1]

            # --- PCA PLOT WITH LABELS ---
            fig2, ax2 = plt.subplots()
            colors = plt.cm.tab10.colors
            for cluster in sorted(df['Cluster'].unique()):
                clustered = df[df['Cluster'] == cluster]
                ax2.scatter(clustered['PC1'], clustered['PC2'], label=f"Cluster {cluster}", alpha=0.7)
                for _, row in clustered.iterrows():
                    ax2.text(row['PC1'], row['PC2'], row['Alternative'], fontsize=8, alpha=0.6)
            ax2.set_xlabel("Principal Component 1")
            ax2.set_ylabel("Principal Component 2")
            ax2.set_title("PCA 2D Clustering View (with Labels)")
            ax2.legend()
            st.pyplot(fig2)

            # --- DOWNLOAD ---
            st.subheader("📥 Download Clustered Result")
            output = BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            st.download_button("⬇️ Download Excel with Clusters and PCA", data=output.getvalue(), file_name="clustered_result_with_alternatives.xlsx")
