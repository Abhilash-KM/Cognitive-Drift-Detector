import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

st.title("🧠 Cognitive Drift Detector")

# Load dataset
df = pd.read_csv("../data/responses.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(df["response"].tolist())

# Baseline vector
baseline_vector = np.mean(embeddings, axis=0)

# Statistical drift scores
drift_scores = []
for emb in embeddings:
    similarity = cosine_similarity([baseline_vector], [emb])[0][0]
    drift_scores.append(1 - similarity)

drift_threshold = np.mean(drift_scores) + np.std(drift_scores)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Autoencoder
X = torch.tensor(embeddings, dtype=torch.float32)
input_dim = X.shape[1]
latent_dim = 32

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model_ae = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_ae.parameters(), lr=0.01)

# Train quickly
for epoch in range(150):
    optimizer.zero_grad()
    outputs = model_ae(X)
    loss = criterion(outputs, X)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    reconstructed = model_ae(X)
    errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()

ae_threshold = errors.mean() + errors.std()

# ---- USER INPUT ----
new_text = st.text_area("Enter a new reasoning response:")

if st.button("Analyze Cognitive Drift"):

    new_embedding = model.encode([new_text])
    new_tensor = torch.tensor(new_embedding, dtype=torch.float32)

    # Statistical Drift
    similarity = cosine_similarity([baseline_vector], new_embedding)[0][0]
    stat_drift = 1 - similarity

    # Neural Drift
    with torch.no_grad():
        reconstructed_new = model_ae(new_tensor)
        neural_error = torch.mean((new_tensor - reconstructed_new) ** 2).item()

    # Cluster
    cluster_id = kmeans.predict(new_embedding)[0]

    st.subheader("Results")
    st.write("Statistical Drift Score:", round(stat_drift, 4))
    st.write("Neural Reconstruction Error:", round(neural_error, 6))
    st.write("Assigned Thinking Mode:", cluster_id)

    if stat_drift > drift_threshold:
        st.error("⚠️ Statistical Drift Detected")
    else:
        st.success("✓ Within Normal Statistical Range")

    if neural_error > ae_threshold:
        st.error("⚠️ Neural Drift Detected")
    else:
        st.success("✓ Neural Pattern Normal")

