#Convert your text responses into numerical embeddings
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load dataset
df = pd.read_csv("data/responses.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset Loaded Successfully")
print("Number of responses:", len(df))

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert responses to embeddings
print("Converting responses into embeddings...")
embeddings = model.encode(df["response"].tolist())

print("Embedding shape:", embeddings.shape)

#PyTorch Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim

# Convert embeddings to torch tensor
X = torch.tensor(embeddings, dtype=torch.float32)

input_dim = X.shape[1]  # 384
latent_dim = 32  # compressed thinking signature

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
        reconstructed = self.decoder(z)
        return reconstructed

model_ae = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(model_ae.parameters(), lr=0.01)

print("\nTraining Autoencoder...")

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model_ae(X)
    loss = criterion(outputs, X)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

#Calculate Reconstruction Error
with torch.no_grad():
    reconstructed = model_ae(X)
    errors = torch.mean((X - reconstructed) ** 2, dim=1)

errors = errors.numpy()

print("\nReconstruction Errors:")
for i, e in enumerate(errors):
    print(f"Response {i+1}: {e:.6f}")

#Detect Neural Drift
threshold_ae = errors.mean() + errors.std()

print("\nAutoencoder Drift Threshold:", threshold_ae)

for i, e in enumerate(errors):
    if e > threshold_ae:
        print(f"⚠️ Neural Drift detected at Response {i+1}")

#clusterring code
from sklearn.cluster import KMeans

print("\nClustering reasoning patterns...")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

df["cluster"] = clusters

for i in range(len(df)):
    print(f"Response {i+1} → Cluster {clusters[i]}")

#Detect Mode Changes
print("\nDetecting Mode Switching...")

mode_switches = 0

for i in range(1, len(df)):
    if df.loc[i, "cluster"] != df.loc[i-1, "cluster"]:
        mode_switches += 1
        print(f"Mode switch at Response {i+1}")

print("\nTotal Mode Switches:", mode_switches)

#visualize your thinking patterns.
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("Reducing dimensions for visualization...")

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
reduced = tsne.fit_transform(embeddings)

plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("Reasoning Pattern Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#visualize using clusters
plt.figure()
plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters)
plt.title("Reasoning Modes (Clustered)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#Visualize Mode Over Time
plt.figure()
plt.plot(df["timestamp"], df["cluster"], marker='o')
plt.title("Thinking Mode Over Time")
plt.xlabel("Date")
plt.ylabel("Cluster ID")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Your Baseline “Thinking Signature” Model
import numpy as np

print("Calculating baseline reasoning signature...")

baseline_vector = np.mean(embeddings, axis=0)

print("Baseline vector shape:", baseline_vector.shape)

#Measure Drift for Each Response
from sklearn.metrics.pairwise import cosine_similarity

print("\nCalculating drift scores...")

drift_scores = []

for emb in embeddings:
    similarity = cosine_similarity(
        [baseline_vector], 
        [emb]
    )[0][0]
    
    drift = 1 - similarity  # higher = more drift
    drift_scores.append(drift)

for i, score in enumerate(drift_scores):
    print(f"\nResponse {i+1}")
    print("Prompt:", df.iloc[i]["prompt"])
    print("Drift Score:", round(score, 4))

    print("\nResponse Lengths:")
for i in range(len(df)):
    print(f"Response {i+1} Length:", len(df.iloc[i]["response"]))

    threshold = np.mean(drift_scores) + np.std(drift_scores)

print("\nDrift Threshold:", round(threshold,4))

for i, score in enumerate(drift_scores):
    if score > threshold:
        print(f"⚠️ Response {i+1} shows significant cognitive drift.")

#Visualize Drift Over Time
import matplotlib.pyplot as plt

# Convert timestamp column to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

plt.figure()
plt.plot(df["timestamp"], drift_scores, marker='o')
plt.title("Cognitive Drift Over Time")
plt.xlabel("Date")
plt.ylabel("Drift Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Drift Comparison Matrix.
print("\n--- Drift Comparison ---")

for i in range(len(df)):
    stat_flag = drift_scores[i] > (np.mean(drift_scores) + np.std(drift_scores))
    neural_flag = errors[i] > threshold_ae
    
    if stat_flag or neural_flag:
        print(f"Response {i+1}: "
              f"Stat Drift={stat_flag}, "
              f"Neural Drift={neural_flag}")

#live cognitive analyzer.
print("\n==============================")
print("REAL-TIME COGNITIVE DRIFT TEST")
print("==============================")

new_text = input("\nEnter a new reasoning response:\n")

# Convert new text to embedding
new_embedding = model.encode([new_text])
new_embedding_tensor = torch.tensor(new_embedding, dtype=torch.float32)

# -------- Statistical Drift --------
similarity = cosine_similarity(
    [baseline_vector],
    new_embedding
)[0][0]

stat_drift = 1 - similarity

print("\nStatistical Drift Score:", round(stat_drift, 4))

if stat_drift > (np.mean(drift_scores) + np.std(drift_scores)):
    print("⚠️ Statistical Drift Detected")
else:
    print("✓ Within normal cognitive range")

# -------- Neural Drift --------
with torch.no_grad():
    reconstructed_new = model_ae(new_embedding_tensor)
    neural_error = torch.mean((new_embedding_tensor - reconstructed_new) ** 2).item()

print("Neural Reconstruction Error:", round(neural_error, 6))

if neural_error > threshold_ae:
    print("⚠️ Neural Drift Detected")
else:
    print("✓ Neural pattern normal")

# -------- Cluster Assignment --------
cluster_id = kmeans.predict(new_embedding)[0]
print("Assigned Thinking Mode (Cluster):", cluster_id)