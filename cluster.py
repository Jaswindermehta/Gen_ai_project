import torch
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# -----------------------------
# Storage for features collected
# across diffusion timesteps
# -----------------------------
feature_storage = []

def reset_features():
    global feature_storage
    feature_storage = []


# -----------------------------
# Store features
# -----------------------------
def store_feature(feature):
    """
    feature: tensor [B,C,H,W] from UNet
    Convert to channel features [B,C]
    """

    B, C, H, W = feature.shape

    # global average pooling
    feature = feature.mean(dim=[2,3])  # [B,C]

    feature_storage.append(feature.detach().cpu())


# -----------------------------
# Dimension indicator
# (temporal change)
# -----------------------------

def save_indicators(indicator, filename="dimension_indicators.json"):
    
    indicator_dict = {
        "num_dimensions": len(indicator),
        "indicators": indicator.tolist()
    }

    with open(filename, "w") as f:
        json.dump(indicator_dict, f, indent=4)

    print(f"Indicators saved to {filename}")

def compute_dimension_indicator(feature_list):

    deltas = []

    for i in range(1, len(feature_list)):

        f1 = feature_list[i]
        f0 = feature_list[i-1]

        delta = torch.abs(f1 - f0)

        deltas.append(delta)

    deltas = torch.stack(deltas)

    indicator = deltas.mean(dim=(0,1))  # [C]

    return indicator


# -----------------------------
# Build dimension matrix
# -----------------------------
def build_dimension_matrix(feature_list):

    feats = torch.cat(feature_list, dim=0)  # [T*B, C]

    dim_matrix = feats.T                    # [C, T*B]

    return dim_matrix.numpy()


# -----------------------------
# Elbow method
# -----------------------------
def find_best_k(dim_matrix, k_min=4, k_max=8):

    distortions = []
    n_samples = dim_matrix.shape[0]

    k_min = 2
    k_max = min(8, n_samples - 1)
    K_range = list(range(k_min, k_max + 1))

    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=0,
            n_init=10
        )

        kmeans.fit(dim_matrix)
        distortions.append(kmeans.inertia_)

    # ----- save elbow plot -----
    plt.figure(figsize=(6,4))
    plt.plot(K_range, distortions, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig("elbow_plot.png")
    plt.close()

    # ----- automatic elbow detection -----
    distortions = np.array(distortions)

    # second derivative approximation
    curvature = np.diff(distortions, 2)

    elbow_index = np.argmin(curvature) + 1

    best_k = K_range[elbow_index]

    print("Distortions:", distortions)
    print("Chosen k:", best_k)

    return best_k

# -----------------------------
# KMeans clustering
# -----------------------------
def cluster_dimensions(dim_matrix, k, method="kmeans"):

    print(f"Clustering method: {method}")

    if method == "kmeans":

        model = KMeans(
            n_clusters=k,
            random_state=0,
            n_init=10
        )

        labels = model.fit_predict(dim_matrix)

    elif method == "spectral":

        model = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            assign_labels="kmeans"
        )

        labels = model.fit_predict(dim_matrix)

    elif method == "hierarchical":

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )

        labels = model.fit_predict(dim_matrix)

    elif method == "gmm":

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0
        )

        labels = model.fit_predict(dim_matrix)

    else:
        raise ValueError("Unknown clustering method")

    return labels


# -----------------------------
# PCA visualization
# -----------------------------
def visualize_clusters(dim_matrix, labels):

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(dim_matrix)

    plt.figure(figsize=(6,5))

    scatter = plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels,
        cmap="tab10"
    )

    plt.title("Feature Dimension Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.colorbar(scatter)

    plt.savefig("cluster_plot.png")
    plt.close()


# -----------------------------
# Main clustering pipeline
# -----------------------------
def run_clustering():
    print("clustering started")
    if len(feature_storage) < 2:
        print("Not enough features collected.")
        return

    print("Computing dimension indicator...")

    indicator = compute_dimension_indicator(feature_storage)

    print("Building dimension matrix...")

    dim_matrix = build_dimension_matrix(feature_storage)

    print("Finding best k using elbow method...")

    

    find_best_k(dim_matrix)

    # After seeing elbow plot choose k
    k = find_best_k(dim_matrix, 4, 8)

    print(f"Running KMeans with k={k}")
    method="kmeans"

    labels = cluster_dimensions(dim_matrix, k,method)

    unique, counts = np.unique(labels, return_counts=True)

    print("Cluster distribution:")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} channels")

    with open("cluster_labels_{method}.json", "w") as f:
        json.dump(labels.tolist(), f, indent=4)

    print("Visualizing clusters with PCA")

    visualize_clusters(dim_matrix, labels)

    print("Indicator shape:", indicator.shape)
    print("Cluster labels shape:", labels.shape)

    print("Computing dimension indicator...")

    indicator = compute_dimension_indicator(feature_storage)

    save_indicators(indicator)