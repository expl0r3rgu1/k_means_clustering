import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from multiprocessing import Pool

def assign_to_centroids(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    closest = np.argmin(distances, axis=0)
    return closest

def compute_centroids(points, closest, n_clusters):
    return np.array([points[closest==k].mean(axis=0) for k in range(n_clusters)])

def kmeans_parallel(points, n_clusters, n_processes, max_iter=100):
    initial_centroids = points[np.random.choice(len(points), n_clusters, replace=False)]
    with Pool(processes=n_processes) as pool:
        for _ in range(max_iter):
            closest = pool.apply(assign_to_centroids, (points, initial_centroids))
            new_centroids = pool.apply(compute_centroids, (points, closest, n_clusters))
            if np.all(initial_centroids == new_centroids):
                break
            initial_centroids = new_centroids
    return initial_centroids, closest

# Generate sample data
n_samples = 1000
n_clusters = 4
n_processes = 2
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=0, cluster_std=0.60)

# Perform k-means clustering
centroids, labels = kmeans_parallel(X, n_clusters, n_processes)

# Plot the points and the cluster centers
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
plt.show()
