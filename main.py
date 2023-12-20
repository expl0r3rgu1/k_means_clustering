import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from multiprocessing import Pool
import time

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

def kmeans_sequential(points, n_clusters, max_iter=100):
    initial_centroids = points[np.random.choice(len(points), n_clusters, replace=False)]
    for _ in range(max_iter):
        closest = assign_to_centroids(points, initial_centroids)
        new_centroids = compute_centroids(points, closest, n_clusters)
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

# Time the parallel version
start_par = time.time()
centroids_par, labels_par = kmeans_parallel(X, n_clusters, n_processes)
end_par = time.time()
print(f"Parallel version took {end_par - start_par} seconds")

# Time the sequential version
start_seq = time.time()
centroids_seq, labels_seq = kmeans_sequential(X, n_clusters)
end_seq = time.time()
print(f"Sequential version took {end_seq - start_seq} seconds")

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the points and the cluster centers (parallel)
axs[0].scatter(X[:, 0], X[:, 1], c=labels_par, s=50, cmap='viridis')
axs[0].scatter(centroids_par[:, 0], centroids_par[:, 1], c='black', s=200, alpha=0.5)
axs[0].set_title('Parallel')

# Plot the points and the cluster centers (sequential)
axs[1].scatter(X[:, 0], X[:, 1], c=labels_seq, s=50, cmap='viridis')
axs[1].scatter(centroids_seq[:, 0], centroids_seq[:, 1], c='black', s=200, alpha=0.5)
axs[1].set_title('Sequential')

# Save the figure
fig.savefig('results/plot.png')

# Show the plot
plt.show()
