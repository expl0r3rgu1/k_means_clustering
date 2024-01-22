import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from multiprocessing import Pool
import time
from functools import partial


def assign_to_centroids(points, centroids):
    distances = np.linalg.norm(points - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)


def compute_centroids(points, closest, n_clusters):
    return np.array([points[closest == k].mean(axis=0) for k in range(n_clusters)])


def assign_and_compute_centroids(chunk, centroids, n_clusters):
    closest = assign_to_centroids(chunk, centroids)
    new_centroids = compute_centroids(chunk, closest, n_clusters)
    return new_centroids, closest


def kmeans_parallel(points, n_clusters, n_processes, max_iter=100, tol=1e-4):
    data_split = np.array_split(points, n_processes)
    initial_centroids = points[np.random.choice(
        len(points), n_clusters, replace=False)]
    with Pool(processes=n_processes) as pool:
        for _ in range(max_iter):
            assign_and_compute_partial = partial(
                assign_and_compute_centroids, centroids=initial_centroids, n_clusters=n_clusters)
            results = pool.map(assign_and_compute_partial, data_split)

            new_centroids_list, closest_list = zip(*results)
            new_centroids = np.mean(new_centroids_list, axis=0)
            closest = np.concatenate(closest_list)

            if np.linalg.norm(initial_centroids - new_centroids) < tol:
                break
            initial_centroids = new_centroids
    return initial_centroids, closest


def kmeans_sequential(points, n_clusters, max_iter=100, tol=1e-4):
    initial_centroids = points[np.random.choice(
        len(points), n_clusters, replace=False)]
    for _ in range(max_iter):
        new_centroids, closest = assign_and_compute_centroids(
            points, initial_centroids, n_clusters)
        if np.linalg.norm(initial_centroids - new_centroids) < tol:
            break
        initial_centroids = new_centroids
    return initial_centroids, closest


# Generate sample data
n_samples = 2000000
n_clusters = 8
n_processes = 4
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters,
                  random_state=0, cluster_std=0.60)

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
axs[0].scatter(centroids_par[:, 0], centroids_par[:, 1],
               c='black', s=200, alpha=0.5)
axs[0].set_title('Parallel')

# Plot the points and the cluster centers (sequential)
axs[1].scatter(X[:, 0], X[:, 1], c=labels_seq, s=50, cmap='viridis')
axs[1].scatter(centroids_seq[:, 0], centroids_seq[:, 1],
               c='black', s=200, alpha=0.5)
axs[1].set_title('Sequential')

# Save the figure
fig.savefig('results/plot.png')

# Show the plot
plt.show()
