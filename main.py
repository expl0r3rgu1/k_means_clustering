import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from multiprocessing import Pool
import matplotlib.pyplot as plt


def kmeans_worker(args):
    data, k, idx_range, max_iter = args
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0,
                    max_iter=max_iter, init='k-means++')
    return kmeans.fit(data[idx_range[0]:idx_range[1]])


def parallel_kmeans(data, k, num_processes=2, max_iter=300):
    num_samples, _ = data.shape
    chunk_size = num_samples // num_processes
    idx_ranges = [(i * chunk_size, (i + 1) * chunk_size)
                  for i in range(num_processes - 1)]
    idx_ranges.append(((num_processes - 1) * chunk_size, num_samples))

    with Pool(num_processes) as pool:
        results = pool.map(
            kmeans_worker, [(data, k, idx_range, max_iter) for idx_range in idx_ranges])

    # Combine the results from different processes
    final_kmeans = results[0]
    for result in results[1:]:
        final_kmeans.cluster_centers_ += result.cluster_centers_
        final_kmeans.labels_ = np.concatenate(
            [final_kmeans.labels_, result.labels_])

    final_kmeans.cluster_centers_ /= num_processes

    return final_kmeans


def plot_clusters(data, kmeans_result):
    plt.scatter(data[:, 0], data[:, 1], c=kmeans_result.labels_,
                cmap='viridis', alpha=0.5, edgecolors='w')
    plt.scatter(kmeans_result.cluster_centers_[
                :, 0], kmeans_result.cluster_centers_[:, 1], marker='x', s=200, c='red')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    data, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

    # Specify the number of clusters (k)
    k = 3

    # Number of parallel processes
    num_processes = 4

    # Maximum number of iterations
    max_iter = 300

    # Perform parallel k-means clustering
    parallel_kmeans_result = parallel_kmeans(data, k, num_processes, max_iter)

    # Visualize the clusters
    plot_clusters(data, parallel_kmeans_result)
