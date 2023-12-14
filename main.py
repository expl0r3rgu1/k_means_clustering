import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from multiprocessing import Pool

def kmeans_worker(args):
    data, k, idx_range = args
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    return kmeans.fit(data[idx_range[0]:idx_range[1]])

def parallel_kmeans(data, k, num_processes=2):
    num_samples, _ = data.shape
    chunk_size = num_samples // num_processes
    idx_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes - 1)]
    idx_ranges.append(((num_processes - 1) * chunk_size, num_samples))

    with Pool(num_processes) as pool:
        results = pool.map(kmeans_worker, [(data, k, idx_range) for idx_range in idx_ranges])

    # Combine the results from different processes
    final_kmeans = results[0]
    for result in results[1:]:
        final_kmeans.cluster_centers_ += result.cluster_centers_
        final_kmeans.labels_ = np.concatenate([final_kmeans.labels_, result.labels_])

    final_kmeans.cluster_centers_ /= num_processes

    return final_kmeans

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    data, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

    # Specify the number of clusters (k)
    k = 3

    # Number of parallel processes
    num_processes = 4

    # Perform parallel k-means clustering
    parallel_kmeans_result = parallel_kmeans(data, k, num_processes)

    # Print the final cluster centers
    print("Final Cluster Centers:\n", parallel_kmeans_result.cluster_centers_)
