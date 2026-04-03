"""
Description: This code is an implementation of the k-means
clustering algorithm. The algorithm initially uses randomely
generated centroids, and assigns each datapoint to the 
nearest centroid. Then, the average point within each cluster is 
calculated, and these will be the new centroids. This is 
repeated until the positions of the centroids do not move. 


Bugs: None known. 
"""


import numpy as np


def find_distances(centroids_pos, data_points):
    return np.sum((data_points[:, None, :] - centroids_pos[None, :, :])**2, axis=2)


def assign_clusters(data_points, k, squared_distances):
    groups_indices = np.argmin(squared_distances, axis=1)
    clusters = [data_points[groups_indices==i] for i in range(k)]
    return clusters

    
def find_averages(clusters):
    averages = []
    rng = np.random.default_rng()
    for group in clusters:
        if len(group) == 0:
            #add random centroid
            averages.append(rng.uniform(-100, 100, size=(2)))
        else:
            averages.append(np.mean(group, axis=0))
    return np.array(averages)


def find_inertia(distances_sq):
    min_dists = np.min(distances_sq, axis=1)
    return np.sum(min_dists)


def kmeans(data, k, iterations):
    """Runs K-means clustering algorithm."""
    rng = np.random.default_rng()
    centroids = rng.uniform(-100, 100, (k, 2))
    inertia_history = []

    for _ in range(iterations):
        distances = find_distances(centroids, data)
        clusters = assign_clusters(data, k, distances)
        centroids = find_averages(clusters)
        
        inertia = find_inertia(distances)
        inertia_history.append(inertia)
    
    return centroids, clusters, inertia_history
    

rng = np.random.default_rng()
data = rng.uniform(-100, 100, size=(150, 2))
k=10

centroids, clusters, inertia_history = kmeans(data, k, 100)
print(inertia_history)
