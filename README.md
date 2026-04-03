# k_means_clustering
Project which aimed to implement K-means clustering using NumPy and visualizing using Matplotlib. Generates random data, and generates k clusters. 

This is done by calculating the distance from each centroid to each data point, and then assigning groups to these centroids. The average coordinate in each group is calculated, which then becomes the location of the new batch of centroids. This is repeated until the inertia, the combined distances from the points to its respective centroid, is minimized.
The visualizer visualizes the k-means clustering algorithm by animating the shifts in the clusters using matplotlib and also graphing the inertia over each iteration

Known issues: None

