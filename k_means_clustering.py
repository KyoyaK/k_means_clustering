#Data points
#Generate k number of random points - centroids
#k number of clusters
    #each point goes in cluster of closest centroid
#Calculate new centroids using average of points
#Repeat until centroids dont move
#Calculate inertia - sum of all distances to respective centroids

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

k = 10
points = 250

rng = np.random.default_rng()

data = rng.uniform(-100, 100, (points, 2))
centroids = rng.uniform(-100, 100, (k, 2))

def find_distances(centroids_pos, data_points):
    #kx1
    centroids_dist_sq = np.diag(np.dot(centroids_pos, centroids_pos.T))
    #100x1
    data_dist_sq = np.diag(np.dot(data_points, data_points.T))
    #need an array of centroid points dotted w data points
    #100xk
    centroids_dot_data = np.dot(data_points, centroids_pos.T)
    return data_dist_sq[:, None]+centroids_dist_sq[None, :]-2*centroids_dot_data


def assign_clusters(data_points, squared_distances):
    #get closest centroid for each
    groups_indices = np.argmin(squared_distances, axis=1)
    return [data_points[groups_indices==i] for i in range(k)]

    
def find_averages(clusters):
    averages_array = np.vstack([np.mean(group, axis=0) for group in clusters])
    return averages_array


def find_inertia(distances_sq):
    min_dists = np.min(distances_sq, axis=1)
    return np.sum(min_dists)


#ANIMATION SECTION


fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1000000)

group_points = []

iteration = 0
iterations = []
intertias = []

inertias_plot = ax2.plot(iterations ,intertias, c="0", label="iterations")[0]
centroid_points = ax.scatter([], [], c="0", label = "centroids")
average_points = ax.scatter([], [], c="r", label="average")

ax.legend()
ax2.legend()


def init():
    global centroids, data, iteration, intertias
    global centroid_points, average_points, group_points, inertias_plot

    centroids = rng.uniform(-100, 100, (k, 2))
    distances_sq = find_distances(centroids, data)
    groups = assign_clusters(data, distances_sq)
    averages = find_averages(groups)
    inertia = find_inertia(distances_sq)

    iteration = 0
    iterations.clear()
    intertias.clear()
    iterations.append(iteration)
    intertias.append(inertia)

    centroid_points = ax.scatter(centroids[:, 0], centroids[:, 1], c="0")
    average_points = ax.scatter(averages[:, 0], averages[:, 1], c="r")
    inertias_plot = ax2.plot(iterations ,intertias)[0]
    inertias_plot.set_xdata(iterations)
    inertias_plot.set_ydata(intertias)


    group_points = []
    for _ in range(k):
        group_x = ax.scatter(groups[_][:, 0], groups[_][:, 1], c=f"C{_}")
        group_points.append(group_x)
    
    return (centroid_points, average_points, inertias_plot) + tuple(group_points)
    

def animate(frame):
    global centroids, data, group_points, iteration, intertias
    global centroid_points, average_points, group_points, inertias_plot

    centroid_points.set_offsets(centroids)


    distances_sq = find_distances(centroids, data)
    groups = assign_clusters(data, distances_sq)
    averages = find_averages(groups)
    centroids = averages
    inertia = find_inertia(distances_sq)


    iteration +=1
    iterations.append(iteration)
    intertias.append(inertia)
    if iterations[0]!=0:
        iterations.insert(0, 0)
    inertias_plot.set_xdata(iterations)
    inertias_plot.set_ydata(intertias)


    for _ in range(k):
        group_points[_].set_offsets(groups[_])
    average_points.set_offsets(averages)
    return (centroid_points, average_points, inertias_plot) + tuple(group_points)

ax2.legend()

anim = animation.FuncAnimation(fig=fig, frames=10, func=animate, init_func=init,
                               interval=150, blit=True, repeat=True)



plt.show()