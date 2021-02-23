# ## Task 3

# Study the performance of these two implementations: memory, speed, quality; compare against [scipy.cluster.vq.kmeans](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html)

from task1 import scatter_clusters, ClusterError, plotter
from task2 import kmeans_cluster_assignment
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import time
import resource

## TASK 2 REQUIRED VARIABLES START
plt.title("Easy problem")
ez_clusters = scatter_clusters(centers=[(3,5), (7, 8), (1,7)],
                               spread=[(1.2,1.3), (1.5, 1.4), (1, 1)],
                               n_points=50)
plt.title("Hard problem")
disasterous = scatter_clusters(centers=[(3.5,4.5), (2.5, 4), (4,5)],
                               spread=[(0.8,0.9), (0.9, 1.1), (0.5, 0.7)],
                               n_points=50)                               
# turn clusters into a bunch of points
def extract_values(cluster):
    points = []
    for group_of_points in cluster.values():
        points.extend(group_of_points)
    return points

ez_points_only = extract_values(ez_clusters)
hard_points_only = extract_values(disasterous)

## TASK 2 REQUIRED VARIABLES END

start_time = time.time()
smart_algo_hardpoints = kmeans(obs=hard_points_only,
                               k_or_guess=3)
print("scipy algorithm ran in %s seconds" % (time.time() - start_time))


start_time = time.time()
homemade_kmeans_hardpoints = kmeans_cluster_assignment(k=3,
                                                       points=hard_points_only,
                                                       visualize=0)
print("My implementation ran in %s seconds" % (time.time() - start_time))

# SCIPY IMPLEMENTATION
# centers
plt.title("SCIPY, HARD")
plt.scatter(x=[p[0] for p in smart_algo_hardpoints[0]],
            y=[p[1] for p in smart_algo_hardpoints[0]],
            s=200,
            alpha=0.5)
# points
plt.scatter(x=[point[0] for point in hard_points_only],
            y=[point[1] for point in hard_points_only])
plt.show()
# MY IMPLEMENTATION
# centers
plt.title("DAUME, HARD")
plt.scatter(x=[p[0] for p in homemade_kmeans_hardpoints],
            y=[p[1] for p in homemade_kmeans_hardpoints],
            s=200,
            alpha=0.5)
# points
plt.scatter(x=[point[0] for point in hard_points_only],
            y=[point[1] for point in hard_points_only])
plt.show()
plotter(disasterous, "THE ORIGINAL DISTRIBUTION BETWEEN CLUSTERS")

## EASY PROBLEM
start_time = time.time()
smart_algo_ez= kmeans(obs=ez_points_only,
                      k_or_guess=3)
print("scipy algorithm ran in %s seconds" % (time.time() - start_time))


start_time = time.time()
homemade_kmeans_ez = kmeans_cluster_assignment(k=3,
                                            points=ez_points_only,
                                            visualize=0)
print("My implementation ran in %s seconds" % (time.time() - start_time))


plt.title("SCIPY, EASY")
# centers
plt.scatter(x=[p[0] for p in smart_algo_ez[0]],
            y=[p[1] for p in smart_algo_ez[0]],
            s=200,
            alpha=0.5)
# points
plt.scatter(x=[point[0] for point in ez_points_only],
            y=[point[1] for point in ez_points_only])
plt.show()
# centers
plt.title("DAUME, EASY")
plt.scatter(x=[p[0] for p in homemade_kmeans_ez],
            y=[p[1] for p in homemade_kmeans_ez],
            s=200,
            alpha=0.5)
# points
plt.scatter(x=[point[0] for point in ez_points_only],
            y=[point[1] for point in ez_points_only])
plt.show()
plotter(ez_clusters,"THE ORIGINAL DISTRIBUTION BETWEEN CLUSTERS")


# Scipy algorithm is expectedly faster and surely works better (having been made by more experienced and educated people). For this task it showed results similar to my implementation, at least on low dimensionality.
# 
# Interesting thing with the easy clusters: if points are distributed correctly, both implementations look similar. However, my implementation rarely assigns all point in 2 of the 3 clusters to one centroid and splits another cluster between two points. Scipy's implementation never resulted in such a false distribution.

