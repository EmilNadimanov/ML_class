# ## Task 1

# Write a clustering problem generator with signature:
# ```
#     def scatter_clusters(
#       centers: ArrayLike,
#       spread: ArrayLike,
#       n_points: int
#     ) -> ArrayLike:
# ```
# For k=3, generate easy and hard problems and plot them; the easy problem might look like figure 3.13 from Daumé.



import numpy as np
import matplotlib.pyplot as plt


from datetime import datetime as dt


class ClusterError(Exception):
    pass


np.random.seed(int(dt.now().timestamp()))


# auxiliary function
def random_point(
    center,
    spread
) -> tuple:
    base_x, base_y = center[0], center[1]
    point_x = base_x + np.random.uniform(low=-spread[0], high=spread[0])
    point_y = base_y + np.random.uniform(low=-spread[1], high=spread[1])
    return (point_x, point_y)


# assuming dimensionality of 2
def scatter_clusters(
    centers: list, # k centers (x, y)
    spread: list, # k pairs of spread values, one for each dimension for center 
    n_points: int # n points to randomly assign to each of the centers
) -> list:
    clusters = dict()   
    for (center, sp) in zip(centers, spread):
        points = [random_point(center, sp) for iteration in range(n_points)]
        clusters[center] = points
    return clusters


ez_clusters = scatter_clusters(centers=[(3,5), (7, 8), (1,7)],
                               spread=[(1.2,1.3), (1.5, 1.4), (1, 1)],
                               n_points=50)

disasterous = scatter_clusters(centers=[(3.5,4.5), (2.5, 4), (4,5)],
                               spread=[(0.8,0.9), (0.9, 1.1), (0.5, 0.7)],
                               n_points=50)

def plotter(clusters, name=""):
    for center, points in clusters.items():
        plt.scatter(x=[point[0] for point in points],
                    y=[point[1] for point in points])
        plt.scatter(center[0],center[1], c='black', s=200, alpha=0.5)
    plt.title(name);
    plt.show()


plotter(ez_clusters, 'easy clusters')



plotter(disasterous, 'hard clusters')



