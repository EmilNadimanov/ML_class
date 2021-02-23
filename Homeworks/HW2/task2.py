# ## Task 2

# Implement K-means clustering as shown in Daumé:
# ```
# def kmeans_cluster_assignment(
#   k: int,
#   points: ArrayLike,
#   centers_guess: Optional[ArrayLike] = None,
#   max_iterations: Optional[int] = None,
#   tolerance: Optional[float] = None
# ) -> ArrayLike:
# ```
# Replot your problems at 5 stages (random initialisation, 25%, 50%, 75%, 100% of iterations), using colours to assign points to clusters. \
# The easy problem plots might look like the coloured plots in figure 3.14 from Daumé.
# 

from task1 import scatter_clusters, ClusterError
import numpy as np
import matplotlib.pyplot as plt

ez_clusters = scatter_clusters(centers=[(3,5), (7, 8), (1,7)],
                               spread=[(1.2,1.3), (1.5, 1.4), (1, 1)],
                               n_points=50)

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


# auxilliary functions
def closest_cluster(cluster_guesses, point):
    distances = np.array([])
    for guess in cluster_guesses:
        distances = np.append(distances, np.linalg.norm(guess-point))
    return tuple(cluster_guesses[np.argmin(distances)])

def plot_the_whole_thing(states):
    init = 0 # initialization
    total = len(states.keys()) # number of iterations, 100%
    tw_five = total // 4  # 25%
    fifty = total // 2 # 50%
    sev_five = round(total / 4 * 3) # 75%
    
    k = len(states[init][1]) # how many centroids we have
    colors = ('b', 'g', 'r', 'c', 'm', 'y')[:k] # I hope  we won't have >6 clusters 
    for each in [init, tw_five, fifty, sev_five, total-1]:
        current = states[each]
        assignment = current[0]
        centers = current[1]
        i = 0
        for center in centers:
            color = colors[i]
            points = assignment[tuple(center)]
            plt.scatter(x=center[0],
                        y=center[1],
                        c=color,
                        s=200,
                        alpha=0.5)
            plt.scatter(x=[point[0] for point in points],
                        y=[point[1] for point in points],
                        c=color)
            i += 1
        plt.title("Iteration %i" % each)
        plt.show()
        plt.pause(0.1)


def kmeans_cluster_assignment(
    k: int,
    points: list,
    centers_guess: list = None,
    max_iterations: int = None,
    tolerance: float = 0.001,
    visualize = False
) -> list:
    iteration=0
    state = {} # save the progress on each iteration to plot it at the end of the function
    max_x, max_y = max([p[0] for p in points]), max([p[1] for p in points])
    min_x, min_y = min([p[0] for p in points]), min([p[1] for p in points])
    
    if centers_guess is None:
        mu = []
        for i in range(k):
            rand_x = np.random.uniform(low=min_x, high=max_x)
            rand_y = np.random.uniform(low=min_y, high=max_y)
            mu += [(rand_x, rand_y)]
        mu = np.array(mu)
    else:
        mu = centers_guess
    
    while((max_iterations is not None and iteration < max_iterations) or (max_iterations is None)):
        old_mu = np.array(mu)
        closest_assignment = {}  # each point is assigned a closest cluster center
                           # a dict (center -> list(points)) is formed      
        try:
            for point in points:
                clust = closest_cluster(mu, point)
                closest_assignment[clust] = closest_assignment.get(clust, list())
                closest_assignment[clust].append(point)
            for i in range(len(mu)):
                closest_points = closest_assignment[tuple(mu[i])]
                mean_x = np.mean([p[0] for p in closest_points])
                mean_y = np.mean([p[1] for p in closest_points])
                mu[i] = np.array([mean_x, mean_y])
        except (KeyError, TypeError) as e:
            raise ClusterError("Not every centroid received the points it needed. Please consider rescattering or try again.") from None
        state[iteration] = (closest_assignment, old_mu) # centers and points' assignment to them
        if np.linalg.norm(mu - old_mu) < tolerance:
            break
        iteration += 1
    if visualize:
        plot_the_whole_thing(state)
    print(f"Completed in {iteration+1} iterations")
    return closest_assignment


# In[15]:

print("Easy problem")
ez_clustering = kmeans_cluster_assignment(k=3,
                                          points=ez_points_only,
                                          visualize=1)


# In[16]:

print("Hard problem")
hard_clustering = kmeans_cluster_assignment(k=3,
                                            points=hard_points_only,
                                            visualize=1)



