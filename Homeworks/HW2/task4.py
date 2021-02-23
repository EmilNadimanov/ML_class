# ## Task 4

# Compute the performance of your algorithm as percent of points assigned the correct cluster. (Your algorithm may order the clusters differently!) Graph this as a function of iterations, at 10% intervals. \
# Make a random 10-90 test-train split; now you train on 90% of the data, and evaluate on the other 10%. How does the performance graph change?


from task1 import scatter_clusters, ClusterError, plotter
from task2 import kmeans_cluster_assignment, closest_cluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import time
import resource


# To evaluate "correctness" of the clusters we got from kmeans we need to match them to the original ones somehow. Given two sets of cluster centers: original **O** and presented for evaluation **E** (both of equal size), I see the following way to do it:
#     We say that **E_1 ∈ E** and **O_1 ∈ O** match if for all **E_n ∈ E** out there **E_1** is the closest one (e.g. based on Euclidean distance) to **O_1**.\
#     \
#     Once two centers are linked, we take two sets of points: one, let us call it **P_o**, assigned to **O_n ∈ O**, another assigned to **E_n ∈ E**, and take their intersection **i**. The ratio **i / |P_o|** will be the score for this **E_n**. We do this for every **O_n ∈ O**.\
#     \
#     It is possible for two points from the set **E** to match the same representative of the set **O**. It is not a bad thing to happen, because if points assinged to **O_n ∈ O** are split equally between **E_1** and **E_2**, we will just say that both latter ones are 50% correct, which will reduce the total score of the clusterisation result under evaluation.\
#     \
#     If we take an easy clusterisation example from above and imagine that two clusters are assigned to the same center (resulting in one "big" cluster) and the third one is split between another two, the "big" cluster would bring 100% accuracy for one of the original clusters it includes and 0% for another one, because it llinks only to the closest one.

## TASK 2 REQUIRED VARIABLES START
plt.title("Easy problem")
ez_clusters = scatter_clusters(centers=[(3,5), (7, 8), (1,7)],
                               spread=[(1.2,1.3), (1.5, 1.4), (1, 1)],
                               n_points=50)
plt.title("Hard problem")
hard_clusters = scatter_clusters(centers=[(3.5,4.5), (2.5, 4), (4,5)],
                               spread=[(0.8,0.9), (0.9, 1.1), (0.5, 0.7)],
                               n_points=50)

## TASK 2 REQUIRED VARIABLES END


#auxilliary functions
def get_performance(to_eval, original):
    performance_list = list()
    pairs = list()
    original_centers = np.array(list(original.keys()))
    for each in to_eval:
        coordinates = np.array(each)
        corresponding = list(sorted(original_centers,
                                    key=lambda orig_coord: np.linalg.norm(coordinates - orig_coord)))[0]
        original_center = tuple(corresponding.tolist())
        pairs.append( (each, original_center) )
    for orig_center, orig_points in original.items():
        matching_results = [p[0] for p in pairs if p[1] == orig_center]
        if len(matching_results) == 0:
            performance_list.append(0)
        else:
            intermediate = list()
            for matching in matching_results:
                original_size = len(orig_points)
                matching_cluster_points = to_eval[matching]
                shared_points = set(orig_points) & set(matching_cluster_points)
                intermediate.append(len(shared_points) / original_size)
            performance_list.append(np.mean(np.array(intermediate)))
    return np.mean(np.array(performance_list))

def build_a_graph(original, computed_states):
    pool = list(computed_states.keys())
    states_to_evaluate = list()
    tenpercent = len(pool) / 10
    borderline = 0
    results = list()
    for iteration in pool:
        if iteration >= borderline or iteration==pool[-1]:
            borderline += tenpercent
            states_to_evaluate.append(iteration)
    for i in states_to_evaluate:
        state_to_evaluate = computed_states[i]
        performance = get_performance(state_to_evaluate[0], original)
        results.append( (i,performance) )
    plt.plot([i[0] for i in results],
             [i[1] for i in results])
    plt.show()


# In[158]:


def kmeans_with_graph(
    original_clusters: dict,
    centers_guess: list = None,
    max_iterations: int = None,
    tolerance: float = 0.001
) -> list:
    points = list()
    for cluster in original_clusters.values():
        points.extend(cluster)
    k = len(original_clusters.keys())
    ##################################
    ## COPYPASTE FROM TASK 2 BEGINS ##
    ##################################
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
        mu = mp.array(centers_guess)
    
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
    #plot_the_whole_thing(state)
    ################################
    ## COPYPASTE FROM TASK 2 ENDS ##
    ################################
    print(f"Completed in {iteration+1} iterations")
    build_a_graph(original_clusters, state)
    return closest_assignment


plt.title("Performance with easy clustering")
ez_kmeans_with_graph = kmeans_with_graph(original_clusters=ez_clusters)

plt.title("Performance with hard clustering")
hard_kmeans_with_graph = kmeans_with_graph(original_clusters=hard_clusters)
