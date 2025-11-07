import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.linalg import null_space, sqrtm

colors = plt.cm.tab20.colors

def main():

    ##### ACQUISITION OF THE CIRCLE DATASET #####
    circle_df = pd.read_csv("Circle.csv", names=["x","y"])
    circle_sm = sim_matrix(circle_df.values)

    ##### ACQUISITION OF THE SPIRAL DATASET #####
    spiral_df = pd.read_csv("Spiral.csv", names=["x","y","labels"])
    spiral_sm = sim_matrix(spiral_df.values)

    ##### GENERATION OF THE 3D DATASET #####
    my_dataset = lines_generator(min_ax=-5, max_ax=5, n_items=10, points_per_line=100, noise=0.1)
    my_df = pd.DataFrame(my_dataset,columns=["x","y","z"])
    my_Sm = sim_matrix(my_dataset)

    #EXAMPLE OF ANALYSIS BLOCK TO BE UNCOMMENTED
    """ # EXAMPLE OF ANALYSIS BLOCK
    # UNCOMMENT ONLY WHEN THIS BLOCK IS NEEDED 
    # ... """

    ##### PLOTTING OF THE CIRCLE DATASET #####
    plt.figure()
    plt.scatter(circle_df["x"],circle_df["y"])
    plt.title("Circle dataset")
    plt.show()

    ##### PLOTTING OF THE SPIRAL DATASET #####
    plt.figure()
    plt.scatter(spiral_df["x"],spiral_df["y"])
    plt.title("Spiral dataset")
    plt.show()

    ##### CIRCLE DATASET ANALYSIS WITH 10 NEAREST NEIGHBORS GRAPH #####
    L_circle_10 = dataset_analysis(circle_df.values, circle_sm, 10, "Circle")
    m = 10
    eigenvalues = eigenvalues_analysis(L_circle_10, m)
    k = [2,3]
    pipeline(L_circle_10,k[0],eigenvalues[:k[0]],circle_df)
    pipeline(L_circle_10,k[1],eigenvalues[:k[1]],circle_df)

    ##### CIRCLE DATASET ANALYSIS WITH 20 NEAREST NEIGHBORS GRAPH #####
    L_circle_20 = dataset_analysis(circle_df.values, circle_sm, 20, "Circle")
    m = 10
    eigenvalues = eigenvalues_analysis(L_circle_20, m)
    k = [2,3]
    pipeline(L_circle_20,k[0],eigenvalues[:k[0]],circle_df)
    pipeline(L_circle_20,k[1],eigenvalues[:k[1]],circle_df)

    ##### CIRCLE DATASET ANALYSIS WITH 40 NEAREST NEIGHBORS GRAPH #####
    L_circle_40 = dataset_analysis(circle_df.values, circle_sm, 40, "Circle")
    m = 10
    eigenvalues = eigenvalues_analysis(L_circle_40, m)
    k = 2
    pipeline(L_circle_40,k,eigenvalues[:k],circle_df)

    ##### CIRCLE DATASET K-MEANS CLUSTERING WITH K=2,3 #####
    comparison_with_kmeans(circle_df,k=2)
    comparison_with_kmeans(circle_df,k=3)

    ##### SPIRAL DATASET ANALYSIS WITH 10 NEAREST NEIGHBORS GRAPH #####
    L_spiral_10 = dataset_analysis(spiral_df.values, spiral_sm, 10, "Spiral")
    m = 7
    eigenvalues = eigenvalues_analysis(L_spiral_10, m)
    k = 3
    pipeline(L_spiral_10,k,eigenvalues[:k],spiral_df)

    ##### SPIRAL DATASET ANALYSIS WITH 20 NEAREST NEIGHBORS GRAPH #####
    L_spiral_20 = dataset_analysis(spiral_df.values, spiral_sm, 20, "Spiral")
    m = 7
    eigenvalues = eigenvalues_analysis(L_spiral_20, m)
    k = 3
    pipeline(L_spiral_20,k,eigenvalues[:k],spiral_df)

    ##### SPIRAL DATASET ANALYSIS WITH 40 NEAREST NEIGHBORS GRAPH #####
    L_spiral_40 = dataset_analysis(spiral_df.values, spiral_sm, 40, "Spiral")
    m = 7
    eigenvalues = eigenvalues_analysis(L_spiral_40, m)
    k = 3
    pipeline(L_spiral_40,k,eigenvalues[:k],spiral_df)

    ##### SPIRAL DATASET K-MEANS CLUSTERING WITH K=3 #####
    comparison_with_kmeans(spiral_df,k=3)

    ##### PLOTTING OF CORRECT CLUSTERING FOR SPIRAL DATASET #####
    plot_spiral_labels(spiral_df)

    ##### NORMALIZED CIRCLE DATASET ANALYSIS WITH 10 NEAREST NEIGHBORS GRAPH #####
    L_circle_sym_10 = dataset_analysis_normalized(circle_df.values, circle_sm, 10, "Circle")
    m = 10
    eigenvalues = eigenvalues_analysis(L_circle_sym_10, m)
    k = [2,3]
    pipeline(L_circle_sym_10,k[0],eigenvalues[:k[0]],circle_df)
    pipeline(L_circle_sym_10,k[1],eigenvalues[:k[1]],circle_df)

    ##### NORMALIZED CIRCLE DATASET ANALYSIS WITH 20 NEAREST NEIGHBORS GRAPH #####
    L_circle_sym_20 = dataset_analysis_normalized(circle_df.values, circle_sm, 20, "Circle")
    m = 10
    eigenvalues = eigenvalues_analysis(L_circle_sym_20, m)
    k = [2,3]
    pipeline(L_circle_sym_20,k[0],eigenvalues[:k[0]],circle_df)
    pipeline(L_circle_sym_20,k[1],eigenvalues[:k[1]],circle_df)

    ##### NORMALIZED CIRCLE DATASET ANALYSIS WITH 40 NEAREST NEIGHBORS GRAPH #####
    L_circle_sym_40 = dataset_analysis_normalized(circle_df.values, circle_sm, 40, "Circle")
    m = 10
    eigenvalues = eigenvalues_analysis(L_circle_sym_40, m)
    k = 2
    pipeline(L_circle_sym_40,k,eigenvalues[:k],circle_df)

    ##### NORMALIZED SPIRAL DATASET ANALYSIS WITH 10 NEAREST NEIGHBORS GRAPH #####
    L_spiral_sym_10 = dataset_analysis_normalized(spiral_df.values, spiral_sm, 10, "Spiral")
    m = 7
    eigenvalues = eigenvalues_analysis(L_spiral_sym_10, m)
    k = 3
    pipeline(L_spiral_sym_10,k,eigenvalues[:k],spiral_df)

    ##### NORMALIZED SPIRAL DATASET ANALYSIS WITH 20 NEAREST NEIGHBORS GRAPH #####
    L_spiral_sym_20 = dataset_analysis_normalized(spiral_df.values, spiral_sm, 20, "Spiral")
    m = 7
    eigenvalues = eigenvalues_analysis(L_spiral_sym_20, m)
    k = 3
    pipeline(L_spiral_sym_20,k,eigenvalues[:k],spiral_df)

    ##### NORMALIZED SPIRAL DATASET ANALYSIS WITH 40 NEAREST NEIGHBORS GRAPH #####
    L_spiral_sym_40 = dataset_analysis_normalized(spiral_df.values, spiral_sm, 40, "Spiral")
    m = 7
    eigenvalues = eigenvalues_analysis(L_spiral_sym_40, m)
    k = 3
    pipeline(L_spiral_sym_40,k,eigenvalues[:k],spiral_df)

    ##### PLOTTING OF THE 3D GENERATED DATASET #####
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(my_dataset[:,0], my_dataset[:,1], my_dataset[:,2])
    plt.title("3D dataset")
    plt.show()

    ##### GENERATED DATASET ANALYSIS WITH 10 NEAREST NEIGHBORS GRAPH #####
    L_lines = dataset_analysis(my_dataset,my_Sm,10,"My lines")
    m = 20
    eigenvalues = eigenvalues_analysis(L_lines, m)
    k = 10
    pipeline_3D(L_lines,k,eigenvalues[:k],my_df)

    ##### GENERATED DATASET K-MEANS CLUSTERING WITH K=10 #####
    comparison_with_kmeans_3D(my_df,k=10)

    ##### GENERATED NORMALIZED DATASET ANALYSIS WITH 10 NEAREST NEIGHBORS GRAPH #####
    L_lines_sym = dataset_analysis_normalized(my_dataset, my_Sm, 10, "My lines")
    m = 20
    eigenvalues = eigenvalues_analysis(L_lines_sym, m)
    k = 10
    pipeline_3D(L_lines_sym,k,eigenvalues[:k],my_df)


    
def sim_matrix(points):
    """
    Builds the similarity matrix of the points of a dataset based on a similarity function.

    Parameters:
        points (numpy.ndarray): The dataset of points.

    Returns:
        numpy.ndarray: The similarity matrix of the dataset.
    """
    n = points.shape[0]
    sm = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            sim_ij = similarity_function(points[i], points[j], 1)
            sm[i,j] = sim_ij
            sm[j,i] = sim_ij
    return sm

def similarity_function(p,q,sigma):
    return np.exp(-np.square(np.linalg.norm(p-q))/(2*np.square(sigma)))

def count_connected_components(graph):
    """
    Counts the number of connected component in an adjacency graph.

    Parameters:
        graph (list[dict]): The adjacency graph on which perform the count of connected components.

    Returns:
        int: The number of connected components.
    """
    visited = set()
    i = 0
    connected_components = 0
    while len(visited) != len(graph):
        if i not in visited:
            curr = dfs(graph, i, set())
            connected_components += 1
            visited.update(curr)
        i += 1
    return connected_components

def dfs(graph, vertex, visited):
    """
    Performs the depth first search on a graph.

    Parameters:
        graph (list[dict]): The adjacency graph on which perform the dfs.
        vertex (int): The current vertex in the recursion.
        visited (set): The set of already visited vertices.

    Returns:
        set: The set of vertices reachable from a starting vertex with a Depth First Search.
    """
    visited.add(vertex)
    for adj in graph[vertex]["adj_vertices"]:
        if adj not in visited:
            dfs(graph, adj, visited)
    return visited

def inv_power_method(A, p, v, max_iter=1000, tol=1e-8):
    """
    Inverse Power Method implementation to compute an eigenpair of a matrix.

    Parameters:
        A (numpy.ndarray): The matrix whose eigenpair has to be computed.
        p (float): An initial guess for an eigenvalue of the matrix.
        v (numpy.ndarray): An initial guess for an eigenvector of the matrix.
        max_iter (int): The maximum number of iterations performed.
        tol (float): The precision to be used to compute the eigenpair.

    Returns:
        numpy.float64: The closest eigenvalue to p of the matrix A.
        numpy.ndarray: An eigenvector of the matrix A.
    """
    n = A.shape[0]    
    try:
        A_pI = np.linalg.inv(A-p*np.eye(n))
    except np.linalg.LinAlgError:
        print(f"p={p} is an eigenvalue for the input matrix.")
        return np.float64(p)
    v = v / np.linalg.norm(v)
    prev_mu = np.inf
    for i in range(max_iter):
        v_plus1 = A_pI @ v
        mu = v_plus1 @ v  
        v_plus1 = v_plus1 / np.linalg.norm(v_plus1)
        if abs(mu - prev_mu) < tol:
            break
        prev_mu = mu
        v = v_plus1
    eigenvalue = 1/mu + p
    if abs(eigenvalue) < tol:
        eigenvalue = np.float64(0)
    eigenvector = v_plus1
    return eigenvalue, eigenvector

def deflation_method(A, num_eigenvalues, tol=1e-8):
    """
    Deflation Method implementation to compute the m smallest eigenvalues.

    Parameters:
        A (numpy.ndarray): The matrix whose eigenvalues have to be computed.
        num_eigenvalues (int): The number of eigenvalues to be computed.
        tol (float): The precision to be used in the Inverse Power Method.

    Returns:
        numpy.ndarray: The array containing the num_eigenvalues computed.
    """
    eigvals = []
    for i in range(num_eigenvalues):
        n = A.shape[0]
        li, xi = inv_power_method(A, 0, np.random.rand(n),tol=tol)
        e1 = np.eye(1, n, 0)
        P = np.eye(n)-(2*np.outer(xi+e1,xi+e1))/np.square(np.linalg.norm(xi+e1))
        B = P @ A @ P
        eigvals.append(li)
        A=B[1:,1:]
    return np.array(eigvals)

def compute_eigenvectors(A, eigenvalues, tol=1e-8):
    """
    Computes the eigenvectors corresponding to the eigenvalues of a given matrix.

    Parameters:
        A (numpy.ndarray): The matrix whose eigenvectors have to be computed.
        eigenvalues (numpy.ndarray): The eigenvalues to the matrix A.
        tol (float): The precision to be used.

    Returns:
        numpy.ndarray: The array containing the eigenvalues computed.
    """
    eigenvectors = []
    prev = np.inf
    for i in range(len(eigenvalues)):
        if np.abs(eigenvalues[i]-prev)>tol:
            v_space = null_space(A-eigenvalues[i]*np.eye(A.shape[0]),rcond=tol).T
            dim = v_space.shape[0]
            for i in range(dim):
                eigenvectors.append(v_space[i])
        prev = eigenvalues[i]
    return np.array(eigenvectors).T

def dataset_analysis(points, Sm, knn, name):
    """
    Performs an analysis on a dataset, doing the following operations:
    -creates a knn graph and its adjacency matrix
    -creates the degree matrix
    -creates the Laplacian matrix
    -counts the number of connected components

    Parameters:
        points (numpy.ndarray): The dataset of the points.
        Sm (numpy.ndarray): The similarity matrix of the points in the dataset.
        knn (int): The number of nearest neighbors for each point.
        name (string): The name of the dataset (only to print the number of connected components).
        
    Returns:
        numpy.ndarray: The Laplacian matrix.
    """
    knn_graph = KnnGraph(knn)
    graph = knn_graph.build_graph(points, Sm)
    W = knn_graph.build_adj_mat()
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    print(f"{name} dataset with {knn} nearest neighbors has {count_connected_components(graph)} connected components")
    return L

def eigenvalues_analysis(L, m):
    """
    Computes the m smallest eigenvalues of L and plots them.

    Parameters:
        L (numpy.ndarray): The Laplacian matrix to be used.
        m (int): The number of eigenvalues to be computed.
    """
    eigenvalues = deflation_method(L,m)
    plt.figure()
    plt.bar(range(1,m+1), eigenvalues)
    plt.xticks(range(1,m+1))
    plt.title(f"{m} smallest eigenvalues of L")
    plt.show()
    return eigenvalues

def pipeline(L, k, eigenvalues, df):
    """
    Does one after the other the following operations:
    -computes U, the matrix of eigenvectors of L;
    -does the K-means clustering on U;
    -plots the points in the dataframe according to the labels provided by K-means in a 2D space.

    Parameters:
        L (numpy.ndarray): The Laplacian matrix to be used.
        k (int): The number of clusters to be computed.
        eigenvalues (numpy.ndarray): The vector containing the k smallest eigenvalues
        df (Pandas.DataFrame): The dataframe containing the points to be plotted.
    """
    U = compute_eigenvectors(L,eigenvalues)
    km = KMeans(k)
    km.fit(U)
    plot_colors = [colors[i] for i in km.labels_]
    plt.figure()
    plt.scatter(df["x"], df["y"], c=plot_colors)
    plt.title(f"Spectral clustering with {k} clusters")
    plt.show()

def comparison_with_kmeans(df, k):
    """
    Plots the K-means clustering of a dataframe.

    Parameters:
        df (Pandas.Dataframe): The dataframe containing the points to be plotted.
        k (int): The number of clusters to be computed.
    """
    km = KMeans(k)
    km.fit(df[["x","y"]].values)
    plot_colors = [colors[i] for i in km.labels_]
    plt.figure()
    plt.scatter(df["x"], df["y"], c=plot_colors)
    plt.title(f"K-means clustering with {k} clusters")
    plt.show()

def plot_spiral_labels(spiral_df):
    """
    Plots the spiral dataset clustering according to the labels contained in the dataframe.

    Parameters:
        spiral_df (Pandas.Dataframe): The spiral dataframe containing the points to be plotted and their labels.
    """
    plot_colors = [colors[i] for i in spiral_df["labels"]]
    plt.figure()
    plt.scatter(spiral_df["x"], spiral_df["y"], c=plot_colors)
    plt.title("Correct clustering for Spiral dataset")
    plt.show()

def dataset_analysis_normalized(points, Sm, knn, name):
    """
    Performs an analysis on a normalized dataset, doing the following operations:
    -creates a knn graph and its adjacency matrix
    -creates the degree matrix and normalizes it
    -creates the normalized Laplacian matrix
    -counts the number of connected components

    Parameters:
        points (numpy.ndarray): The dataset of the points.
        Sm (numpy.ndarray): The similarity matrix of the points in the dataset.
        knn (int): The number of nearest neighbors for each point.
        name (string): The name of the dataset (only to print the number of connected components).
        
    Returns:
        numpy.ndarray: The normalized Laplacian matrix.
    """
    knn_graph = KnnGraph(knn)
    graph = knn_graph.build_graph(points, Sm)
    W = knn_graph.build_adj_mat()
    D = np.diag(np.sum(W, axis=1))
    D_norm = sqrtm(D)
    D_norm = np.linalg.inv(D_norm)
    I = np.eye(W.shape[0])
    L_sym = I - (D_norm @ W @ D_norm)
    print(f"{name} dataset with {knn} nearest neighbors has {count_connected_components(graph)} connected components")
    return L_sym

def lines_generator(min_ax,max_ax,n_items,points_per_line,noise):
    """
    Creates a dataset containing n_items lines, represented by points with some noise.

    Parameters:
        min_ax (int): The lowest value on an axis where we can have a starting point for a line.
        max_ax (int): The highest value on an axis where we can have a starting point for a line.
        n_items (int): The number of lines to be generated.
        points_per_line (int): The number of points representing a line.
        noise (float): The standard deviation of a single point from the main line.

    Returns:
        numpy.ndarray: The dataset with n_items lines shaped figures.
    """
    np.random.seed(338658)
    data = []
    for i in range(n_items):
        start = np.random.uniform(min_ax,max_ax,3)
        dir = np.random.uniform(min_ax,max_ax,3)
        dir = dir/np.linalg.norm(dir)
        length = np.abs(max_ax-min_ax) * np.random.uniform(0.2,0.8)
        for p in np.linspace(0,length,points_per_line):
            data.append(start+p*dir +np.random.normal(0,noise,3))
    return np.array(data)

def pipeline_3D(L, k, eigenvalues, df):
    """
    Does one after the other the following operations:
    -computes U, the matrix of eigenvectors of L;
    -does the K-means clustering on U;
    -plots the points in the dataframe according to the labels provided by K-means in a 3D space.

    Parameters:
        L (numpy.ndarray): The Laplacian matrix to be used.
        k (int): The number of clusters to be computed.
        eigenvalues (numpy.ndarray): The vector containing the k smallest eigenvalues.
        df (Pandas.DataFrame): The dataframe containing the points to be plotted.
    """
    U = compute_eigenvectors(L,eigenvalues)
    km = KMeans(k)
    km.fit(U)
    plot_colors = [colors[i] for i in km.labels_]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["x"],df["y"],df["z"], c=plot_colors)
    plt.title(f"Spectral clustering with {k} clusters")
    plt.show()

def comparison_with_kmeans_3D(df, k):
    """
    Plots the 3D K-means clustering of a dataframe.

    Parameters:
        df (Pandas.Dataframe): The dataframe containing the points to be plotted.
        k (int): The number of clusters to be computed.
    """
    km = KMeans(k)
    km.fit(df[["x","y","z"]].values)
    plot_colors = [colors[i] for i in km.labels_]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["x"],df["y"],df["z"], c=plot_colors)
    plt.title(f"K-means clustering with {k} clusters")
    plt.show()

class KnnGraph:

    def __init__(self,k):
        self.k = k

    def build_graph(self, points, similarities):
        """
        Builds an adjacency graph based on their similarities.

        Parameters:
            points (numpy.ndarray): The dataset of points.
            similarities (numpy.ndarray): The similarity matrix of the dataset.

        Returns:
            list[dict]: The adjacency graph with k nearest neighbors.
        """
        n = points.shape[0]
        self.graph = [{} for i in range(n)]
        for i in range(n):
            indices = np.argsort(similarities[i])[:n-self.k-1:-1]
            self.graph[i]["adj_vertices"] = indices
            self.graph[i]["edges"] = similarities[i][indices]
        for i in range(n):
            for j in self.graph[i]["adj_vertices"]:
                if i not in self.graph[j]["adj_vertices"]:
                    self.graph[j]["adj_vertices"] = np.append(self.graph[j]["adj_vertices"], i)
                    self.graph[j]["edges"] = np.append(self.graph[j]["edges"], similarities[i][j])            
        return self.graph
    
    def build_adj_mat(self):
        """
        Builds the weighted adjacency matrix corresponding to an adjacency graph.

        Returns:
            numpy.ndarray: The adjacency matrix W.
        """
        n = len(self.graph)
        self.adj_mat = np.zeros((n,n))
        for i in range(n):
            self.adj_mat[i][self.graph[i]["adj_vertices"]] = np.array(self.graph[i]["edges"])
        return self.adj_mat
   
main()