import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(0)

def main():

    #EXAMPLE OF ANALYSIS BLOCK TO BE UNCOMMENTED
    """ # EXAMPLE OF ANALYSIS BLOCK
    # UNCOMMENT ONLY WHEN THIS BLOCK IS NEEDED 
    # ... """

    # TOY DATAFRAME GENERATION
    toy_df = gen_toy_df()
    print(toy_df)

    # RANDOM RATINGS DATAFRAME GENERATION
    ratings_df = generate_dataframe(num_users=100,num_movies=35)

    # TOY DATAFRAME SVD ANALYSIS
    toy_users, toy_films = svd_analysis(toy_df)
    print(f"Recommendations for movie 2 in the toy dataframe: {recommendation(2,toy_films)[:4]}") # best 4 recommendations given movie 2
    print(np.linalg.norm(toy_df[2]-toy_df[3]))
    print(np.linalg.norm(toy_df[2]-toy_df[4]))
    print(np.linalg.norm(toy_df[2]-toy_df[0]))
    print(np.linalg.norm(toy_df[2]-toy_df[1]))

    # RANDOM RATINGS DATAFRAME ANALYSIS (BEST AND WORST)
    users, films = svd_analysis(ratings_df)
    print(f"Best 5 recommendations for movie 0 in the random dataframe: {recommendation(0,films)[:5]}") #BEST 5 RECOMMENDATIONS
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_19"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_28"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_1"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_4"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_24"]))
    print(f"Worst 5 recommendations for movie 0 in the random dataframe: {recommendation(0,films)[-5:]}") #WORST 5 RECOMMENDATIONS
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_27"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_20"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_12"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_33"]))
    print(np.linalg.norm(ratings_df["Movie_0"]-ratings_df["Movie_29"]))

    # RANDOM RATINGS COMPARISON WITH PCA
    r = [19, 28, 1, 4, 24] #RECOMMENDED LABELS
    notr = [27, 20, 12, 33, 29] #NOT RECOMMENDED LABELS
    pca = PCA(3)
    Y = pca.fit_transform(ratings_df.values.T)
    sg_3d = plt.figure()
    ax_sg_3d = sg_3d.add_subplot(111, projection='3d')
    ax_sg_3d.text(Y[0, 0], Y[0, 1], Y[0, 2], str(0), fontsize=12)
    for i in r:
        ax_sg_3d.text(Y[i, 0], Y[i, 1], Y[i, 2], str(i), fontsize=12, color="g")
    for i in notr:
        ax_sg_3d.text(Y[i, 0], Y[i, 1], Y[i, 2], str(i), fontsize=12, color="r")
    ax_sg_3d.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
    plt.grid()
    plt.show()

    # USERS RECOMMENDATION ON TOY DATAFRAME
    print(f"Recommendations for user 0 in toy dataframe: {recommendation(0,toy_users)[:3]}")
    print(np.linalg.norm(toy_df.iloc[0]-toy_df.iloc[7]))
    print(np.linalg.norm(toy_df.iloc[0]-toy_df.iloc[5]))
    print(np.linalg.norm(toy_df.iloc[0]-toy_df.iloc[9]))

def bidiagonalize(A, tol=1e-8):
    """
    Perform bidiagonalization of a matrix A using Householder transformations.
    
    Parameters:
    - A (np.ndarray): Input matrix (m x n) with m>=n to be bidiagonalized.
    - tol (float): Tolerance for setting small values in B to zero.
    
    Returns:
    - (np.ndarray): Bidiagonal matrix.
    - (np.ndarray): Left orthogonal transformation matrix.
    - (np.ndarray): Right orthogonal transformation matrix.
    """
    m, n = A.shape
    if m<n: 
        print("Please give a matrix with more rows than columns")
        return
    B = A.copy()
    P = np.eye(m)
    H = np.eye(n)
    for i in range(n):
        k = i+1
        a = B[i:, i]
        a = a + np.sign(a[0]) * np.linalg.norm(a) * np.eye(1, a.shape[0], 0).flatten()
        Pbar = np.eye(m)
        Pbar[i:, i:] -= 2 * np.outer(a, a) / np.square(np.linalg.norm(a))
        B = Pbar @ B
        P = P @ Pbar
        if k <= n - 2:
            b = B[i, i+1:]
            b = b + np.sign(b[0]) * np.linalg.norm(b) * np.eye(1, b.shape[0], 0).flatten()
            Hbar = np.eye(n)
            Hbar[i+1:, i+1:] -= 2 * np.outer(b, b) / np.square(np.linalg.norm(b))
            B = B @ Hbar
            H = H @ Hbar
    B[np.abs(B) < tol] = 0.0
    return B, P, H

def svd(A, tol=1e-8):
    """
    Computes the SVD of the matrix A.

    Parameters:
        A (np.ndarray): The matrix whose SVD is to be computed

    Returns:
        np.ndarray: The matrix of left singular vectors U.
        np.ndarray: The array of singular values.
        np.ndarray: The matrix of RIGHT singular vectors v.
    """
    B,_,H = bidiagonalize(A,tol)
    eigvals, eigvects = np.linalg.eigh(B.T@B)
    sorted_indices = np.argsort(eigvals)[::-1]
    Qbar = eigvects[:,sorted_indices]
    Q = H @ Qbar
    C = A @ Q
    U,R = np.linalg.qr(C)
    V = Q
    sign_diag = np.sign(np.diag(R))
    R = R * sign_diag
    U = U * sign_diag
    R[np.abs(R)<tol] = 0
    return U, np.abs(np.diag(R)), V.T

def svd_analysis(ratings_df):
    """
    Performs the analysis of the singular values of the dataframe.

    Parameters:
        ratings_df (pd.DataFrame): The dataframe containing all the films.

    Returns:
        pd.DataFrame: users representation (matrix U).
        pd.DataFrame: films representation (matrix V).
    """
    U,S,Vh = svd(ratings_df.values)
    plt.figure()
    plt.plot(S)
    plt.xticks(range(len(S)),labels=range(1,len(S)+1))
    plt.axvline(2)
    plt.show()
    k = 3 # to be customized for each analysis
    users = pd.DataFrame(U[:,:k])
    films = pd.DataFrame(Vh[:k,:].T)
    return users, films

def recommendation(comparison, films_df):
    """
    Computes the recommendations for a similar film in the film dataframe, given a comparison film.

    Parameters:
        comparison (int): The film to be compared.
        films_df (pd.DataFrame): The dataframe whose rows are the representation of all the films obtained via SVD.

    Returns:
        list: ordered recommended films (most recommended first).
    """
    global rec
    rec = []
    for i in range(len(films_df.index)):
        if i != comparison:
            rec.append({"film":i, "dis":np.linalg.norm(films_df.iloc[comparison]-films_df.iloc[i])})
    final_rec = [f["film"] for f in sorted(rec, key=lambda x:x["dis"])]
    return final_rec

def generate_dataframe(num_users, num_movies):
    """
    Generates a DataFrame where rows are users, columns are movies, and values are 1 if the user has watched the film,
    and 0 if not. Each user watches a random number of movies, up to half the total number of movies.

    Parameters:
        num_users (int): Number of users.
        num_movies (int): Number of movies.

    Returns:
        pd.DataFrame: User-movie binary DataFrame.
    """
    movie_columns = [f"Movie_{i}" for i in range(num_movies)]
    user_rows = [f"User_{i}" for i in range(num_users)]
    df = pd.DataFrame(0, index=user_rows, columns=movie_columns, dtype=int)
    for user in user_rows:
        num_watched = np.random.randint(1, int(float(num_movies/2)) + 1)
        watched_movies = np.random.choice(movie_columns, num_watched, replace=False)
        df.loc[user, watched_movies] = 1
    return df

def gen_toy_df():
    """
    Generates a predefined toy dataframe.

    Returns:
        pd.DataFrame: Toy user-movie dataframe.
    """
    mat = np.array([[0,1,0,0,0],
                    [1,0,0,0,1],
                    [0,0,0,0,1],
                    [0,1,1,1,0],
                    [1,0,1,1,1],
                    [1,1,1,1,1],
                    [0,0,1,1,1],
                    [0,1,0,0,0],
                    [0,0,1,1,1],
                    [1,1,0,1,1],])
    return pd.DataFrame(mat)

main()