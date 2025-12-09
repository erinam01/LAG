import numpy as np
from scipy.sparse import coo_array
import matplotlib.pyplot as plt
from scipy.linalg import null_space


FIG21 = np.array([
        [0,1,1,1],
        [0,0,1,1],
        [1,0,0,0],
        [1,0,1,0]
    ], dtype=float)

FIG22 = np.array([
        [0,1,0,0,0],
        [1,0,0,0,0],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,0,1,1,0]
    ], dtype=float)

def power_method(A, v0, shift, max_iter=1000, rel_tol=1e-10):
    # if shift = 0 it's just normal power method A-I
    # finds highest ABSOLUTE VALUE eigenvalue

    # initialization
    v = v0 / np.linalg.norm(v0)
    lambda_old = np.inf
    m = 0

    while m <= max_iter:
        # ltiply by A
        w = A @ v
        
        # normalize new vector
        v = w / np.linalg.norm(w) #normalizing keeps only the direction of the vector
        
        # Rayleigh quotient for eigenvalue approximation
        lambda_new = v @ (A @ v)
        
        # Convergence test:
        # |λ_new - λ_old| <= |λ_new| * rel_tol
        if np.abs(lambda_new - lambda_old) <= np.abs(lambda_new) * rel_tol:
            return lambda_new, v, m
        
        lambda_old = lambda_new
        m += 1

    # If max_iter reached without convergence
    return lambda_new, v, m

def check_dominant_eigenvalue(A, computed_eigenvalue):

    # compute dominant eigenvalue using numpy for verification
    eigvals = np.linalg.eig(A)[0]
    dominant = eigvals[np.argmax(np.abs(eigvals))] #absolute value

    if not np.isclose(dominant, computed_eigenvalue):
        return False
    else:
        return True

def plot_graph_from_adjacency_matrix(adj_matrix):
    # DEBUG
    import networkx as nx

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    plt.figure(figsize=(6,4))
    nx.draw(G, with_labels=True, arrows=True, arrowstyle='->', arrowsize=15)
    plt.show()

def build_google_link_matrix(G, a=0.85):

    """
    print("-------ANALYZING LINKS-------")

    id_to_url = {}
    A = None

    rows = []
    cols = []

    with open("hollins.dat", "r") as f:
    line1 = f.readline().strip()
    n, e = line1.split()
    print(n, e)
    N = int(n)
    E = int(e)
    # number of nodes, e number of edges
    density = (E/(N*(N-1)))
    # density is equal to the number of edges / n(n-1) because no self loops allowed
    print("Density:", density)

    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Case 1: ID → URL
        if len(parts) == 2 and parts[1].startswith("http"):
            node = int(parts[0])
            url = parts[1]
            id_to_url[node] = url

        # Case 2: ID → ID (graph edges)
        elif len(parts) == 2:
            src = int(parts[0])
            dst = int(parts[1])
            rows.append(src-1)
            cols.append(dst-1)

        else:
            print("Skipping unrecognized line:", line)


    data = np.ones(len(rows), dtype=float)

    # Build COO sparse matrix
    G_coo = coo_array((data, (rows, cols)), shape=(N, N)) #easier to build using coo

    # Convert to CSR
    G = G_coo.tocsr() #better for arithmetics --> CSR format
    print("CSR matrix shape:", G.shape)
    print("Number of nonzeros:", G.nnz)

    plt.spy(G)
    plt.title('Sparsity pattern of matrix G')
    plt.show()


    print("-------COMPUTING PAGE RANK-------")
    """

def link_matrix(matrix):
    n = matrix.shape[0]
    L = np.zeros_like(matrix, dtype=float)

    for j in range(n):               # for each column
        in_links = matrix[:, j].sum() #num links in going to j
        out_links = matrix[j, :].sum() #num links going out from j
        #DEBUG
        # print(f"Node {j+1}: L{j+1} = {in_links}, n{j+1} = {out_links}")

        if in_links > 0:
            L[:, j] = matrix[j,:] / out_links
        else:
            L[:, j] = 1/n            # dangling node handling

    return L

def compute_eigenspace(A, eigenvalue, tol=1e-10):
    n = A.shape[0]
    M = A - eigenvalue * np.eye(n)
    eig_space = null_space(M, rcond=tol)
    return eig_space

def exercise_1():

    print("Exercise 1")

    H = link_matrix(FIG21)
    print(H)

    dominant_eigen, dominant_vector = power_method(H, np.ones(H.shape[0]), shift=0)[:2]
    importance_vector = dominant_vector
    if not check_dominant_eigenvalue(H, dominant_eigen) or not np.isclose(dominant_eigen, 1.0):
        print(f"Mistake in computing dominant eigenvalue: {dominant_eigen}")
        return
    print(f"Importance scores: ", importance_vector)

    print("Add page 5, connected and connecting to page 3")
    FIG21_EXTENDED = np.array([
        [0,1,1,1,0],
        [0,0,1,1,0],
        [1,0,0,0,1],
        [1,0,1,0,0],
        [0,0,1,0,0]
    ], dtype=float)

    H_EXTENDED = link_matrix(FIG21_EXTENDED)
    eigenvalues_ext, eigenvectors_ext = np.linalg.eig(H_EXTENDED)
    sum_eigenvector_ext = np.sum(eigenvectors_ext[:,0])
    importance_vector_ext = eigenvectors_ext[:,0]/sum_eigenvector_ext

    print("Importance vector extended:", np.round(importance_vector_ext, 3))

    if importance_vector_ext[2] > importance_vector[2]:
        print("The importance of page 3 has increased after adding page 5.")
        print(f"Old importance: {importance_vector[2]}, New importance: {importance_vector_ext[2]}")
    else:
        print("The importance of page 3 has not increased after adding page 5.")

def exercise_2():
    print("====== EXERCISE 2 ======")
    # V1(A): eigenspace for eigenvalue 1 of A

    #the new web with three or more subwebs is taked from fig 22 to which I add another subweb of two nodes connected only to each other
    num_subwebs = 3
    subweb = np.array([
        [0,1,0,0,0,0,0],
        [1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,1,1,0,0,0],
        [0,0,0,0,0,0,1],
        [0,0,0,0,0,1,0]
    ], dtype=float)

    H = link_matrix(subweb)
    n = H.shape[0]

    dominant_eigenvalue,_, _ = power_method(H, np.ones(n), shift=0)

    if not check_dominant_eigenvalue(H, dominant_eigenvalue):
        print("Mistake in computing dominant eigenvalue:", dominant_eigenvalue)
        # return
    
    eigvals = np.linalg.eigvals(H)
    print("dominant eigenvalues:", eigvals[np.argsort(-np.abs(eigvals))][:5])
    print("the power method will not converge to a unique eigenvector as per paper and theory.")

    eig_space = compute_eigenspace(H, dominant_eigenvalue)

    dim_eig_space = eig_space.shape[1]
    print("Dimension of eigenspace for eigenvalue 1: ", dim_eig_space)
    print("Components of the web: ", num_subwebs)

    if dim_eig_space >= num_subwebs:
        print("The dimension of the eigenspace is equal or greater to the number of components in the web --> disconnected subwebs.")
    else:
        print("The dimension of the eigenspace is less than the number of components in the web.")

def exercise_3():
    print("====== EXERCISE 3 ======")
    
    FIG22_EXTENDED = FIG22.copy()
    FIG22_EXTENDED[4][0] = 1  # Adding a link from page 5 to page 1

    H22 = link_matrix(FIG22)
    n = H22.shape[0]
    H22_EXTENDED = link_matrix(FIG22_EXTENDED)
    n_ext = H22_EXTENDED.shape[0]
    
    v0 = np.ones(n_ext)

    dominant_eigenvalue, _, _ = power_method(H22, v0, shift=0)
    if not check_dominant_eigenvalue(H22, dominant_eigenvalue) or not np.isclose(dominant_eigenvalue, 1.0) :
        print(f"Mistake in computing dominant eigenvalue for FIG22: {dominant_eigenvalue}")
        return
    
    dominant_eigenvalue_ext, _, _ = power_method(H22_EXTENDED, v0, shift=0)
    if not check_dominant_eigenvalue(H22_EXTENDED, dominant_eigenvalue_ext) or not np.isclose(dominant_eigenvalue_ext, 1.0):
        print(f"Mistake in computing dominant eigenvalue for FIG22_EXTENDED: {dominant_eigenvalue_ext}")
        return


    original_eigenspace = compute_eigenspace(H22, dominant_eigenvalue)
    extended_eigenspace = compute_eigenspace(H22_EXTENDED, dominant_eigenvalue_ext)

    dim_original = original_eigenspace.shape[1]
    dim_extended = extended_eigenspace.shape[1]

    print("Dimension of eigenspace for lambda = 1 for original matrix: ", dim_original)
    print("Dimension of eigenspace for lambda = 1 for extended matrix: ", dim_extended)
    if dim_extended >= dim_original:
        print(f"For a graph with with r = 2 disconnected subwebs Dim(V1(A_EXTENDED)) >= r = {dim_extended}")
        print("The dimension of the eigenspace has not changed after adding a link between two subwebs, because the web still contains two disconnected subwebs.")
    else: 
        print("The dimension of the eigenspace has decreased after adding a link between two subwebs.")

def exercise_4():
    print("====== EXERCISE 4 ======")
    H = FIG21.copy()
    H[2][0] = 0  # removing the link from page 3 to page 1
    A = link_matrix(H)
    print("Link matrix A:")
    print(A)
    print("Column 3 has all zeros, indicating that page 3 is a dangling node.")
    print("Columns sum:" , A.sum(axis=0))

    print("From paper: The corresponding substochastic matrix must have a positive eigenvalue λ ≤ 1 "
    "and a corresponding eigenvector x with non-negative entries (called the Perron eigenvector) " \
    "that can be used to rank the web pages.")
    # find eigenvector <= 1
    dominant_eigenvalue, dominant_eigenvector, _ = power_method(A, np.ones(A.shape[0]), shift=0)
    print("Dominant eigenvalue:", dominant_eigenvalue)
    if not dominant_eigenvalue < 1:
        print("Mistake: Dominant eigenvalue is not less than 1.")
        return
    else:
        print("Dominant eigenvalue is less than 1, as expected.")
    print("Dominant eigenvector (Perron eigenvector):", dominant_eigenvector)
    if np.all(dominant_eigenvector >= 0):
        print("All entries in the dominant eigenvector are non-negative, as expected.")
    else:
        print("Mistake: Some entries in the dominant eigenvector are negative.")
        return
    


def main():
    # print("\n\n")
    exercise_1()
    print("\n\n")
    # exercise_2()
    # print("\n\n")
    # exercise_3()
    # print("\n\n")
    # exercise_4()


if __name__ == "__main__":
    main()



# NULL SPACE --> CONTROLLA COME FARE/SE USARE SVD ETC