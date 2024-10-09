import sys

import pandas as pd
from scipy.linalg import lu, cholesky, solve
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numpy import array, zeros, diag, diagflat, dot
from IPython.display import display
from tabulate import tabulate


def read_matrix_from_file(file_path):
    matrix = {}

    with open(file_path, 'r') as file:
        line_number = 0
        for line in file:
            if line_number > 1:
                row, col, value = map(float, line.split())
                matrix[int(row)-1, int(col)-1] = value
            line_number += 1

    return matrix


def create_matrix_from_file(file_path):
    matrix_data = read_matrix_from_file(file_path)

    if not matrix_data:
        print("Matrix data is empty or invalid.")
        return None

    rows = max(row for row, _ in matrix_data.keys()) + 1
    cols = max(col for _, col in matrix_data.keys()) + 1

    result_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for (row, col), value in matrix_data.items():
        result_matrix[row][col] = value

    return result_matrix


def lu_decomposition(matrix):
    P, L, U = lu(matrix)

    '''
    def lu(a, permute_l=False, overwrite_a=False, check_finite=True,
           p_indices=False):
           
        a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
        if a1.ndim < 2:
            raise ValueError('The input array must be at least two-dimensional.')

        # Also check if dtype is LAPACK compatible
        if a1.dtype.char not in 'fdFD':
            dtype_char = lapack_cast_dict[a1.dtype.char]
            if not dtype_char:  # No casting possible
                raise TypeError(f'The dtype {a1.dtype} cannot be cast '
                                'to float(32, 64) or complex(64, 128).')

            a1 = a1.astype(dtype_char[0])  # makes a copy, free to scratch
            overwrite_a = True

        *nd, m, n = a1.shape
        k = min(m, n)
        real_dchar = 'f' if a1.dtype.char in 'fF' else 'd'

        # Empty input
        if min(*a1.shape) == 0:
            if permute_l:
                PL = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
                U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
                return PL, U
            else:
                P = (np.empty([*nd, 0], dtype=np.int32) if p_indices else
                     np.empty([*nd, 0, 0], dtype=real_dchar))
                L = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
                U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
                return P, L, U

        # Scalar case
        if a1.shape[-2:] == (1, 1):
            if permute_l:
                return np.ones_like(a1), (a1 if overwrite_a else a1.copy())
            else:
                P = (np.zeros(shape=[*nd, m], dtype=int) if p_indices
                     else np.ones_like(a1))
                return P, np.ones_like(a1), (a1 if overwrite_a else a1.copy())

        # Then check overwrite permission
        if not _datacopied(a1, a):  # "a"  still alive through "a1"
            if not overwrite_a:
                # Data belongs to "a" so make a copy
                a1 = a1.copy(order='C')
            #  else: Do nothing we'll use "a" if possible
        # else:  a1 has its own data thus free to scratch

        # Then layout checks, might happen that overwrite is allowed but original
        # array was read-only or non-contiguous.

        if not (a1.flags['C_CONTIGUOUS'] and a1.flags['WRITEABLE']):
            a1 = a1.copy(order='C')

        if not nd:  # 2D array

            p = np.empty(m, dtype=np.int32)
            u = np.zeros([k, k], dtype=a1.dtype)
            lu_dispatcher(a1, u, p, permute_l)
            P, L, U = (p, a1, u) if m > n else (p, u, a1)

        else:  # Stacked array

            # Prepare the contiguous data holders
            P = np.empty([*nd, m], dtype=np.int32)  # perm vecs

            if m > n:  # Tall arrays, U will be created
                U = np.zeros([*nd, k, k], dtype=a1.dtype)
                for ind in product(*[range(x) for x in a1.shape[:-2]]):
                    lu_dispatcher(a1[ind], U[ind], P[ind], permute_l)
                L = a1

            else:  # Fat arrays, L will be created
                L = np.zeros([*nd, k, k], dtype=a1.dtype)
                for ind in product(*[range(x) for x in a1.shape[:-2]]):
                    lu_dispatcher(a1[ind], L[ind], P[ind], permute_l)
                U = a1

        # Convert permutation vecs to permutation arrays
        # permute_l=False needed to enter here to avoid wasted efforts
        if (not p_indices) and (not permute_l):
            if nd:
                Pa = np.zeros([*nd, m, m], dtype=real_dchar)
                # An unreadable index hack - One-hot encoding for perm matrices
                nd_ix = np.ix_(*([np.arange(x) for x in nd] + [np.arange(m)]))
                Pa[(*nd_ix, P)] = 1
                P = Pa
            else:  # 2D case
                Pa = np.zeros([m, m], dtype=real_dchar)
                Pa[np.arange(m), P] = 1
                P = Pa

        return (L, U) if permute_l else (P, L, U)
    '''
    return P, L, U


def ll_transpose_decomposition(matrix):
    L = cholesky(matrix, lower=True)

    '''
    def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
        c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True,
                             check_finite=check_finite)
        return c
    '''
    Lt = np.transpose(L)
    return L, Lt


def save_table_as_image(variables, values, filename):
    fig, ax = plt.subplots()
    table_data = list(zip(variables, values))
    table = ax.table(cellText=table_data, colLabels=["Variable", "Value"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Hide axes
    ax.axis('off')

    # Save the image
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_matrix(matrix, title):
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.title(title)
    plt.colorbar()
    plt.show()


def main():
    file_path = "./nos7.mtx"  # Replace with the actual file path

    matrix = create_matrix_from_file(file_path)
    if matrix:
        print("Matrix:")
        for row in matrix:
            print(row)

    # LU decomposition
    P, L, U = lu_decomposition(matrix)

    print("LU Decomposition:")
    print("P:")
    print(P)
    print("\nL:")
    print(L)
    print("\nU:")
    print(U)

    # LL^t decomposition
    L, Lt = ll_transpose_decomposition(matrix)

    print("\nLL^t Decomposition:")
    print("L:")
    print(L)
    print("\nLt:")
    print(Lt)

    # Plot L, Lt matrices
    '''
    plot_matrix(P, "Matrix Permutation")
    plot_matrix(L, "Matrix Lower_triangular")
    plot_matrix(U, "Matrix Upper_triangular")
    plot_matrix(Lt, "Matrix Lower_triangular_transpose")
    '''

    size_x = len(matrix)
    size = len(matrix)
    np.random.seed(32)
    initialized_x = np.random.uniform(low=-10.0, high=10.0, size=(size_x))
    print("initialized_x_vector:", initialized_x)
    b = dot(matrix, initialized_x)
    print("RHS(b):", b)
    augmented_matrix = np.concatenate((matrix, b.reshape(-1, 1)), axis=1)
    print("augmented matrix:", augmented_matrix)

    '''
    # Determine the number of images needed based on the number of variables
    num_images = 1  # adjust based on the desired number of images
    variables_per_image = size // num_images
    # Save tables as images
    for j in range(num_images):
        start_idx = j * variables_per_image
        end_idx = min((j + 1) * variables_per_image, size)
        image_filename = f'x_initialize.png'
        save_table_as_image([f'X{k + 1}' for k in range(start_idx, end_idx)], initialized_x[start_idx:end_idx],
                                    image_filename)
    '''
    # Making numpy array of n size and initializing
    # to zero for storing solution vector
    x = np.zeros(size)

    # Applying Gauss Elimination
    for i in range(size):
        if augmented_matrix[i][i] == 0.0:
            sys.exit('Divide by zero')
        for j in range(i + 1, size):
            ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
            for k in range(size+1):
                augmented_matrix[j][k] = augmented_matrix[j][k] - ratio * augmented_matrix[i][k]
    # Back Substitution
    x[size - 1] = augmented_matrix[size - 1][size] / augmented_matrix[size - 1][size - 1]
    for i in range(size - 2, -1, -1):
        x[i] = augmented_matrix[i][size]
        for j in range(i + 1, size):
            x[i] = x[i] - augmented_matrix[i][j] * x[j]
        x[i] = x[i] / augmented_matrix[i][i]
    # Displaying solution
    print('\nRequired solution is: ')
    for i in range(size):
        print('X%d = %f' % (i, x[i]), end='\t')

    '''
    image_size = 100
    # Determine the number of images needed based on the number of variables
    num_images = 1  # adjust based on the desired number of images
    variables_per_image = size // num_images

    # Save tables as images
    for i in range(num_images):
        start_idx = i * variables_per_image
        end_idx = min((i + 1) * variables_per_image, size)
        image_filename = f'solution_table_Gaussian_elimination.png'
        # save_table_as_image([f'X{j + 1}' for j in range(start_idx, end_idx)], x[start_idx:end_idx], image_filename)
    '''

    def jacobi(A, b, N=10, x=None):
        # Create an initial guess if needed

        # Create a vector of the diagonal elements of A
        # and subtract them from A
        D = diag(A)
        # R is L+U
        R = A - diagflat(D)

        # Iterate for N times
        for i in range(N):
            x_previous = x
            x = (b - dot(R, x)) / D
            norm = np.linalg.norm(x - x_previous, ord=np.inf)
            print(norm)
            '''
            for j in range(num_images):
                start_idx = j * variables_per_image
                end_idx = min((j + 1) * variables_per_image, size)
                image_filename = f'solution_jacobi_iter{i+1}.png'
                save_table_as_image([f'X{k + 1}' for k in range(start_idx, end_idx)], x[start_idx:end_idx],
                                    image_filename)
            '''
        return x

    guess = np.random.rand(size)
    print("initialized_guess for jacobi is:", guess)
    sol = jacobi(np.array(matrix), b, N=10, x=guess)
    print()
    print("Jacobi solution after 10 iterations is:", sol)
    print()

    '''
    image_size = 100
    # Determine the number of images needed based on the number of variables
    num_images = 1  # adjust based on the desired number of images
    variables_per_image = size // num_images
    # Save tables as images
    for i in range(num_images):
        start_idx = i * variables_per_image
        end_idx = min((i + 1) * variables_per_image, size)
        image_filename = f'solution_Jacobi_final.png'
        save_table_as_image([f'X{j + 1}' for j in range(start_idx, end_idx)], sol[start_idx:end_idx], image_filename)
    '''

    def sor_solver(A, b, omega, initial_guess, convergence_criteria):
        residual_list = []
        x_k = initial_guess[:]
        residual = np.linalg.norm(np.matmul(A, x_k) - b)  # Initial residual
        iterations = 0
        while residual > convergence_criteria:
            for i in range(A.shape[0]):
                x_k_old = x_k
                sigma = 0
                x_k_old = x_k
                for j in range(A.shape[1]):
                    if j != i:
                        sigma += A[i][j] * x_k[j]
                x_k[i] = (1 - omega) * x_k[i] + (omega / A[i][i]) * (b[i] - sigma)
                norm = np.linalg.norm(x_k - x_k_old, ord=np.inf)
            residual = np.linalg.norm(np.matmul(A, x_k) - b)
            iterations += 1
            '''
            if iterations < 5:
                for j in range(num_images):
                    start_idx = j * variables_per_image
                    end_idx = min((j + 1) * variables_per_image, size)
                    image_filename = f'solution_Gauss-Siedel_iter{iterations}.png'
                    save_table_as_image([f'X{k + 1}' for k in range(start_idx, end_idx)], x_k[start_idx:end_idx],
                                                image_filename)
            '''
            print('Residual: {0:10.6g}'.format(residual))
            residual_list.append(residual)
        return x_k

    # An example case that mirrors the one in the Wikipedia article
    residual_convergence = 1e-5
    omega = 1.5  # Relaxation factor

    initial_guess = np.random.rand(size)
    print("initial guess for SOR is:", initial_guess)
    SOR_answer = sor_solver(array(matrix), b, omega, initial_guess, residual_convergence)
    print("SOR answer is:", SOR_answer)


if __name__ == "__main__":
    main()
