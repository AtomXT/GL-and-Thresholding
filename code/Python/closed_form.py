# -*- coding:utf-8 -*-
# author:TongXu
# datetime:2020/12/22 8:40
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.io import loadmat


def soft_thresholding(m, tau):
    """
    soft thresholding(m, tau)

        Do soft-thresholding to matrix `m` with threshold `tau`.

        Parameters
        ----------
        m : array_like
            Covariance matrix.
        tau : float number
              The regularization parameter for soft thresholding.

        Returns
        -------
        k_pos, k_neg : array_like
                       Positive and negative number in thresholed `m`.
        ii_pos, jj_pos, ii_neg, jj_neg : array_like
                                         Row and column indices of `k_pos` and `k_neg`.
    """
    con1 = m - tau
    con2 = m + tau
    ii_pos, jj_pos = np.where(con1 > 0)
    ii_neg, jj_neg = np.where(con2 < 0)
    k_pos = con1[ii_pos, jj_pos]
    k_neg = con2[ii_neg, jj_neg]
    return ii_pos, jj_pos, ii_neg, jj_neg, k_pos, k_neg


def sigma_res(ii, jj, kk, xd, nn, tau):
    """
    sigma_res(ii, jj, kk, xd, nn, tau)

        Calculate `res`.
        It has the same values as sigma at positions picked by thresholding, and the same diagonals as sigma.

        Parameters
        ----------
        ii, jj : array_like
                 Indices of values in sparse matrix.
        kk : array_like
             Values to generate the sparse matrix.
        xd : array_like
             Diagonals of the sample covariance matrix.
        nn : float number
             The dimension of the sparse matrix
        tau : float number
              The regularization parameter for soft thresholding.

        Returns
        -------
        res : sparse matrix
                   matrix `res`.

    """
    res = csc_matrix((kk + tau * np.sign(kk), (ii, jj)), shape=(nn, nn))
    res = res + res.T
    res = res + diags(xd)
    return res


def a_s(ii, jj, kk, xd, nn):
    """
    a_s(ii, jj, kk, xd, nn)

        Form the closed-form solution a and matrix s.

        Parameters
        ----------
        Refer to `sigma_res`.

        Returns
        -------
        a : sparse matrix
            The closed-form solution.
        s : sparse matrix
            Thresholded sigma matrix and the diagonals remain unchanged.

    """
    s = csc_matrix((kk, (ii, jj)), shape=(nn, nn))
    ii_a, jj_a = s.nonzero()
    kk_a = s[ii_a, jj_a].A1

    aa = -kk_a / (xd[ii_a] * xd[jj_a] - np.square(kk_a))
    a = csc_matrix((aa, (ii_a, jj_a)), shape=(nn, nn))
    a = a + a.T
    s = s + s.T
    a_temp = (sum(-a.multiply(s)) + csc_matrix(np.ones(nn))) / csc_matrix(xd)

    a = a + diags(a_temp.A1)
    s = s + diags(xd)
    return a, s


def closed_form(X, tau):
    """
    closed_form(X, tau)

        Calculate the closed-form solution of thresholding method relative to parameter `tau`.

        Parameters
        ----------
        X : sparse matrix
            Sample covariance matrix.
        tau : float number
              Regularization parameter.

        Returns
        -------
        S : sparse matrix
            Thresholded sigma matrix.
        A : sparse matrix
            Closed-form solution.
        sigmaRes : sparse matrix
    """
    nn, mm = X.shape

    if nn <= 0:
        # If problem size small, just do it explicitly
        M = np.triu(np.dot(X, X.T))
        ii_pos, jj_pos, ii_neg, jj_neg, k_pos, k_neg = soft_thresholding(M, tau)
        S = csc_matrix((np.concatenate((k_pos, k_neg)),
                        np.concatenate((np.concatenate((ii_pos, jj_pos)),
                                        np.concatenate((ii_neg, jj_neg))))), shape=(nn, nn))
        S = S + np.triu(S, 1).T
        return S

    MEMORY_LIMIT = 500  # If we allocate more than this number times n nonzeros, terminate prematurely.

    # Get partition size
    skip = int(np.ceil(np.sqrt(nn * MEMORY_LIMIT)))
    print("phase 1: done")

    # Partition x into sub matrix
    rows = np.append(range(0, nn, skip), nn)  # beginning row of every sub_matrix
    p = len(rows) - 1

    # Threshold one sub_matrix of M=x*x' at a time
    ii = []
    jj = []
    kk = []
    total_nz = 0

    Xd = []

    X = np.array_split(X, rows[1:-1])

    for j in range(p):
        this_ii = []
        this_jj = []
        this_kk = []
        for i in range(j + 1):
            Mij = np.matmul(X[i], X[j].T) / mm
            if i == j:
                Xd.append(np.diag(Mij))
                Mij = np.triu(Mij, 1)

            # Do soft-thresholding
            ii_pos, jj_pos, ii_neg, jj_neg, k_pos, k_neg = soft_thresholding(Mij, tau)

            # Record nonzeros
            this_ii.append(np.concatenate((ii_pos, ii_neg)) + rows[i])
            this_jj.append(np.concatenate((jj_pos, jj_neg)) + rows[j])
            this_kk.append(np.concatenate((k_pos, k_neg)))

            # Sum nonzeros
            total_nz += len(this_ii[i])

            # Check for memory issues
            if total_nz > MEMORY_LIMIT * 10 * nn:
                print("REACHED MEMORY LIMIT. EXITING....")
                return

        # Assemble this column
        ii.append(np.concatenate(this_ii))
        jj.append(np.concatenate(this_jj))
        kk.append(np.concatenate(this_kk))

    # Assemble all columns
    ii = np.concatenate(ii)
    jj = np.concatenate(jj)
    kk = np.concatenate(kk)
    Xd = np.concatenate(Xd)

    # form sparse matrix
    A, S = a_s(ii, jj, kk, Xd, nn)
    sigmaRes = sigma_res(ii, jj, kk, Xd, nn, tau)
    return S, A, sigmaRes


def closed_form_error(A, invsigma):
    """
    closed_form_error(A, invsigma)

        Calculate approximation errors, including TPR, FPR, and Normalized error.

        Parameters
        ----------
        A : sparse matrix
            The closed-form solution.
        invsigma : Inverse of true sigma matrix.
    """
    Sparse_closed = csc_matrix(np.abs(A) > 1e-5)
    Sparse_true = csc_matrix(np.abs(invsigma) > 1e-5)

    TPR_closed = (Sparse_true.multiply(Sparse_closed)).sum() / Sparse_true.sum()
    FPR_closed = (Sparse_closed.sum() - (Sparse_true.multiply(Sparse_closed)).sum()) / (
            Sparse_true.shape[0] * Sparse_true.shape[1] - Sparse_true.sum())
    print("TPR and FPR for closed form: \n TPR_closed={}\n FPR_closed={}".format(TPR_closed, FPR_closed))

    E = A - invsigma
    nnz_E = E[np.nonzero(E)]
    nnz_invsigma = invsigma[np.nonzero(invsigma)]
    norm_error = np.linalg.norm(nnz_E) / np.linalg.norm(nnz_invsigma)
    print("Normalized error norm for closed form: {}".format(norm_error))


if __name__ == '__main__':
    # Example
    dim = 12000
    n = 6000
    K = 0.435
    lam = K * np.sqrt(np.log(dim) / n)

    data = loadmat("test.mat")
    x = data["x"]  # Sample data
    invSigma = data["invSigma"]  # Inverse of true Sigma matrix.
    S, A, Sigma_res = closed_form(x, lam)

    # Errors
    closed_form_error(A, invSigma)
