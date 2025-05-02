# hankel_utils.py
import warnings

import numpy as np
from scipy.signal import max_len_seq


def blkhankel(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Construct a block Hankel matrix from column and row data

    Args:
        C: Column data (2D array)
        R: Row data (2D array)

    Returns:
        2D array: Block Hankel matrix
    """
    nrC, nc = C.shape
    nr, ncR = R.shape

    # 1) Dimension consistency
    if nrC % nr != 0:
        raise ValueError("Number of rows in C must be an integer multiple of rows in R")
    if ncR % nc != 0:
        raise ValueError("Number of columns in R must be an integer multiple of columns in C")

    p = nrC // nr   # block‐rows
    q = ncR // nc   # block‐cols

    # 2) Conflict check
    C_bottom = C[(p-1)*nr: , :]
    R_top    = R[: , :nc]
    tol = np.sqrt(np.finfo(float).eps)
    if not np.allclose(C_bottom, R_top, atol=tol):
        warnings.warn("Data conflict: bottom‐left block overwritten by C", UserWarning)

    # 3) Handle trivial cases
    if p == 1:
        H = R.copy()
        H[:, :nc] = C
        return H
    if q == 1:
        return C.copy()

    # 4) Build full Hankel
    H = np.zeros((nr * p, nc * q), dtype=C.dtype)
    H[:, :nc] = C

    for j in range(1, q):
        H[0:(p-1)*nr, j*nc:(j+1)*nc]   = H[nr:p*nr, (j-1)*nc:j*nc]
        H[(p-1)*nr:p*nr, j*nc:(j+1)*nc] = R[:, j*nc:(j+1)*nc]

    return H

def get_markov_matrix(
    C: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    pp: np.ndarray
) -> np.ndarray:
    """
    Construct a block‐matrix of Markov parameters determined by exponents in pp.

    The (i,j) block is C @ A**pp[i,j] @ B if pp[i,j] >= 0, otherwise zeros.

    Args:
        C: (m × nx) output matrix
        A: (nx × nx) state transition matrix
        B: (nx × n) input matrix
        pp: (rows × cols) array of nonnegative integer exponents

    Returns:
        M: (m·rows × n·cols) block matrix of Markov parameters
    """
    m = C.shape[0]  # Number of outputs
    n = B.shape[1]  # Number of inputs
    rows, cols = pp.shape

    # Initialize output matrix
    M = np.zeros((rows * m, cols * n))

    for i in range(rows):
        for j in range(cols):
            exp_ij = pp[i, j]
            if exp_ij >= 0:
                A_pow = np.linalg.matrix_power(A, int(exp_ij))
                M[i*m:(i+1)*m, j*n:(j+1)*n] = C @ A_pow @ B

    return M


def make_smm(z, Tp, Tf, M):
    """
    Create past- and future-horizon Hankel matrices (Hzp, Hzf) from a signal z.

    The signal z may be shape (K, nz) or (nz, K).  We form T = Tp + Tf,
    then build a block‐Hankel matrix with M columns (default M = K − T + 1).
    If M is too large for the signal length, raises an error; if smaller,
    truncates the extra samples with a warning.

    Args:
        z: 2D array of shape (K, nz) or (nz, K)
        Tp: past horizon length
        Tf: future horizon length
        M: desired number of columns; if None, defaults to K − (Tp+Tf) + 1

    Returns:
        Hzp: (nz*Tp × M) past‐horizon Hankel
        Hzf: (nz*Tf × M) future‐horizon Hankel
    """

    # Ensure 2D
    if z.ndim != 2:
        raise ValueError("z must be a 2D array")

    # Shape to (nz, K)
    nrz, ncz = z.shape
    if nrz > ncz:
        z = z.T
    nz, K = z.shape

    T = Tp + Tf

    # Calculate M if not provided
    if M is None:
        M = K - T + 1

    # Required data length
    Krqd = M + T - 1
    if Krqd > K:
        raise ValueError(f"Signal length ({K}) too short for M={M} and T={T}")
    if Krqd < K:
        warnings.warn(f"Last {K-Krqd} points of signal discarded", UserWarning)
        z = z[:, :Krqd]

    zvec = z.flatten(order='F')
    # Build the full Hankel using our blkhankel implementation
    C = zvec[: T * nz].reshape(T * nz, 1)
    R = z[:, T-1 : ]  # from column T to end (0-based index T-1)
    H = blkhankel(C, R)

    # Split into past and future blocks
    Hzp = H[: Tp * nz, :]
    Hzf = H[Tp * nz : T * nz, :]

    return Hzp, Hzf

def generate_prbs(n, num_channels=1, min_val=-1, max_val=1):
    """Generate multi-channel PRBS signal"""
    prbs = []
    for _ in range(num_channels):
        # Generate maximum-length PRBS sequence
        seq, _ = max_len_seq(n.bit_length())
        signal = seq[:n]
        # Scale and offset to desired range
        signal = min_val + (max_val - min_val) * (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        prbs.append(signal)
    return np.column_stack(prbs)
