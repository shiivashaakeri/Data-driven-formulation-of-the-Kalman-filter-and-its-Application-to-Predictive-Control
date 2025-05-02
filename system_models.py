# flight_model.py
import warnings

import control as ct
import numpy as np
from control import ss


import numpy as np
import control as ct

def flight_model(gust_sigma=10):
    """
    Generate continuous-time model of Boeing 747 longitudinal dynamics,
    including Dryden gust models with MATLAB-matched coefficients.

    Args:
        gust_sigma (float or list): Turbulence intensities [horizontal, vertical]

    Returns:
        ct.StateSpace: Combined system with 4 inputs, 4 outputs
    """
    # Handle gust_sigma input
    if isinstance(gust_sigma, (int, float)):
        gust_sigmau = gust_sigma
        gust_sigmav = gust_sigma
    elif len(gust_sigma) == 2:
        gust_sigmau, gust_sigmav = gust_sigma
    else:
        raise ValueError("gust_sigma must be scalar or 2-element list")

    # __ Aircraft dynamics __
    Ac = np.array([
        [-0.003,  0.039,   0.0,   -0.322],
        [-0.065, -0.319,   7.74,   0.0],
        [ 0.02,  -0.101,  -0.429,  0.0],
        [ 0.0,    0.0,     1.0,    0.0]
    ])

    # B = [B_u | B_w]
    Bc = np.array([
        [ 0.010,  1.000, -1.000,  0.000],
        [-0.180, -0.040,  0.000, -1.000],
        [-1.160,  0.598,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  0.000]
    ])

    Cc = np.array([
        [1,  0, 0,  0],       # velocity
        [0, -1, 0,  7.74],    # climb rate
    ])
    Dc = np.zeros((2, 4))

    Gdyn = ct.ss(Ac, Bc, Cc, Dc,
                 inputs=["elev", "throt", "wu", "wv"],
                 outputs=["vel", "climb"],
                 name="Aircraft")

    # __ MATLAB-matched Horizontal Gust model __
    A_gu = [[-0.4423]]
    B_gu = [[2.0]]
    C_gu = [[2.653]]
    D_gu = [[0.0]]
    Ggu = ct.ss(A_gu, B_gu, C_gu, D_gu,
                inputs=["wh_in"], outputs=["wu_out"],
                name="HorzGust")

    # __ MATLAB-matched Vertical Gust model __
    A_gw = [[-0.8846, -0.3912],
            [0.5,      0.0]]
    B_gw = [[4.0],
            [0.0]]
    C_gw = [[1.625, 0.8298]]
    D_gw = [[0.0]]
    Ggw = ct.ss(A_gw, B_gw, C_gw, D_gw,
                inputs=["wv_in"], outputs=["wv_out"],
                name="VertGust")

    # __ Connect total system __
    Gs = ct.interconnect(
        [Gdyn, Ggu, Ggw],
        connections=[
            ("Aircraft.wu", "HorzGust.wu_out"),
            ("Aircraft.wv", "VertGust.wv_out"),
        ],
        inplist=[
            "Aircraft.elev", "Aircraft.throt",
            "HorzGust.wh_in", "VertGust.wv_in"
        ],
        outlist=[
            "Aircraft.vel", "Aircraft.climb",
            "HorzGust.wu_out", "VertGust.wv_out"
        ]
    )

    return Gs


def make_ss_smm_pred(Lp: np.ndarray, nu: int, nw: int, ny: int, Tp: int, Ts: float) -> ss:
    """
    Create state-space predictive model from SMM decomposition.

    Args:
        Lp:  Lower‐triangular Lp from LQ (shape [(nu+nw+ny)*Tp, (nu+nw)*Tp + nxy])
        nu:  # control inputs
        nw:  # disturbance inputs
        ny:  # outputs
        Tp:  past horizon
        Ts:  sampling time

    Returns:
        ss: discrete‐time state‐space predictor (xu; xw; xy) → (u(k−1); w(k−1); y(k−1))
    """
    # 1) Validate and infer nxy
    nrL, ncL = Lp.shape
    expected_rows = (nu + nw + ny) * Tp
    if nrL != expected_rows:
        raise ValueError(f"Lp has {nrL} rows, expected {expected_rows}")
    nxy = ncL - (nu + nw) * Tp
    if not (1 <= nxy <= ny * Tp):
        raise ValueError(f"Invalid nxy={nxy}. Must satisfy 1 <= nxy <= {ny * Tp}")

    # 2) Partition Lp
    uidx = slice(0, nu * Tp)
    widx = slice(nu * Tp, (nu + nw) * Tp)
    yidx = slice((nu + nw) * Tp, (nu + nw + ny) * Tp)
    xyidx = slice((nu + nw) * Tp, (nu + nw) * Tp + nxy)

    Luup = Lp[uidx, uidx]
    Lwup = Lp[widx, uidx]
    Lwwp = Lp[widx, widx]
    Lyup = Lp[yidx, uidx]
    Lywp = Lp[yidx, widx]
    Lyyp = Lp[yidx, xyidx]

    # 3) Build shift matrices
    # Zp = diag(ones(Tp-1), 1)
    Zp = np.zeros((Tp, Tp))
    for i in range(Tp - 1):
        Zp[i, i + 1] = 1
    Zup = np.kron(Zp, np.eye(nu))
    Zwp = np.kron(Zp, np.eye(nw))
    Zyp = np.kron(Zp, np.eye(ny))

    # Build the indicator vector as a column
    PIk = np.zeros((Tp, 1))
    PIk[-1, 0] = 1

    # Build PIuk, PIwk, PIyk correctly
    PIuk = np.kron(PIk, np.eye(nu))   # shape → (nu*Tp, nu)
    PIwk = np.kron(PIk, np.eye(nw))   # shape → (nw*Tp, nw)
    PIyk = np.kron(PIk, np.eye(ny))   # shape → (ny*Tp, ny)
    # PIyp: rank condition for y-state
    PIyp = np.kron(np.hstack([np.eye(Tp - 1), np.zeros((Tp - 1, 1))]), np.eye(ny))

    # 5) Compute A and B blocks via solves (Luup \ X -> solve(Luup, X))
    Auu = np.linalg.solve(Luup, Zup.dot(Luup))
    Auw = np.zeros((nu * Tp, nw * Tp))
    Auy = np.zeros((nu * Tp, nxy))
    Buu = np.linalg.solve(Luup, PIuk)
    Buw = np.zeros((nu * Tp, nw))

    Awu = np.linalg.solve(Lwwp, Zwp.dot(Lwup) - Lwup.dot(Auu))
    Aww = np.linalg.solve(Lwwp, Zwp.dot(Lwwp))
    Awy = np.zeros((nw * Tp, nxy))
    Bwu = -np.linalg.solve(Lwwp, Lwup.dot(Buu))
    Bww = np.linalg.solve(Lwwp, PIwk)

    if nxy > (Tp - 1) * ny:
        warnings.warn("Direct feedthrough y(k+1)→xy(k+1) ignored", UserWarning)

    Ayu = np.linalg.solve(PIyp.dot(Lyyp), PIyp.dot(Zyp.dot(Lyup) - Lyup.dot(Auu) - Lywp.dot(Awu)))
    Ayw = np.linalg.solve(PIyp.dot(Lyyp), PIyp.dot(Zyp.dot(Lywp) - Lywp.dot(Aww)))
    Ayy = np.linalg.solve(PIyp.dot(Lyyp), PIyp.dot(Zyp.dot(Lyyp)))
    Byu = -np.linalg.solve(PIyp.dot(Lyyp), PIyp.dot(Lyup.dot(Buu) + Lywp.dot(Bwu)))
    Byw = -np.linalg.solve(PIyp.dot(Lyyp), PIyp.dot(Lywp.dot(Bww)))

    # 6) Assemble full A, B
    Ap = np.block(
        [
            [Auu, Auw, Auy],
            [Awu, Aww, Awy],
            [Ayu, Ayw, Ayy],
        ]
    )
    Bup = np.vstack([Buu, Bwu, Byu])
    Bwp = np.vstack([Buw, Bww, Byw])

    # 7) Build C matrices for u(k−1), w(k−1), y(k−1)
    Cu = np.hstack([PIuk.T.dot(Luup), np.zeros((nu, nw * Tp)), np.zeros((nu, nxy))])
    Cw = np.hstack([PIwk.T.dot(Lwup), PIwk.T.dot(Lwwp), np.zeros((nw, nxy))])
    Cy = PIyk.T.dot(np.hstack([Lyup, Lywp, Lyyp]))

    # 8) Create state-space object
    # Inputs: [u; w], Outputs: [u(k-1); w(k-1); y(k-1)]
    Buf = np.hstack([Bup, Bwp])  # shape ((nu+nw+ny)*Tp, nu+nw)
    Cuf = np.vstack([Cu, Cw, Cy])
    Duf = np.zeros((nu + nw + ny, nu + nw))

    return ss(Ap, Buf, Cuf, Duf, Ts)
