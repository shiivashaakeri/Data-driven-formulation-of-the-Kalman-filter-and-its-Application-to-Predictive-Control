# smm_pred.py
import numpy as np
from scipy.linalg import svd


def make_SMM_pred(Hup: np.ndarray,
    Huf: np.ndarray,
    Hyp: np.ndarray,
    Hyf: np.ndarray,
    nu: int,
    ny: int,
    nx: int,
    ySigma: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear predictor from [up; yp; uf] to yf:
        yf_est = Eup*up + Eyp*ypmeas + Euf*uf

    Args:
        Hup:  (nu*Tp × M) past-input Hankel
        Huf:  (nu*Tf × M) future-input Hankel
        Hyp:  (ny*Tp × M) past-output Hankel
        Hyf:  (ny*Tf × M) future-output Hankel
        nu:   number of inputs
        ny:   number of outputs
        nx:   reduced state dimension for make_xz
        ySigma: (ny × ny) output noise covariance; if None, assumed uncorrelated

    Returns:
        Eup: (ny*Tf × nu*Tp) mapping from past inputs to future outputs
        Eyp: (ny*Tf × ny*Tp) mapping from past outputs to future outputs
        Eyuf: (ny*Tf × nu*Tf) mapping from future inputs to future outputs
    """
    # 1) Determine correlation flag
    corr = ySigma is not None
    if corr and ySigma.shape != (ny, ny):
        raise ValueError("ySigma must be square of shape (ny, ny)")

    # 2) Dimensions
    nuTp, _ = Hup.shape
    nyTf, _ = Hyf.shape
    Tp = nuTp // nu
    Tf = nyTf // ny

    # 3) Reduced SMM description
    Lup, Lyup, Lyp, Luf, Lyuf, Suu, Suy, Syu, Syy, Qnp, Qup, Qyp, Quf = make_XZ(
    Hup, Huf, Hyp, Hyf, nu, ny, nx)

    # 4) Scale ySigma for numerical conditioning
    if corr:
        svals = np.linalg.svd(ySigma, compute_uv=False)
        mx, mn = svals.max(), svals.min()
        if mx < ny**2 * 1e-12:
            corr = False
        else:
            if mn < (nyTf)**2 * 1e-12 or (mx/mn) > 1e6:
                raise ValueError("Condition number of ySigma is too large")
            scale = 1.0 / np.sqrt(mx * mn)
            ySigma = scale * ySigma

    # 5) Compute Exy
    if corr:
        iy = np.linalg.solve(ySigma, np.eye(ny))
        iYSigma = np.kron(np.eye(Tp), iy)
        Exy = np.linalg.solve((Lyp.T @ iYSigma @ Lyp), (Lyp.T @ iYSigma))
    else:
        Exy = np.linalg.solve((Lyp.T @ Lyp), Lyp.T)

    # 6) Inverses of Lup and Luf
    iLup = np.linalg.solve(Lup, np.eye(nu * Tp))        # Lup should remain square
    iLuf = np.linalg.pinv(Luf)

    # 7) Build predictors
    Eyup = Lyup @ iLup
    Eyuf = Lyuf @ iLuf
    Eup = ((Syu - Eyuf @ Suu) @ iLup
           + Eyuf @ Suy @ Exy @ Eyup
           - Syy @ Exy @ Eyup)
    Eyp = (Syy - Eyuf @ Suy) @ Exy

    return Eup, Eyp, Eyuf



def make_XZ(Hup, Huf, Hyp, Hyf, nu, ny, nx):
    nuTp, M = Hup.shape
    nyTf, _ = Hyf.shape
    Tp = nuTp // nu
    Tf = nyTf // ny

    Hyup = np.vstack((Hup, Hyp))
    Hyuf = np.vstack((Huf, Hyf))

    # Full QR decomposition
    Qp, RpT = np.linalg.qr(Hyup.T, mode='complete')
    Lp = RpT.T

    Lup = Lp[0:Tp*nu, 0:Tp*nu]
    Qup = Qp[:, 0:Tp*nu]

    Lyup = Lp[Tp*nu:Tp*(nu+ny), 0:Tp*nu]
    Lyp  = Lp[Tp*nu:Tp*(nu+ny), Tp*nu:Tp*nu + nx]
    Qyp  = Qp[:, Tp*nu:Tp*nu + nx]

    Qnp = Qp[:, Tp*nu + nx :]

    Hyuf_proj = Hyuf @ Qnp
    Qf, RfT = np.linalg.qr(Hyuf_proj.T, mode='complete')
    Lf = RfT.T

    Luf  = Lf[0:nu*Tf, 0:nu*Tf]
    Quf  = Qf[:, 0:nu*Tf]
    Lyuf = Lf[nu*Tf:(nu+ny)*Tf, 0:nu*Tf]

    Suu = Huf @ Qup
    Suy = Huf @ Qyp
    Syu = Hyf @ Qup
    Syy = Hyf @ Qyp

    return Lup, Lyup, Lyp, Luf, Lyuf, Suu, Suy, Syu, Syy, Qnp, Qup, Qyp, Quf

def make_Lp_S_Lf(Huwyp: np.ndarray,
    Huwyf: np.ndarray,
    nu: int,
    nw: int,
    ny: int,
    nxy: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create reduced SMM description for three input signals u, w, & y.

    Args:
        Huwyp: Past‐horizon Hankel for [u; w; y], shape ((nu+nw+ny)*Tp, M)
        Huwyf: Future‐horizon Hankel for [u; w; y], shape ((nu+nw+ny)*Tf, M)
        nu:     Number of u channels
        nw:     Number of w channels
        ny:     Number of y channels
        nxy:    Reduced dimension for the [u; w] + y null‐space

    Returns:
        Lp:    Lower‐triangular past‐horizon factor, shape ((nu+nw+ny)*Tp, (nu+nw)*Tp + nxy)
        S:     Future‐horizon mapping for past states, shape ((nu+nw+ny)*Tf, (nu+nw)*Tp + nxy)
        Lf:    Lower‐triangular future‐horizon factor for zetau/zetaw, shape ((nu+nw+ny)*Tf, (nu+nw)*Tf)
        Lyyf:  Truncated bottom‐right block of Lf (future y→y), shape (ny*Tf, ny*Tf)
    """
    # Dimensions
    nTp, M = Huwyp.shape
    nTf, _ = Huwyf.shape
    block_size = nu + nw + ny
    Tp = nTp // block_size
    Tf = nTf // block_size

    # LQ (QR of transpose) on past data
    Qp, RpT = np.linalg.qr(Huwyp.T)
    Lp = RpT.T

    # Truncate Lp to remove null‐space beyond nxy
    cols_keep = (nu + nw) * Tp + nxy
    Lp = Lp[:, :cols_keep]

    # Partition Qp
    Quwyp = Qp[:, :cols_keep]
    Qnp   = Qp[:, cols_keep:M]

    # Future‐horizon mapping for past states
    S = Huwyf @ Quwyp

    # LQ on future data projected into null‐space of past
    _, RfT = np.linalg.qr((Huwyf @ Qnp).T)
    Lf_full = RfT.T

    # Extract Lyyf (future‐y to future‐y block)
    start = (nu + nw) * Tf
    end   = (nu + nw + ny) * Tf
    Lyyf = Lf_full[start:end, start:end]

    # Truncate Lf to columns corresponding to zetau & zetaw
    Lf = Lf_full[:, : (nu + nw) * Tf]

    return Lp, S, Lf, Lyyf