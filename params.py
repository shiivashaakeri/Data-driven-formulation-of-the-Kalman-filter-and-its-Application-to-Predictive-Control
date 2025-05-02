# params.py

import numpy as np

# Plotting parameters
PLOT_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
}
FIG_SIZE = (11.20, 4.20)
COLORS = {
    "primary": "b",
    "secondary": "r",
}

# System parameters
GUST_SIGMA = np.array([10.0, 10.0])  # [horizontal, vertical] turbulence intensity

# Signal/model dimensions
NY = 2  # number of outputs (velocity, climb rate)
NU = 2  # number of control inputs (elevator, throttle)
NW = 2  # number of disturbance inputs (gust channels)

# Model data parameters
K_DATA = 2500  # length of data used to build Hankel matrices
K_INIT = 30  # initialization length (past horizon, Tp)
TP = K_INIT  # past horizon
TF = 20  # future horizon
T = TP + TF  # total horizon length
M = K_DATA - T + 1  # number of columns of Hankel matrices

# Gust simulation parameters
K_GUST = 5000  # length of gust simulation
DT = 0.1  # simulation timestep [s]
TS = DT  # sampling time for controller [s]
TSIM_GUST = DT * np.arange(K_GUST)

# Model simulation parameters
TSIM_MODEL = TS * np.arange(K_DATA + TF)

# Closed-loop parameters
T_CLP = 30.0  # closed-loop duration [s]
T_CLP_GRID = np.arange(0, T_CLP + TS, TS)
SIM_LEN_CLP = len(T_CLP_GRID)  # number of time steps
Y_REF_CONST = np.array([10.0, 0.0])
T_REF_CLP = TS * np.arange(SIM_LEN_CLP + TF)
Y_REF = np.tile(Y_REF_CONST, (len(T_REF_CLP), 1))

# Noise covariance
STD_DEV_V = 0.25 * np.eye(NY)
SIGMA_V = STD_DEV_V @ STD_DEV_V  # measurement noise covariance
SIGMA_W = np.eye(NW)  # disturbance covariance

# Performance weights
Q_PERF = 10.0 * np.eye(NY)
R_PERF = 0.01 * np.eye(NU)

# Input/output bounds
ULB = np.array([-20.0, -20.0])
UUB = np.array([20.0, 20.0])
YLB = np.array([-25.0, -15.0])
YUB = np.array([25.0, 15.0])

# Export control
EXPORT_FIGS = False

# Monte Carlo parameters
N_MC = 30

# PRBS generation settings
PRBS_SCALE = 3.0
PRBS_LENGTH = K_DATA + TF
PRBS_NU_NW = NU + NW
