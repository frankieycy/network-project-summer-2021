'''
Example run of FHN DIV25 network
@ Frankie Yeung (2020 Dec)
'''
import numpy as np
import matplotlib.pyplot as plt
import network as n
np.random.seed(0)

myNet = n.network()
myNet.loadGraph("DIV25_PREmethod.txt")

size            = myNet.size
dt, T           = 5e-4, 2e3
sigma           = 2
x0, x1          = -1, 1
y0, y1          = -1, 1
epsilon, alpha  = 0.1, 0.95

args = {
    "network-size":         size,
    "step-size":            dt,
    "terminal-time":        T,
    "noise-covariance":     sigma**2 * np.eye(size),
    "init-condition-x":     np.random.uniform(x0, x1, size),
    "init-condition-y":     np.random.uniform(y0, y1, size),
    "epsilon":              epsilon,
    "alpha":                alpha
}

myNet.initDynamics_FHN(
    args["init-condition-x"],
    args["init-condition-y"],
    args["epsilon"],
    args["alpha"],
    args["noise-covariance"]
)

myNet.runDynamics(
    args["step-size"],
    args["terminal-time"]
)
