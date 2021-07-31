'''
Example run of Logistic DIV25 network
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
sigma           = 0.5
x0, x1          = 0.9, 1.1
r0              = 10

args = {
    "network-size":         size,
    "step-size":            dt,
    "terminal-time":        T,
    "noise-covariance":     sigma**2 * np.eye(size),
    "init-condition":       np.random.uniform(x0, x1, size),
    "intrinsic-coef":       [r0] * size,
}

myNet.initDynamics(
    args["init-condition"],
    args["intrinsic-coef"],
    args["noise-covariance"]
)

myNet.runDynamics(
    args["step-size"],
    args["terminal-time"]
)
