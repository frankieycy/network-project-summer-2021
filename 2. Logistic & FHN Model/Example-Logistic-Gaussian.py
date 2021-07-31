'''
Example run of Logistic Gaussian network
@ Frankie Yeung (2020 Dec)
'''
import numpy as np
import matplotlib.pyplot as plt
import network as n
np.random.seed(0)

size            = 100
p               = 0.2
w               = [10, 2]
dt, T           = 5e-4, 2e3
sigma           = 1
x0, x1          = 0, 5
r0              = 10

args = {
    "network-size":         size,
    "connect-prob":         p,
    "gij-mean-spread":      w,
    "step-size":            dt,
    "terminal-time":        T,
    "noise-covariance":     sigma**2 * np.eye(size),
    "init-condition":       np.random.uniform(x0, x1, size),
    "intrinsic-coef":       [r0] * size,
}

myNet = n.network()
myNet.randomDirectedWeightedGraph(
    args["network-size"],
    args["connect-prob"],
    args["gij-mean-spread"][0],
    args["gij-mean-spread"][1]
)

myNet.initDynamics(
    args["init-condition"],
    args["intrinsic-coef"],
    args["noise-covariance"]
)

myNet.runDynamics(
    args["step-size"],
    args["terminal-time"]
)
