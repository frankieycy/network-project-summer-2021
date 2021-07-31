import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
np.random.seed(0)
plt.switch_backend('Agg')

args = {
    "orig-network":                         "DIV66.npy",            # coupling strength matrix in .npy format
    "save-directory":                       "DIV66 ref network/",   # directory to save outputs and plots
    "plot-degree-and-strength-dist":        False,                  # whether to plot degree & strength distributions
    "comb-plot-degree-and-strength-dist":   False,                  # whether to plot degree & strength distributions as a combined graph
    "save-ref-networks-as-txt":             False,                  # whether to save coupling strengths of ref networks
    "save-ref-networks-as-npy":             True,                   # whether to save coupling strengths of ref networks
    "degree-dist-plot-xlim":                [0, 200],               # x-limits of degree distribution plots
    "strength-dist-plot-xlim":              [-0.02, 0.02]           # x-limits of strength distribution plots
}

refNetworks = [
    "Gaussian random",
    "shuffle rows",
    "shuffle cols",
    "shuffle entries",
    "replace entries with Gaussian"
]

signTypes = [
    "type0",    # no sign condition
    "typeA",    # same orig sign
    "typeB"     # majority sign
]

outputs = dict()

def makeDirectory(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def main():
    makeDirectory(args["save-directory"])
    origCouplingMtrx = np.load(args["orig-network"])
    n = origCouplingMtrx.shape[0]
    outputs["orig"] = origCouplingMtrx

    for network in refNetworks:
        refCouplingMtrx = np.zeros((n, n))

        if network == "Gaussian random":
            connectProb = (origCouplingMtrx != 0).sum() / (n * (n - 1))
            nonzeroCoupling = origCouplingMtrx[origCouplingMtrx != 0]
            couplingMean = nonzeroCoupling.mean()
            couplingStd = nonzeroCoupling.std()
            refAdjacency = (np.random.uniform(size=(n, n)) < connectProb).astype(int)
            refCouplingMtrx = np.random.normal(couplingMean, couplingStd, size=(n, n)) * refAdjacency

        elif network == "shuffle rows":
            refCouplingMtrx = origCouplingMtrx.copy()
            list(map(np.random.shuffle, refCouplingMtrx))

        elif network == "shuffle cols":
            tmpCouplingMtrx = origCouplingMtrx.T.copy()
            list(map(np.random.shuffle, tmpCouplingMtrx))
            refCouplingMtrx = tmpCouplingMtrx.T.copy()

        elif network == "shuffle entries":
            nonzeroCoupling = origCouplingMtrx[origCouplingMtrx != 0]
            np.random.shuffle(nonzeroCoupling)
            nonzeroIdx = np.where(origCouplingMtrx != 0)
            refCouplingMtrx[nonzeroIdx] = nonzeroCoupling

        elif network == "replace entries with Gaussian":
            nonzeroCoupling = origCouplingMtrx[origCouplingMtrx != 0]
            couplingMean = nonzeroCoupling.mean()
            couplingStd = nonzeroCoupling.std()
            nonzeroIdx = np.where(origCouplingMtrx != 0)
            refCouplingMtrx[nonzeroIdx] = np.random.normal(couplingMean, couplingStd, len(nonzeroIdx[0]))

        for type in signTypes:
            name = "%sref_%s_%s" % (args["orig-network"].split(".")[-2], type, network)
            signs = np.repeat(-1, n)

            if type == "type0":
                outputs[name] = refCouplingMtrx
                continue

            elif type == "typeA":
                signs[origCouplingMtrx.sum(axis=0) > 0] = 1

            elif type == "typeB":
                signs[(refCouplingMtrx > 0).sum(axis=0) - (refCouplingMtrx < 0).sum(axis=0)] = 1

            outputs[name] = signs * np.abs(refCouplingMtrx)

    if args["save-ref-networks-as-txt"]:
        for network in outputs:
            tmpCouplingMtrx = outputs[network].T.copy()
            fileName = args["save-directory"] + network + ".txt"
            nonzeroIdx = np.argwhere(tmpCouplingMtrx != 0).T
            np.savetxt(fileName, np.column_stack((nonzeroIdx[0], nonzeroIdx[1], tmpCouplingMtrx[nonzeroIdx[0], nonzeroIdx[1]])), fmt='%d %d %.10f')

    if args["save-ref-networks-as-npy"]:
        for network in outputs:
            tmpCouplingMtrx = outputs[network].T.copy()
            fileName = args["save-directory"] + network + ".npy"
            np.save(fileName, tmpCouplingMtrx)

    if args["plot-degree-and-strength-dist"]:
        for network in outputs:
            refCouplingMtrx = outputs[network]
            refAdjacency = refCouplingMtrx != 0

            degrees_in = refAdjacency.sum(axis=1)
            degrees_out = refAdjacency.sum(axis=0)

            with np.errstate(invalid='ignore'):
                strengths_in = np.nan_to_num(refCouplingMtrx.sum(axis=1) / degrees_in)
                strengths_out = np.nan_to_num(refCouplingMtrx.sum(axis=0) / degrees_out)

            #### degree distribution ###########################################
            fig = plt.figure()
            fileName = args["save-directory"] + "degree_" + network + ".png"
            x_deg = np.linspace(
                args["degree-dist-plot-xlim"][0],
                args["degree-dist-plot-xlim"][1],
                200)
            density_in = gaussian_kde(degrees_in)
            density_out = gaussian_kde(degrees_out)
            plt.plot(x_deg, density_in(x_deg), c="r", label="in-degree")
            plt.plot(x_deg, density_out(x_deg), c="b", label="out-degree")
            plt.xlim(args["degree-dist-plot-xlim"])
            plt.ylim(bottom=0)
            plt.xlabel('degree')
            plt.title(network)
            plt.legend()
            fig.tight_layout()
            fig.savefig(fileName)
            plt.close()

            #### strength distribution #########################################
            fig = plt.figure()
            fileName = args["save-directory"] + "strength_" + network + ".png"
            x_str = np.linspace(
                args["strength-dist-plot-xlim"][0],
                args["strength-dist-plot-xlim"][1],
                200)
            density_in = gaussian_kde(strengths_in)
            density_out = gaussian_kde(strengths_out)
            plt.plot(x_str, density_in(x_str), c="r", label="in-strength")
            plt.plot(x_str, density_out(x_str), c="b", label="out-strength")
            plt.xlim(args["strength-dist-plot-xlim"])
            plt.ylim(bottom=0)
            plt.xlabel('strength')
            plt.title(network)
            plt.legend()
            fig.tight_layout()
            fig.savefig(fileName)
            plt.close()

    if args["comb-plot-degree-and-strength-dist"]:
        densityDict = dict()
        x_deg = np.linspace(
            args["degree-dist-plot-xlim"][0],
            args["degree-dist-plot-xlim"][1],
            200)
        x_str = np.linspace(
            args["strength-dist-plot-xlim"][0],
            args["strength-dist-plot-xlim"][1],
            200)
        for network in outputs:
            refCouplingMtrx = outputs[network]
            refAdjacency = refCouplingMtrx != 0

            degrees_in = refAdjacency.sum(axis=1)
            degrees_out = refAdjacency.sum(axis=0)

            with np.errstate(invalid='ignore'):
                strengths_in = np.nan_to_num(refCouplingMtrx.sum(axis=1) / degrees_in)
                strengths_out = np.nan_to_num(refCouplingMtrx.sum(axis=0) / degrees_out)

            densityDict[network] = {
                "density_indeg": gaussian_kde(degrees_in),
                "density_outdeg": gaussian_kde(degrees_out),
                "density_instr": gaussian_kde(strengths_in),
                "density_outstr": gaussian_kde(strengths_out),
            }

        #### degree distribution ###########################################
        fig = plt.figure()
        fileName = args["save-directory"] + "degree_comb.png"
        density_in = gaussian_kde(degrees_in)
        density_out = gaussian_kde(degrees_out)
        for network in outputs:
            color = np.random.rand(3,)
            plt.plot(x_deg, densityDict[network]["density_indeg"](x_deg), c=color, linestyle="-", label=network)
            plt.plot(x_deg, densityDict[network]["density_outdeg"](x_deg), c=color, linestyle="--")
        plt.xlim(args["degree-dist-plot-xlim"])
        plt.ylim(bottom=0)
        plt.xlabel('degree')
        plt.title(network)
        plt.legend()
        fig.tight_layout()
        fig.savefig(fileName)
        plt.close()

        #### strength distribution #########################################
        fig = plt.figure()
        fileName = args["save-directory"] + "strength_comb.png"
        density_in = gaussian_kde(strengths_in)
        density_out = gaussian_kde(strengths_out)
        for network in outputs:
            color = np.random.rand(3,)
            plt.plot(x_str, densityDict[network]["density_instr"](x_str), c=color, linestyle="-", label=network)
            plt.plot(x_str, densityDict[network]["density_outstr"](x_str), c=color, linestyle="--")
        plt.xlim(args["strength-dist-plot-xlim"])
        plt.ylim(bottom=0)
        plt.xlabel('strength')
        plt.title(network)
        plt.legend()
        fig.tight_layout()
        fig.savefig(fileName)
        plt.close()

    # print(outputs)

if __name__ == "__main__":
    main()
