'''
Effective Connectivity Estimation based on Noise-induced Covariance Method
@ Frankie Yeung (2021 Jun)
'''
import os,glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm,inv
from sklearn.mixture import GaussianMixture

out = 'cov-out/'
plot = 'cov-plot/'

def makeDirectory(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def printOuterProdSum(file, avgStates, statesMatrix, shift):
    ''' print time-lagged outer product sum to file '''
    print('starting printTimeCovOuterProdSum on file %s, shift %d'%(file,shift))
    matrixSum = 0
    demeanStatesMatrix = statesMatrix.T-avgStates
    iter = demeanStatesMatrix.shape[0]
    size = demeanStatesMatrix.shape[1]
    _ = np.empty((size,size))
    for t in tqdm(range(iter-shift)):
        matrixSum += np.outer(demeanStatesMatrix[t+shift],demeanStatesMatrix[t],_)
    np.save(out+file,matrixSum)
    return matrixSum

def estimateInfoMatrix(tau, K0files, K1files):
    ''' estimate info matrix Qij based on outer product calculations '''
    print('starting estimateInfoMatrix on K0files '+','.join(K0files)+', K1files '+','.join(K1files))
    K0 = sum([np.load(f) for f in K0files])
    K1 = sum([np.load(f) for f in K1files])
    print('taking logm')
    infoMatrix = logm(K1.dot(inv(K0)))/tau
    np.save(out+'info.npy',infoMatrix)
    return infoMatrix

def covMethodReconstruction(tau, statesMatrix, numBatch=1):
    '''
    container function to split calculations in batch and
    reconstruct by aggregating calculation batches
    '''
    [makeDirectory(dir) for dir in [out,plot]]
    timeSteps = statesMatrix.shape[1]
    timeStepsPerBatch = int(timeSteps/numBatch)
    avgStates = statesMatrix.mean(axis=1)
    np.save(out+'avg.npy',avgStates)
    for i in range(numBatch):
        t0 = i*timeStepsPerBatch
        t1 = (i+1)*timeStepsPerBatch
        file0 = 'outer0_t=%06dto%06d.npy'%(t0,t1)
        file1 = 'outer1_t=%06dto%06d.npy'%(t0,t1)
        tmpStatesMatrix = statesMatrix[:,t0:t1]
        printOuterProdSum(file0,avgStates,statesMatrix,0)
        printOuterProdSum(file1,avgStates,statesMatrix,1)
    K0files = glob.glob(out+'outer0_t=*.npy')
    K1files = glob.glob(out+'outer1_t=*.npy')
    infoMatrix = estimateInfoMatrix(tau,K0files,K1files)
    return infoMatrix

def reconstructAdjacency(M, Truth, pThres=.5, stdThres=2, plotFreq=100):
    '''
    reconstruct adjacency matrix using Gaussian mixture model on
    each column of info matrix Mij
    '''
    print('starting reconstructAdjacency on pThres %.2f, stdThres %.2f, plotFreq %d'%
        (pThres,stdThres,plotFreq))
    n = M.shape[0]
    adjMatrix = np.zeros((n,n))
    for j in tqdm(range(n)):
        x = list(range(n))
        x = x[:j]+x[j+1:]
        m = M[x,j].reshape(-1,1)
        g = GaussianMixture(n_components=2,random_state=0).fit(m)
        prob = g.predict_proba(m)
        group0Mean = g.means_[0,0]
        group1Mean = g.means_[1,0]
        nullIdx = np.argmin(np.abs(g.means_))
        nullGroupVar = g.covariances_[nullIdx,0]
        if np.abs(group0Mean-group1Mean)<stdThres*np.sqrt(nullGroupVar): continue
        groupLabels = []
        for p in prob:
            if p[nullIdx]>pThres: groupLabels.append(nullIdx)
            else: groupLabels.append(1-nullIdx)
        groupLabels = np.array(groupLabels)
        adjMatrix[x,j] = (groupLabels!=nullIdx)
        if j%plotFreq==0:
            x = np.array(x)
            m = m.flatten()
            a0 = g.means_[nullIdx,0]
            a1 = g.means_[1-nullIdx,0]
            b0 = np.sqrt(nullGroupVar)
            i0 = (groupLabels==nullIdx)
            i1 = (groupLabels!=nullIdx)
            fig = plt.figure()
            plt.scatter(x[i0],m[i0],c='k',s=.5)
            plt.scatter(x[i1],m[i1],c='r',s=.5)
            for i in [-std_thres,0,std_thres]:
                plt.axhline(y=a0+i*b0,c='k',ls='--',label='$\mu_0+(%.1f)\sigma_0$'%i)
            plt.axhline(y=a1,c='r',ls='--',label='$\mu_1$')
            FP = (1-Truth[:,j]).dot(adjMatrix[:,j])
            FN = Truth[:,j].dot(1-adjMatrix[:,j])
            NL = Truth[:,j].sum()
            plt.title('$(j=%d)\;FP=%d,FN=%d,N_L=%d$'%(j,FP,FN,NL))
            plt.legend()
            fig.tight_layout()
            fig.savefig(plot+'Mij_j=%d.png'%j)
            plt.close()
    np.save(out+'adj_p=%.2f_std=%.2f.npy'%(pThres,stdThres),adjMatrix)
    return adjMatrix

def main():
    pass

if __name__=='__main__':
    main()
