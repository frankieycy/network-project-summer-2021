'''
Functional Connectivity Estimation based on Filtered Normalized Cross-Correlation Hsitogram (FNCCH)
@ Frankie Yeung (2021 May)
'''
import os
from tqdm import tqdm
import numpy as np
import pycorrelate as pc
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('text', usetex=True) # comment this out if no tex distribution is installed
plt.switch_backend('agg')

def saveTxtCoupling(file, Coupling):
    ''' save coupling matrix in "i j gji" format '''
    a = Coupling.T
    idx = np.argwhere(a!=0).T
    np.savetxt(file,np.column_stack((idx[0],idx[1],a[idx[0],idx[1]])),fmt='%d %d %.10f')

class FunctionalNetwork:
    def __init__(self, spkTimeFile, withIdx=False):
        '''
        @ spkTimeFile   : name of spike time stamp file
        @ withIdx       : whether the start of each row carries the node index
        '''
        self.readSpkTimes(spkTimeFile, withIdx)
        self.setNpSpkDict()

    def readSpkTimes(self, file, withIdx=False):
        '''
        * read in spike time stamp file
        * the following format is assumed
            TotalSpikeCounts TimeStamp1 TimeStamp2 ...
            or
            NodeIndex TotalSpikeCounts TimeStamp1 TimeStamp2 ...
        '''
        with open(file) as f:
            spkTimes = f.readlines()
        self.size = len(spkTimes)
        self.spkDict = {i: [int(t) for t in spkTimes[i].split()[(2 if withIdx else 1):]] for i in range(self.size)}
        self.setNpSpkDict()

    def setNpSpkDict(self):
        ''' convert spike time stamps to np array format '''
        for i in range(self.size): self.spkDict[i] = np.array(self.spkDict[i])

    def crossCorrHistogram(self, i, j, w):
        ''' cross-correlation histogram of window width w '''
        ti = self.spkDict[i]
        tj = self.spkDict[j]

        if len(ti)==0 or len(tj)==0: return np.zeros(w)
        tau = np.arange(-w/2, w/2+1, dtype=int)
        a = 1/np.sqrt(len(ti) * len(tj))
        C = pc.pcorrelate(ti, tj, tau)
        return a * np.array(C)

    def reconstruct(self, w):
        '''
        * network reconstruction via FNCCH method
        * threshold criteria imposed (exc nodes: 2 sd, inh nodes: 1 sd)
        '''
        print('reconstructing network via FNCCH method with cross-correlation window w = %d' % w)
        print('calculation logs are stored in folder out-w=%d/' % w)
        self.outFolder = 'out-w=%d' % w + '/'
        if not os.path.exists(self.outFolder): os.makedirs(self.outFolder)

        n = self.size
        self.Coupling = np.zeros((n,n))
        t_exc = []; t_inh = []
        C_exc = []; C_inh = []
        idx_exc = []; idx_inh = []
        pbar = tqdm(total=n*(n-1)//2)

        for i in range(self.size):
            for j in range(i+1,self.size):
                pbar.update()
                C = self.crossCorrHistogram(i,j,w)
                if not C.any(): continue # uncorrelated at all
                C -= np.mean(C) # normalization by subtraction of mean
                absC = np.abs(C)
                tau = np.argwhere(absC==np.max(absC)).flatten() # multiple max in absC
                posTau = tau[tau>w//2]; negTau = tau[tau<=w//2]
                Tau = []
                if posTau.size: Tau.append(posTau[0])   # i->j (take first value in array)
                if negTau.size: Tau.append(negTau[-1])  # j->i (take last value in array)
                for t in Tau:
                    idx = (j,i) if t>w//2 else (i,j) # link direction
                    if C[t]>0: # peak (>0)
                        t_exc.append(t)
                        idx_exc.append(idx)
                        C_exc.append(C[t])
                    else: # trough (<0)
                        t_inh.append(t)
                        idx_inh.append(idx)
                        C_inh.append(C[t])
        pbar.close()

        ''' imposing threshold criteria (exc nodes: 2 sd, inh nodes: 1 sd) '''
        nonEmptyExcLinks = False
        if C_exc:
            t_exc = np.array(t_exc)
            C_exc = np.array(C_exc)
            idx_exc = np.array(idx_exc)
            np.save(self.outFolder + 't_exc.npy', t_exc)
            np.save(self.outFolder + 'C_exc.npy', C_exc)
            np.save(self.outFolder + 'idx_exc.npy', idx_exc.T)

            absC_exc = C_exc
            thres_exc = np.mean(absC_exc) + 2*np.std(absC_exc)
            i_exc = np.argwhere(absC_exc>thres_exc).flatten()
            if i_exc.size>0:
                nonEmptyExcLinks = True
                t_exc = t_exc[i_exc]
                C_exc = C_exc[i_exc]
                idx_exc = idx_exc[i_exc].T
                self.Coupling[idx_exc[0],idx_exc[1]] = C_exc
                np.save(self.outFolder + 't_exc_thres.npy', t_exc)
                np.save(self.outFolder + 'C_exc_thres.npy', C_exc)
                np.save(self.outFolder + 'idx_exc_thres.npy', idx_exc)

        nonEmptyInhLinks = False
        if C_inh:
            t_inh = np.array(t_inh)
            C_inh = np.array(C_inh)
            idx_inh = np.array(idx_inh)
            np.save(self.outFolder + 't_inh.npy', t_inh)
            np.save(self.outFolder + 'C_inh.npy', C_inh)
            np.save(self.outFolder + 'idx_inh.npy', idx_inh.T)

            absC_inh = np.abs(C_inh)
            thres_inh = np.mean(absC_inh) + 1*np.std(absC_inh)
            i_inh = np.argwhere(absC_inh>thres_inh).flatten()
            if i_inh.size>0:
                nonEmptyInhLinks = True
                t_inh = t_inh[i_inh]
                C_inh = C_inh[i_inh]
                idx_inh = idx_inh[i_inh].T
                self.Coupling[idx_inh[0],idx_inh[1]] = C_inh
                np.save(self.outFolder + 't_inh_thres.npy', t_inh)
                np.save(self.outFolder + 'C_inh_thres.npy', C_inh)
                np.save(self.outFolder + 'idx_inh_thres.npy', idx_inh)

        np.save(self.outFolder + 'Coupling.npy', self.Coupling)
        saveTxtCoupling(self.outFolder + 'Coupling.txt', self.Coupling)

        self.log = {
            't_exc'     : t_exc     if nonEmptyExcLinks else np.array([]),
            'C_exc'     : C_exc     if nonEmptyExcLinks else np.array([]),
            'idx_exc'   : idx_exc   if nonEmptyExcLinks else np.array([[],[]]),
            't_inh'     : t_inh     if nonEmptyInhLinks else np.array([]),
            'C_inh'     : C_inh     if nonEmptyInhLinks else np.array([]),
            'idx_inh'   : idx_inh   if nonEmptyInhLinks else np.array([[],[]]),
            'Coupling'  : self.Coupling
        }

    def physicalFilter(self, w, t, idx, deltaT, deltaD, tThres, vThres):
        '''
        @ deltaT = physical time unit (ms)
        @ deltaD = physical distance unit (micron)
        * impose time & distance criteria to remove unphysical connections
        '''
        tFilter = (deltaT*np.abs(t-w/2)>=tThres)
        n = int(np.sqrt(self.size+1))
        gridPos = {i: deltaD*np.array([(i+1)//n,(i+1)%n]) for i in range(self.size)}
        with np.errstate(invalid='ignore'):
            propSpeed = np.nan_to_num(np.array([np.sqrt(np.sum((gridPos[idx[0,i]]-gridPos[idx[1,i]])**2)) \
                for i in range(len(t))])/t)
        vFilter = (propSpeed<=vThres)
        return tFilter * vFilter

    def applyPhysicalFilter(self, w, deltaT, deltaD, tThres, vThres):
        ''' apply filter to remove unphysical reconstructions '''
        t = np.concatenate([self.log['t_exc'],self.log['t_inh']])
        C = np.concatenate([self.log['C_exc'],self.log['C_inh']])
        idx = np.concatenate([self.log['idx_exc'],self.log['idx_inh']],axis=1)
        i = self.physicalFilter(w, t, idx, deltaT, deltaD, tThres, vThres)
        t = t[i]; C = C[i]; idx = idx[:,i]

        n = self.size
        self.Coupling_phyfilter = np.zeros((n,n))
        self.Coupling_phyfilter[idx[0],idx[1]] = C
        np.save(self.outFolder + 't_phyfilter.npy', t)
        np.save(self.outFolder + 'C_phyfilter.npy', C)
        np.save(self.outFolder + 'idx_phyfilter.npy', idx)
        np.save(self.outFolder + 'Coupling_phyfilter.npy', self.Coupling_phyfilter)
        saveTxtCoupling(self.outFolder + 'Coupling_phyfilter.txt', self.Coupling_phyfilter)

        self.log.update({
            't_phyfilter'           : t,
            'C_phyfilter'           : C,
            'idx_phyfilter'         : idx,
            'Coupling_phyfilter'    : self.Coupling_phyfilter
        })

    # ======================================================================== #
    # plot & print functions

    def plotCrossCorrHistogram(self, file, i, j, w, title=None):
        ''' plot cross-correlation histogram over tau = [-w/2,w/2] '''
        tau = np.arange(-w/2,w/2,dtype=int)
        C = self.crossCorrHistogram(i,j,w)
        m = np.mean(C)
        t = np.argmax(np.abs(C-m))
        fig = plt.figure()
        plt.scatter(tau,C,c='k',s=.5,label='CCH')
        plt.scatter(t-w/2,C[t],c='r',s=2)
        plt.axhline(y=m,c='k',ls='--',label='mean level')
        plt.annotate(r'$C(\tau^*)$',((t-w/2)*1.01,C[t]*1.01),color='r')
        if title: plt.title(title)
        plt.xlabel(r'$\tau$')
        plt.ylabel('cross-correlation')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def printSpkTime(self, file, withIdx=False):
        ''' print spike time stamps (self.spkDict) to file '''
        print('printing spkie time stamps to %s' % file)
        f = open(file,'w')
        for i in range(self.size):
            f.write((str(i) + ' ' if withIdx else '') + \
            str(len(self.spkDict[i])) + ' ' + ' '.join(str(x) for x in self.spkDict[i]) + '\n')
        f.close()
