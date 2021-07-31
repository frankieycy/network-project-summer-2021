'''
Implementation of Logistic & FHN Network Models
@ Frankie Yeung (2020 Dec)
'''
import os,glob
import util
# import istarmap
import pandas as pd
import numpy as np
from numba import njit,prange
from scipy.linalg import logm,inv,cholesky,eig
from scipy.stats import gaussian_kde,norm,expon,lognorm,pareto
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from time import time
from tqdm import tqdm
from multiprocessing import Pool,cpu_count
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('text', usetex=True) # comment this out if no tex distribution is installed
plt.switch_backend('agg')

def signedNormal(sign, mean=0, spread=1):
    # normal random number restricted to be positive
    # WARNING: resulting distribution is non-normal due to sign restriction
    # for use in setAsGaussianRefGraph()
    assert (sign=='pos' or sign=='neg')
    a = np.random.normal(mean,spread)
    if sign=='pos' and a>0: return a
    elif sign=='neg' and a<0: return a
    else: return signedNormal(sign,mean,spread)

@njit
def normalVector(mean=0, spread=1, size=1):
    # vector of normal random numbers (faster than np.random.normal())
    # SPEED NEEDS VERIFICATION
    a = np.empty(size)
    for i in range(size):
        a[i] = np.random.normal(mean,spread)
    return a

@njit
def outer(x,y):
    # outer product (faster than np.outer())
    # SPEED NEEDS VERIFICATION
    a = np.empty((x.size,y.size))
    for i in range(x.size):
        for j in range(y.size):
            a[i,j] = x[i]*y[j]
    return a

@njit
def outerSubtract(x,y):
    # outer subtraction
    # in vector x and y and return matrix with entries y[j]-x[i]
    a = np.empty((x.size,y.size))
    for i in range(x.size):
        for j in range(y.size):
            a[i,j] = y[j]-x[i]
    return a

#==============================================================================#

class graph:
    def __init__(self):
        self.internalGraph = False  # graph is internally generated (via randomDirectedWeightedGraph() etc.)

    def loadGraph(self, couplingFile, multiplier=None):
        # load graph (connectivity & couplings) from file
        # file format: i j Coupling(j->i)
        # file used: DIV25_PREmethod
        # unstated node pairs are unconnected
        print(' loading coupling file from %s ...'%couplingFile)
        data = np.loadtxt(couplingFile)
        if multiplier: data[:,2] *= multiplier # scale the couplings
        indices = list(map(tuple,(data[:,(1,0)]-1).astype(int)))
        size = np.max(indices)+1

        self.size = size
        self.Coupling = np.zeros((size,size))
        for j in range(np.size(data,0)):
            self.Coupling[indices[j]] = data[j,2]
        self.Adjacency = (self.Coupling!=0).astype(int)

        self.initialize()

    def loadNpGraph(self, couplingFile):
        # load graph (connectivity & couplings) from npy file
        print(' loading npy coupling file from %s ...'%couplingFile)
        self.Coupling = np.load(couplingFile)
        self.size = self.Coupling.shape[0]
        self.Adjacency = (self.Coupling!=0).astype(int)
        self.initialize()

    def randomDirectedWeightedGraph(self, size, connectProb, couplingMean, couplingSpread):
        # random directed graph with Gaussian couplings
        self.internalGraph = True
        self.size = size
        self.Adjacency = (np.random.uniform(size=(size,size))<connectProb).astype(int)
        self.Coupling = np.random.normal(couplingMean,couplingSpread,size=(size,size))*self.Adjacency
        self.initialize()

    def randomUnidirectedWeightedGraph(self, size, connectProb, couplingMean, couplingSpread):
        # random uni-directed graph with Gaussian couplings
        self.internalGraph = True
        self.size = size
        self.Adjacency = np.triu((np.random.uniform(size=(size,size))<connectProb).astype(int),k=0)
        self.Coupling = np.random.normal(couplingMean,couplingSpread,size=(size,size))*self.Adjacency
        self.initialize()

    def setAsGaussianRefGraph(self):
        # USED FOR **LOGISTIC** REFERENCE NETWORKS (BAD METHOD!)
        # (reference network) random directed graph with Gaussian couplings
        # positive couplings sampled from a Gaussian constructed from original positive couplings
        # negative couplings sampled from a Gaussian constructed from original negative couplings
        # NOTE: this REPLACES original couplings!
        print(' setting as Gaussian reference graph ...')
        Coupling = self.Coupling

        posCoupling = Coupling[Coupling>0]
        negCoupling = Coupling[Coupling<0]
        idx_pos = np.where(Coupling>0)
        idx_neg = np.where(Coupling<0)
        mu_pos = posCoupling.mean()
        sigma_pos = posCoupling.std()
        mu_neg = negCoupling.mean()
        sigma_neg = negCoupling.std()

        self.Coupling = np.zeros((self.size,self.size))
        self.Coupling[idx_pos] = [signedNormal('pos',mu_pos,sigma_pos) for _ in range(len(idx_pos[0]))]
        self.Coupling[idx_neg] = [signedNormal('neg',mu_neg,sigma_neg) for _ in range(len(idx_neg[0]))]
        # self.Coupling[idx_pos] = abs(np.random.normal(mu_pos,sigma_pos,len(idx_pos[0])))
        # self.Coupling[idx_neg] = -abs(np.random.normal(mu_neg,sigma_neg,len(idx_neg[0])))

        self.initialize()

    def setAsShuffledRefGraph(self, axis=0):
        # USED FOR **FHN** REFERENCE NETWORKS
        # (reference network) graph with shuffled entries in rows (axis=0) or cols (axis=1)
        # shuffle rows: preserve in-degrees and in-strengths
        # shuffle cols: preserve out-degrees and out-strengths
        # NOTE: this REPLACES original couplings!
        print(' setting as row/col-shuffled reference graph ...')
        if axis==0: list(map(np.random.shuffle,self.Coupling)) # shuffle entries in rows
        elif axis==1: # shuffle entries in cols
            transposeCoupling = self.Coupling.T
            list(map(np.random.shuffle,transposeCoupling))
            self.Coupling = transposeCoupling.T
        self.initialize()

    def setAsEntryShuffledRefGraph(self):
        # USED FOR **FHN** REFERENCE NETWORKS
        # (reference network) graph with non-zero entries shuffled
        # shuffle entries: preserve in- and out-degrees and re-distribute in- and out-strengths
        # NOTE: this REPLACES original couplings!
        print(' setting as entry-shuffled reference graph ...')
        Coupling = self.Coupling
        nonzeroCoupling = Coupling[Coupling!=0]
        np.random.shuffle(nonzeroCoupling)
        idx_nonzero = np.where(Coupling!=0)
        self.Coupling = np.zeros((self.size,self.size))
        self.Coupling[idx_nonzero] = nonzeroCoupling
        self.initialize()

    def setAsEntryReplacedRefGraph(self):
        # USED FOR **FHN** REFERENCE NETWORKS
        # (reference network) random directed graph with Gaussian couplings
        # couplings sampled from a Gaussian constructed from original couplings
        # replace entries: preserve in- and out-degrees and re-distribute in- and out-strengths
        # NOTE: this REPLACES original couplings!
        print(' setting as Gaussian reference graph ...')
        Coupling = self.Coupling
        nonzeroCoupling = Coupling[Coupling!=0]
        mu_nonzero = nonzeroCoupling.mean()
        sigma_nonzero = nonzeroCoupling.std()

        idx_nonzero = np.where(Coupling!=0)
        self.Coupling = np.zeros((self.size,self.size))
        self.Coupling[idx_nonzero] = np.random.normal(mu_nonzero,sigma_nonzero,len(idx_nonzero[0]))
        self.initialize()

    def initialize(self):
        # sparse representation
        self.sparseCoupling = csr_matrix(self.Coupling)
        self.couplingNonZeroIdx = self.sparseCoupling.nonzero()

        # node degrees & strengths
        self.degrees_in = self.Adjacency.sum(axis=1)
        self.degrees_out = self.Adjacency.sum(axis=0)

        self.Coupling_pos = np.where(self.Coupling>0,self.Coupling,0)
        self.Adjacency_pos = (self.Coupling>0).astype(int)
        self.degrees_posIn = self.Adjacency_pos.sum(axis=1)

        with np.errstate(invalid='ignore'):
            self.strengths_in = np.nan_to_num(self.Coupling.sum(axis=1)/self.degrees_in)
            self.strengths_out = np.nan_to_num(self.Coupling.sum(axis=0)/self.degrees_out)
            self.strengths_posIn = np.nan_to_num(self.Coupling_pos.sum(axis=1)/self.degrees_posIn)

        # node classifications by signs of strengths (INFREQUENTLY USED)
        # self.idx_PosIn = np.argwhere(self.strengths_in>0).flatten()
        # self.idx_NegIn = np.argwhere(self.strengths_in<0).flatten()
        # self.idx_PosOut = np.argwhere(self.strengths_out>0).flatten()
        # self.idx_NegOut = np.argwhere(self.strengths_out<0).flatten()
        # self.idx_PosInPosOut = np.argwhere((self.strengths_in>0)&(self.strengths_out>0)).flatten()
        # self.idx_PosInNegOut = np.argwhere((self.strengths_in>0)&(self.strengths_out<0)).flatten()
        # self.idx_NegInPosOut = np.argwhere((self.strengths_in<0)&(self.strengths_out>0)).flatten()
        # self.idx_NegInNegOut = np.argwhere((self.strengths_in<0)&(self.strengths_out<0)).flatten()

    def calcConnectProb(self):
        # calculate empirical connection probability
        return self.Adjacency.sum()/(self.size*(self.size-1))

    def calcSparseness(self):
        # calculate sparseness of coupling matrix
        return (self.Adjacency!=0).sum()/(self.size*(self.size-1))

    def isBidirectional(self):
        # check if adjacency matrix is symmetric
        return np.allclose(self.Adjacency,self.Adjacency.T,0,0)

    def printAdjacency(self, file, cm=''):
        # print adjacency matrix to file
        np.savetxt(file,self.Adjacency,fmt='%d',header=cm)

    def printCoupling(self, file, cm=''):
        # print coupling matrix to file
        np.savetxt(file,self.Coupling,fmt='%.4f',header=cm)

    def printDegStrenToCsv(self, file):
        # print degree & strength measures to csv file
        print(' printing degree & strength measures to %s ...'%file)
        data = {
            'k_in': self.degrees_in,
            'k_out': self.degrees_out,
            'k_posIn': self.degrees_posIn,
            's_in': self.strengths_in,
            's_out': self.strengths_out,
            's_posIn': self.strengths_posIn
        }
        dataFrame = pd.DataFrame(data,columns=list(data.keys()))
        dataFrame.to_csv(file)

#==============================================================================#
#==============================================================================#

class network(graph):
    # CONVENTION: underscore at the end of variable name refers to time series
    def __init__(self):
        self.emailNotify = False
        self.statesLog = [] # log for multiple runs (HIGH MEMORY COST!)

    def setEmail(self, emailFrom, emailPw, emailTo):
        # set up email notifier of program progress (QUITE REDUNDANT!)
        self.emailNotify = True
        self.emailHandler = util.emailHandler(emailFrom, emailPw, emailTo)

    # intrinsic dynamics ======================================================#

    @staticmethod
    @njit
    def intrinsicFunc_FHNfast(size,eps,x,y):
        # FHN DYNAMICS
        # intrinsic dynamics (fast version)
        a = np.empty(size)
        for i in range(size):
            a[i] = (x[i]-x[i]*x[i]*x[i]/3-y[i])/eps
        return a

    @staticmethod
    @njit
    def intrinsicFunc(r,x):
        # intrinsic dynamics
        return r*x*(1-x)

    @staticmethod
    @njit
    def intrinsicFunc_fast(size,r,x):
        # intrinsic dynamics (fast version)
        a = np.empty(size)
        for i in range(size):
            a[i] = r[i]*x[i]*(1-x[i])
        return a

    # coupling function =======================================================#

    @staticmethod
    @njit
    def couplingFunc_diffusive(x,y):
        # diffusive coupling function
        return y-x

    @staticmethod
    @njit
    def couplingFuncDerivY_diffusive(x,y):
        # y-derivative of diffusive coupling function (used in calcInfoMatrix())
        return 1

    @staticmethod
    @njit
    def couplingFunc_synaptic(x,y):
        # synaptic coupling function
        beta1,beta2,y0 = 0.01,1,0
        return 1/beta1*(1+np.tanh(beta2*(y-y0)))

    @staticmethod
    @njit
    def couplingFuncDerivY_synaptic(x,y):
        # y-derivative of synaptic coupling function (used in calcInfoMatrix())
        beta1,beta2,y0 = 0.01,1,0
        a = np.cosh(beta2*(y-y0))
        return beta2/beta1/(a*a)

    # initialization FROM START ===============================================#

    def initDynamics_FHN(self, initStates, initStatesY, epsilon, alpha, noiseCovariance):
        # FHN DYNAMICS
        # initialize node states and set intrinsic coef & noise cov
        print(' initializing dynamics ...')
        self.dynamics = "FHN"
        self.states_ = {i:[] for i in range(self.size)}
        self.time = 0
        self.time_ = [0]
        self.iter = 1

        self.epsilon = epsilon
        self.alpha = alpha

        self.initStates = np.array(initStates)
        self.initStatesY = np.array(initStatesY)
        self.states = self.initStates
        self.statesY = self.initStatesY
        for i in range(self.size):
            self.states_[i].append(self.states[i])

        self.setIntrinsicAndNoise([],noiseCovariance)

    def initDynamics(self, initStates, intrinsicCoef, noiseCovariance):
        # initialize node states and set intrinsic coef & noise cov
        print(' initializing dynamics ...')
        self.dynamics = "Logistic"
        self.states_ = {i:[] for i in range(self.size)}
        self.time = 0
        self.time_ = [0]
        self.iter = 1

        self.initStates = np.array(initStates)
        self.states = self.initStates
        for i in range(self.size):
            self.states_[i].append(self.states[i])

        self.setIntrinsicAndNoise(intrinsicCoef,noiseCovariance)

    def setIntrinsicAndNoise(self, intrinsicCoef, noiseCovariance):
        # set intrinsic coef & noise cov
        self.intrinsicCoef = np.array(intrinsicCoef)
        self.noiseCovariance = np.array(noiseCovariance)
        if np.allclose(self.noiseCovariance,np.zeros((self.size,self.size))):
            self.noiseChol = np.zeros((self.size,self.size))
        else:
            self.noiseChol = cholesky(self.noiseCovariance)
        if np.allclose(self.noiseChol,self.noiseChol[0,0]*np.eye(self.size)):
            self.sigma = self.noiseChol[0,0]

    # initialization FROM FILE ================================================#

    def continueDynamics_FHN(self, file, fileY, epsilon, alpha, noiseCovariance):
        # FHN DYNAMICS
        # continue dynamics from read time series data
        print(' initializing dynamics from %s & %s ...'%(file,fileY))
        self.epsilon = epsilon
        self.alpha = alpha
        self.statesY = np.load(fileY) # fileY is npy file
        self.readDynamics(file)
        self.setIntrinsicAndNoise([],noiseCovariance)

    def continueDynamics(self, file, intrinsicCoef, noiseCovariance):
        # continue dynamics from read time series data
        print(' initializing dynamics from %s ...'%file)
        self.readDynamics(file)
        self.setIntrinsicAndNoise(intrinsicCoef, noiseCovariance)

    def readDynamics(self, file):
        # read time series data from file
        print(' reading dynamics from %s ...'%file)
        if isinstance(file,list):
            # p = Pool(cpu_count()-1)
            # if file[0][-4:]=='.npy': data = np.vstack(p.map(np.load,file)) # [npy files]
            # else: data = np.vstack(p.map(util.loadcsv,file)) # [csv files]
            if file[0][-4:]=='.npy': data = np.vstack([np.load(f) for f in file]) # [npy files]
            else: data = np.vstack([util.loadcsv(f) for f in file]) # [csv files]
        else:
            if file[-4:]=='.npy': data = np.load(file) # npy file
            else: data = util.loadcsv(file) # csv file
        print(' finished reading dynamics ...')

        self.size = data.shape[1]-1
        self.initStates = data[0,1:]
        self.states = data[-1,1:]

        if file[:6]=='(cont)': # cont file for continuation (contains only two time steps)
            self.states_ = {i:[] for i in range(self.size)}
            self.time_ = []
        else: # not cont file
            # self.states_ = {i:data[:,i+1].tolist() for i in range(self.size)} # in most cases, this line is optional
            self.time_ = data[:,0].tolist()

        self.states_np = data[:,1:].T
        self.time = data[-1,0]
        self.timeStep = data[1,0]-data[0,0]
        self.sqrtTimeStep = np.sqrt(self.timeStep)
        self.iter = int(self.time/self.timeStep)

    def saveNpDynamics(self, file):
        # print time series data to file (npy format)
        # npy file for fast loading
        print(' printing dynamics to %s ...'%file)
        np.save(file,np.vstack((self.time_,self.states_np)).T)

    # run program =============================================================#

    def runInBatches(self, files, params, dt, start, end, step):
        # FHN DYNAMICS
        # run dynamics and save time series in batches
        # files: [fileNameX, fileNameY]
        # params: [epsilon, alpha, noiseCovariance, ic, icY] (include last two if start=0)
        iter = int((end-start)/step)
        for i in range(iter):
            print(' <<<< running dynamics in batches >>>> %d/%d'%(i+1,iter))
            if start+i*step==0: self.initDynamics_FHN(params[3],params[4],params[0],params[1],params[2])
            else: self.continueDynamics_FHN('(cont)%s_%.1esteps.csv'%(files[0],start+i*step),'(cont)%s_%.1esteps.npy'%(files[1],start+i*step),
                params[0],params[1],params[2])
            self.runDynamics(dt,start+(i+1)*step)
            self.saveNpDynamics('%s_%.1eto%.1esteps.npy'%(files[0],start+i*step,start+(i+1)*step))
            self.printContFile_FHN('%s_%.1esteps.csv'%(files[0],start+(i+1)*step),'%s_%.1esteps.npy'%(files[1],start+(i+1)*step))

    def findPeaksInBatches(self, file, peakFile, timeRange, h, d, batches=1):
        # find peaks in batches
        # NOTE: length(files) must be divisible by batches
        # file: time series file pattern
        # peakFile: peak count file name
        files = glob.glob(file)
        files.sort(key=lambda f: int(float(f[-16:-9]))) # f[-16:-9] is ending time step (scientific format)
        batchFiles = len(files)//batches
        batchTime = (timeRange[1]-timeRange[0])//batches
        peakCounts = []
        for i in range(batches):
            print(' <<<< finding peaks in batches >>>> %d/%d'%(i+1,batches))
            self.readDynamics(files[i*batchFiles:(i+1)*batchFiles])
            self.findPeaks(height=h,distance=d)
            peakCounts.append(np.array(self.peakCount))
            np.save('%s_t=%dto%d_h=%.2f_d=%d.npy'%(peakFile,timeRange[0]+i*batchTime,timeRange[0]+(i+1)*batchTime,h,d),self.peakCount) # npy peak count
            self.printPeakTime('%s_t=%dto%d_h=%.2f_d=%d.csv'%(peakFile,timeRange[0]+i*batchTime,timeRange[0]+(i+1)*batchTime,h,d)) # csv peak time
        np.save('%s_t=%dto%d_h=%.2f_d=%d.npy'%(peakFile,timeRange[0],timeRange[1],h,d),sum(peakCounts))

    def plotDynamicsForNodes(self, folder, file, nodes, iterSlice=None, ylimRange=None, withPeaks=False):
        # plot time series for each node
        # plots are stored in folder
        if not os.path.exists(folder): os.mkdir(folder)
        if withPeaks:
            for i in nodes: self.plotDynamicsWithPeaks(i,folder+'/'+file+'_node%04d.png'%i,iterSlice=iterSlice,ylimRange=ylimRange)
        else:
            for i in nodes: self.plotDynamics(folder+'/'+file+'_node%04d.png'%i,nodes=[i],iterSlice=iterSlice,ylimRange=ylimRange,color='k')

    # generate dynamics =======================================================#

    def getStateChanges_FHNx(self):
        # FHN DYNAMICS (x-dimension)
        # instantaneous node changes

        #### IMPLEMENTATION 3 (WeightedCoupling) #### DIFFUSIVE COUPLING FUNC
        # WeightedCoupling = self.sparseCoupling.multiply(csr_matrix(
        #     (self.states[self.couplingNonZeroIdx[1]]-self.states[self.couplingNonZeroIdx[0]],self.couplingNonZeroIdx),
        #     shape=(self.size,self.size)
        # ))

        #### IMPLEMENTATION 4 (WeightedCoupling) #### SYNAPTIC COUPLING FUNC
        myRow = self.couplingFunc_synaptic(None,self.states)
        WeightedCoupling = self.sparseCoupling.multiply(myRow)

        randomVector = np.random.normal(size=self.size)

        changes = (self.intrinsicFunc_FHNfast(self.size,self.epsilon,self.states,self.statesY)+\
            WeightedCoupling.sum(axis=1).A1)*self.timeStep+\
            self.sigma*randomVector*self.sqrtTimeStep

        return changes

    def getStateChanges_FHNy(self):
        # FHN DYNAMICS (y-dimension)
        # instantaneous node changes
        return (self.states+self.alpha)*self.timeStep

    def getStateChanges(self):
        # instantaneous node changes
        # changes as an np array

        #### IMPLEMENTATION 1 (WeightedCoupling) #### STANDARD
        # WeightedCoupling = np.empty((self.size,self.size))
        # for i in range(self.size):
        #     WeightedCoupling[i] = self.couplingFunc_synaptic(self.states[i],self.states)
        # WeightedCoupling *= self.Coupling

        #### IMPLEMENTATION 2 (WeightedCoupling) #### DEPRECATED
        # WeightedCoupling = self.sparseCoupling.multiply(outerSubtract(self.states,self.states))

        #### IMPLEMENTATION 3 (WeightedCoupling) #### DIFFUSIVE COUPLING FUNC
        WeightedCoupling = self.sparseCoupling.multiply(csr_matrix(
            (self.states[self.couplingNonZeroIdx[1]]-self.states[self.couplingNonZeroIdx[0]],self.couplingNonZeroIdx),
            shape=(self.size,self.size)
        ))

        #### IMPLEMENTATION 4 (WeightedCoupling) #### SYNAPTIC COUPLING FUNC
        # using feature of synaptic coupling function
        # myRow = self.couplingFunc_synaptic(None,self.states)
        # WeightedCoupling = self.sparseCoupling.multiply(myRow)
        # WeightedCoupling = np.multiply(self.Coupling,myRow) # DEPRECATED

        randomVector = np.random.normal(size=self.size)

        #### IMPLEMENTATION 1 (changes) #### STANDARD
        # changes = (self.intrinsicFunc(self.intrinsicCoef,self.states)+\
        #     WeightedCoupling.sum(axis=1))*self.timeStep+\
        #     self.noiseChol.dot(randomVector)*self.sqrtTimeStep

        #### IMPLEMENTATION 2 (changes) #### IID NOISE
        # changes = (self.intrinsicFunc_fast(self.size,self.intrinsicCoef,self.states)+\
        #     WeightedCoupling.sum(axis=1))*self.timeStep+\
        #     self.sigma*randomVector*self.sqrtTimeStep
        #     # self.sigma*normalVector(size=self.size)*self.sqrtTimeStep

        #### IMPLEMENTATION 3 (changes) #### IID NOISE & SPARSE COUPLINGS (VERY FAST FOR LARGE NETWORKS)
        changes = (self.intrinsicFunc_fast(self.size,self.intrinsicCoef,self.states)+\
            WeightedCoupling.sum(axis=1).A1)*self.timeStep+\
            self.sigma*randomVector*self.sqrtTimeStep

        return changes

    def runDynamics(self, timeStep, totIter, silent=True):
        # iterate node states according to dynamical equations
        self.timeStep = timeStep
        self.sqrtTimeStep = np.sqrt(timeStep)
        self.endTime = timeStep*totIter

        pbar = tqdm(total=totIter) # progress bar
        pbar.update(self.iter)

        # DEPRECATED TIME COUNTER
        # startTimer = time()

        while self.iter<totIter:
            if self.dynamics == "FHN":
                #### FHN INTRINSIC DYNAMICS ####
                states_new = self.states+self.getStateChanges_FHNx()
                self.statesY += self.getStateChanges_FHNy()
                self.states = states_new
            elif self.dynamics == "Logistic":
                #### LOGISTIC INTRINSIC DYNAMICS ####
                self.states += self.getStateChanges()

            for i in range(self.size):
                self.states_[i].append(self.states[i])
            self.time += self.timeStep
            self.time_.append(self.time)
            self.iter += 1

            if silent:
                pbar.update()
                # DEPRECATED TIME COUNTER
                # print(' t = %7.2f | %c %.2f %%\r'%(self.time,util.progressBars[self.iter%4],100*self.time/self.endTime),end='')
            elif self.iter%100==0:
                print(' t = %7.2f/%7.2f | '%(self.time,self.endTime)+\
                    ' | '.join(['x%d = %7.2f'%(i,x) for i,x in enumerate(self.states)][0:min(4,self.size)]),end='')
                if self.size>4: print(' ...')

            if self.emailNotify:
                if self.iter%(totIter//5)==0:
                    self.emailHandler.sendEmail('running: t = %.2f/%.2f'%(self.time,self.endTime))

        pbar.close()

        # DEPRECATED TIME COUNTER
        # endTimer = time()
        # print('\n runDynamics() takes %.2f seconds'%(endTimer-startTimer))

        self.states_np = np.array([self.states_[i] for i in range(self.size)])
        # self.statesLog.append(self.states_) # HIGH MEMORY COST!

    # post-dynamics processing ================================================#

    def removeTransient(self, transientSteps):
        # remove (initial) transient states
        # look into dynamics plot to determine what to remove
        print(' removing transient states ...')
        self.iter -= transientSteps
        self.states_np = self.states_np[:,transientSteps:]
        del self.time_[:transientSteps]
        # optional because state dict is infrequently used for processing
        # for i in range(self.size): del self.states_[i][:transientSteps]

    def setSteadyStates(self, file=None):
        # set steady states with a noise-free network
        # or load steady states from file
        # REQUIRE: noiseCovariance = 0 (run a noise-free network separately)
        print(' setting steady states '+('from %s '%file if file else '')+'...')
        if file: # csv format
            data = np.loadtxt(file,delimiter=',',skiprows=1)
            self.steadyStates = data[1:]
        else: self.steadyStates = self.states_np[:,-1] # last time step

    def calcTimeAvg(self):
        # compute time average of node states
        # should approx steady states
        print(' computing time average of states ...')
        self.avgStates = self.states_np.mean(axis=1)
        if self.emailNotify: self.emailHandler.sendEmail('calcTimeAvg() completes')

    def setTimeAvg(self, file):
        # load time average of node states from file
        print(' setting time average of states from %s ...'%file)
        self.avgStates = np.load(file) # npy format

    def printTimeAvg(self, file):
        # print time average of node states to file
        print(' printing time average of states to %s ...'%file)
        np.save(file,self.avgStates) # npy format

    def calcStatesFluc(self):
        # compute mean & s.d. of fluctuations around steady states
        # REQUIRE: steady states have to be computed or loaded first
        print(' computing mean & s.d. of fluctuations around steady states ...')
        self.statesFluc = self.states_np-self.steadyStates.reshape(-1,1) # fluc ard steady states
        self.flucMean = self.statesFluc.mean(axis=1)
        self.flucSd = self.statesFluc.std(axis=1)
        if self.emailNotify: self.emailHandler.sendEmail('calcStatesFluc() completes')

    # analysis ================================================================#

    def calcInfoMatrix(self):
        # compute information matrix of network (theoretical)
        # diagonal entries not usable (because inaccurate)
        print(' computing info matrix (Qij) ...')

        #### IMPLEMENTATION 1 #### STANDARD
        # self.InfoMatrix = np.empty((self.size,self.size))
        # for i in range(self.size):
        #     self.InfoMatrix[i] = self.couplingFuncDerivY_synaptic(self.steadyStates[i],self.steadyStates)
        #     util.showProgress(i+1,self.size)

        #### IMPLEMENTATION 2 #### DIFFUSIVE COUPLING FUNC
        myRow = self.couplingFuncDerivY_diffusive(None,self.steadyStates)
        self.InfoMatrix = self.sparseCoupling.multiply(myRow).A

        #### IMPLEMENTATION 3 #### SYNAPTIC COUPLING FUNC
        # using feature of synaptic coupling function
        # myRow = self.couplingFuncDerivY_synaptic(None,self.steadyStates)
        # self.InfoMatrix = self.sparseCoupling.multiply(myRow).A
        # self.InfoMatrix = np.multiply(self.Coupling,myRow) # DEPRECATED

        if self.emailNotify: self.emailHandler.sendEmail('calcInfoMatrix() completes')

    #### IMPLEMENTATION 1 #### STANDARD
    def timeCovarianceMatrix(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step

        matrixSum = 0
        statesFluc = self.states_np.T-self.avgStates # fluc ard avg states
        iter = statesFluc.shape[0] # num of rows in statesFluc
        _ = np.empty((self.size,self.size)) # for faster np.outer()

        for t in tqdm(range(iter-shift)):
            matrixSum += np.outer(statesFluc[t+shift],statesFluc[t],_)

        return matrixSum/(iter-shift)

    #### IMPLEMENTATION 2 #### CHUNKED CALCULATIONS (BEST)
    def printTimeCovOuterProdSum(self, shift, file):
        # print outer product sum (for computing time covariance matrix)
        # shift = multiple of time step
        # this is SUGGESTED (calculate outer product sum for each time chunk)
        print(' computing outer product sum and printing to %s ...'%file)

        matrixSum = 0
        statesFluc = self.states_np.T-self.avgStates
        iter = statesFluc.shape[0]
        _ = np.empty((self.size,self.size))

        for t in tqdm(range(iter-shift)):
            matrixSum += np.outer(statesFluc[t+shift],statesFluc[t],_)

        np.save(file,matrixSum)

    #### IMPLEMENTATION 3 #### PARALLEL (MAY BE EVEN SLOWER THAN STANDARD!)
    def timeCovarianceMatrix_parallel(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step

        p = Pool(cpu_count()-1)

        matrixSum = 0
        statesFluc = self.states_np.T-self.avgStates
        iter = statesFluc.shape[0]
        _ = np.empty((self.size,self.size))

        if shift==0:
            for myOuter in tqdm(p.istarmap(np.outer,zip(statesFluc,statesFluc,[_]*iter)),total=iter): matrixSum += myOuter
        else:
            for myOuter in tqdm(p.starmap(np.outer,zip(statesFluc[:-shift],statesFluc[shift:],[_]*iter)),total=iter-shift): matrixSum += myOuter

        return matrixSum/(iter-shift)

    #### IMPLEMENTATION 4 #### NJIT (MAY BE EVEN SLOWER THAN STANDARD!)
    @staticmethod
    @njit(parallel=True,nogil=True)
    def timeCovarianceMatrix_fast(shift, size, states_np, avgStates):
        # compute time covariance matrix (fast version)
        # shift = multiple of time step

        matrixSum = np.zeros((size,size))
        statesFluc = states_np.T-avgStates
        iter = statesFluc.shape[0]
        _ = np.empty((size,size))

        for t in prange(iter-shift):
            matrixSum += np.outer(statesFluc[t+shift],statesFluc[t],_)

        return matrixSum/(iter-shift)

    def estInfoMatrix(self, K0file=None, K1file=None):
        # estimate information matrix of network (empirical)
        # should approx InfoMatrix, compare via plotInfoMatrix
        # load outer prod sum files with K0file & K1file
        print(' estimating info matrix (Mij) ...')

        if K0files and K1files: # load from outer prod sum files (SUGGESTED)
            K0 = np.sum([np.load(f) for f in K0file])
            K1 = np.sum([np.load(f) for f in K1file])
        else: # compute from scratch
            K0 = self.timeCovarianceMatrix(0)
            K1 = self.timeCovarianceMatrix(1)

            # K0 = self.timeCovarianceMatrix_parallel(0)
            # K1 = self.timeCovarianceMatrix_parallel(1)

            # K0 = self.timeCovarianceMatrix_fast(0,self.size,self.states_np,self.avgStates)
            # K1 = self.timeCovarianceMatrix_fast(1,self.size,self.states_np,self.avgStates)

        print(' estimating info matrix (Mij) ... taking logm')
        self.InfoMatrix_est = logm(K1.dot(inv(K0)))/self.timeStep

        if self.emailNotify: self.emailHandler.sendEmail('estInfoMatrix() completes')

    def checkLogmCondition(self):
        # check if condition that log(exp(Q))=Q is satisfied
        print(' checking if condition that log(exp(Q))=Q is satisfied ...')
        eigVal,_ = eig(self.InfoMatrix)
        print(' - condition (tau max_i |Im lambda_i| < pi) is',
            self.timeStep*np.max(np.imag(eigVal))<np.pi)

    def findPeaks(self, height, distance, iterSlice=None):
        # COMMONLY USED!
        # find peaks with a defined threshold (height) & separation (distance) for all nodes
        # peak indices start at 0
        print(' finding peaks ...')
        if not iterSlice: iterSlice = slice(self.iter)
        self.peakHeight = height
        self.peakDict = {}
        self.peakCount = []
        for i in range(self.size):
            peaks,_ = find_peaks(self.states_np[i,iterSlice],height=height,distance=distance)
            self.peakDict[i] = peaks
            self.peakCount.append(len(peaks))

    def printPeakTime(self, file):
        # print peak time (peakDict) to file
        print(' printing peak time to %s ...'%file)
        f = open(file,'w')
        for i in range(self.size): f.write(str(i)+' '+str(len(self.peakDict[i]))+' '+' '.join(str(x) for x in self.peakDict[i])+'\n')
        f.close()

    # plots and outputs =======================================================#

    def plotInfoMatrix(self, file, title=None):
        # plot information matrix: theoretical vs empirical
        # REQUIRE: calcInfoMatrix() and estInfoMatrix() beforehand
        print(' plotting info matrix (Qij vs Mij) to %s ...'%file)

        Q = self.InfoMatrix
        M = self.InfoMatrix_est
        Q = Q[~np.eye(Q.shape[0],dtype=bool)].reshape(Q.shape[0],-1).flatten()
        M = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1).flatten()

        fig = plt.figure()
        plt.scatter(Q,M,s=1,c='k')
        if title: plt.title(title)
        plt.xlabel('$Q_{ij}$')
        plt.ylabel('$M_{ij}$')
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def plotEstInfoMatrix(self, file, title=None):
        # plot estimated information matrix: distribution
        print(' plotting estimated info matrix (Mij distribution) to %s ...'%file)

        M = self.InfoMatrix_est
        M = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1).flatten()
        mean = np.mean(M)
        sd = np.std(M)

        fig = plt.figure()
        x = np.linspace(np.min(M),np.max(M),200)
        density = gaussian_kde(M)
        plt.plot(x,density(x),'k',label='$\sigma=%.1f$'%sd)
        plt.scatter(M,[0]*len(M),s=1,c='k')
        plt.axvline(x=-sd,c='k',ls='--')
        plt.axvline(x=+sd,c='k',ls='--')
        plt.ylim(bottom=0)
        if title: plt.title(title)
        plt.xlabel('$M_{ij}$')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def plotDegreeStrengthDist(self, degreeFile, strengthFile, degreeTitle=None, strengthTitle=None):
        # DEPENDS ONLY ON GRAPH NOT TIME SERIES
        # plot degree and strength distribution
        # require only graph (time series not necessary)
        print(' plotting degree & strength distribution to %s & %s ...'%(degreeFile,strengthFile))

        fig = plt.figure()
        minDeg = np.percentile((self.degrees_in,self.degrees_out),0)
        maxDeg = np.percentile((self.degrees_in,self.degrees_out),98)
        x = np.linspace(minDeg,maxDeg,200)
        density_in = gaussian_kde(self.degrees_in)
        density_out = gaussian_kde(self.degrees_out)
        plt.plot(x,density_in(x),'k',label='in-degrees')
        plt.plot(x,density_out(x),'k--',label='out-degrees')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        if degreeTitle: plt.title(degreeTitle)
        plt.xlabel('degrees $k_i$')
        plt.legend()
        fig.tight_layout()
        fig.savefig(degreeFile)
        plt.close()

        fig = plt.figure()
        minStren = np.percentile((self.strengths_in,self.strengths_out),2)
        maxStren = np.percentile((self.strengths_in,self.strengths_out),98)
        x = np.linspace(minStren,maxStren,200)
        density_in = gaussian_kde(self.strengths_in)
        density_out = gaussian_kde(self.strengths_out)
        plt.plot(x,density_in(x),'k',label='in-strengths')
        plt.plot(x,density_out(x),'k--',label='out-strengths')
        plt.ylim(bottom=0)
        if strengthTitle: plt.title(strengthTitle)
        plt.xlabel('strengths $s_i$')
        plt.legend()
        fig.tight_layout()
        fig.savefig(strengthFile)
        plt.close()

    # DEPRECATED
    def plotStrengthDist(self, file):
        # DEPENDS ONLY ON GRAPH NOT TIME SERIES
        # plot strength distribution (node classifications by signs of strengths)
        # require only graph (time series not necessary)
        print(' plotting log in- & out-strength distribution to %s & %s ...'%(file+'logInstren.png',file+'logOutstren.png'))

        fig = plt.figure()
        a = np.log(self.strengths_in[self.idx_PosIn])
        b = np.log(abs(self.strengths_in[self.idx_NegIn]))
        a = (a-a.mean())/a.std()
        b = (b-b.mean())/b.std()
        minStren = np.percentile(np.concatenate([a,b]),0)
        maxStren = np.percentile(np.concatenate([a,b]),100)
        x = np.linspace(minStren,maxStren,200)
        density_a = gaussian_kde(a)
        density_b = gaussian_kde(b)
        plt.plot(x,density_a(x),'r',label='$\log(s_\mathrm{in})(>0)$')
        plt.plot(x,density_b(x),'b',label='$\log(|s_\mathrm{in}|)(<0)$')
        plt.plot(x,norm(loc=0,scale=1).pdf(x),'k--')
        plt.ylim(bottom=0)
        plt.xlabel('log(strengths)')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file+'logInStren.png')
        plt.close()

        fig = plt.figure()
        a = np.log(self.strengths_out[self.idx_PosOut])
        b = np.log(abs(self.strengths_out[self.idx_NegOut]))
        a = (a-a.mean())/a.std()
        b = (b-b.mean())/b.std()
        minStren = np.percentile(np.concatenate([a,b]),0)
        maxStren = np.percentile(np.concatenate([a,b]),100)
        x = np.linspace(minStren,maxStren,200)
        density_a = gaussian_kde(a)
        density_b = gaussian_kde(b)
        plt.plot(x,density_a(x),'r',label='$\log(s_\mathrm{out})(>0)$')
        plt.plot(x,density_b(x),'b',label='$\log(|s_\mathrm{out}|)(<0)$')
        plt.plot(x,norm(loc=0,scale=1).pdf(x),'k--')
        plt.ylim(bottom=0)
        plt.xlabel('log(strengths)')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file+'logOutStren.png')
        plt.close()

    def plotFlucDist(self, file, nodes, title=None):
        # plot distribution of fluctuations around steady states
        print(' plotting fluctuation distribution (xi-Xi distribution) to %s ...'%file)

        fig = plt.figure()
        minFluc = np.percentile(self.statesFluc[nodes],2)
        maxFluc = np.percentile(self.statesFluc[nodes],98)
        x = np.linspace(minFluc,maxFluc,200)
        colors = list(map(tuple,np.random.rand(self.size,3)))
        for i in nodes:
            density = gaussian_kde(self.statesFluc[i])
            mean = self.statesFluc[i].mean()
            sd = self.statesFluc[i].std()
            plt.plot(x,density(x),label='node %d: $\mu=%.4f,\sigma=%.4f$'%(i,mean,sd),c=colors[i])
            plt.plot(x,norm(loc=mean,scale=sd).pdf(x),c=colors[i],ls='--')
        plt.ylim(bottom=0)
        if title: plt.title(title)
        plt.xlabel('fluctuation $x_i-X_i$')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def plotFlucSdDist(self, file, title=None):
        # plot distribution of s.d. of fluctuations around steady states
        print(' plotting fluctuation s.d. distribution (xi-Xi s.d. distribution) to %s ...'%file)

        fig = plt.figure()
        minFlucSd = np.percentile(self.flucSd,2)
        maxFlucSd = np.percentile(self.flucSd,98)
        x = np.linspace(minFlucSd,maxFlucSd,200)
        density = gaussian_kde(self.flucSd)
        mean = self.flucSd.mean()
        sd = self.flucSd.std()
        plt.plot(x,density(x),label='$\mu=%.4f,\sigma=%.4f$'%(mean,sd),c='k')
        # plt.plot(x,norm(loc=mean,scale=sd).pdf(x),ls='--')
        plt.ylim(bottom=0)
        if title: plt.title(title)
        plt.xlabel('fluctuation s.d.')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def plotFlucSdAgainstDegStren(self, degreeFile, strengthFile, degreeTitle=None, strengthTitle=None, strenLogScale=False,
        onlyIn=False, onlyOut=False):
        # plot s.d. of fluctuations against degrees & strengths
        print(' plotting s.d. of fluctuations against degrees & strengths to %s & %s ...'%(degreeFile,strengthFile))

        fig = plt.figure()
        if not onlyOut: plt.scatter(self.degrees_in,self.flucSd,label='in-degrees',s=1,c='b')
        if not onlyIn: plt.scatter(self.degrees_out,self.flucSd,label='out-degrees',s=1,c='r')
        if degreeTitle: plt.title(degreeTitle)
        plt.xlabel('degrees')
        plt.ylabel('s.d. of fluctuation')
        plt.legend()
        fig.tight_layout()
        fig.savefig(degreeFile)
        plt.close()

        fig = plt.figure()
        if strenLogScale:
            if not onlyOut: plt.scatter(np.log(abs(self.strengths_in)),self.flucSd,label='log in-strengths',s=1,c='b')
            if not onlyIn: plt.scatter(np.log(abs(self.strengths_out)),self.flucSd,label='log out-strengths',s=1,c='r')
            plt.xlabel('log strengths')
        else:
            if not onlyOut: plt.scatter(self.strengths_in,self.flucSd,label='in-strengths',s=1,c='b')
            if not onlyIn: plt.scatter(self.strengths_out,self.flucSd,label='out-strengths',s=1,c='r')
            plt.xlabel('strengths')
        if degreeTitle: plt.title(degreeTitle)
        plt.ylabel('s.d. of fluctuation')
        plt.legend()
        fig.tight_layout()
        fig.savefig(strengthFile)
        plt.close()

    # DEPRECATED
    def plotCrossSecDist(self, t, file, title=None, xlimRange=None, ylimRange=None):
        # plot cross-sectional distribution of fluctuations around steady states
        # i.e. distribution of all nodes at some time
        # t = time steps
        print(' plotting cross-sectional fluctuation distribution (xi-Xi distribution) to %s ...'%file)

        fig = plt.figure()
        if xlimRange: x = np.linspace(xlimRange[0],xlimRange[1],200)
        else:
            minFluc = np.percentile(self.statesFluc[:,t],2)
            maxFluc = np.percentile(self.statesFluc[:,t],98)
            x = np.linspace(minFluc,maxFluc,200)
        density = gaussian_kde(self.statesFluc[:,t])
        mean = self.statesFluc[:,t].mean()
        sd = self.statesFluc[:,t].std()
        plt.plot(x,density(x),label='time=%.4f: $\mu=%.4f,\sigma=%.4f$'%(self.timeStep*t,mean,sd),c='k')
        if ylimRange: plt.ylim(ylimRange)
        else: plt.ylim(bottom=0)
        if title: plt.title(title)
        plt.xlabel('fluctuation $x_i-X_i$')
        plt.legend(loc='lower center')
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def plotDynamics(self, file, title=None, nodes=None, iterSlice=None, color=None, ylimRange=None, withSteadyStates=False):
        # COMMONLY USED!
        # plot time series data to file
        print(' plotting dynamics to %s ...'%file)

        fig = plt.figure(figsize=(12,6))
        if not nodes: nodes = range(self.size)
        if not iterSlice: iterSlice = slice(self.iter)

        plt.xlim(self.time_[iterSlice][0],self.time_[iterSlice][-1])
        if ylimRange: plt.ylim(ylimRange)

        if color: colors = [color]*self.size
        else: colors = list(map(tuple,np.random.rand(self.size,3)))

        for i in nodes:
            plt.plot(self.time_[iterSlice],self.states_np[i,iterSlice],c=colors[i])
            if withSteadyStates: plt.axhline(y=self.steadyStates[i],c=colors[i],ls='--')

        if title: plt.title(title)
        plt.xlabel('time $t$')
        plt.ylabel('states $\\{x_j\\}_{1:%d}$'%self.size)
        plt.grid()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def printDynamics(self, file, iterSlice=None, lastSlice=False):
        # print time series data to file (csv format)
        print(' printing dynamics to %s ...'%file)

        if not iterSlice: iterSlice = slice(self.iter)
        if lastSlice: data = np.concatenate(([self.time_[-1]],self.states_np[:,-1])).reshape((1,self.size+1))
        else: data = np.vstack((self.time_[iterSlice],self.states_np[:,iterSlice])).T

        head = 't,'+','.join(map(str,range(self.size)))
        np.savetxt(file,data,delimiter=',',fmt='%.4f',header=head,comments='')

    def printContFile(self, file):
        # print time series data to cont file (csv format)
        # NO NEED to prepend '(cont)' to file
        self.printDynamics('(cont)'+file,iterSlice=slice(-2,None))

    def printContFile_FHN(self, file, fileY):
        # FHN DYNAMICS
        # print time series data to cont file (csv format)
        # NO NEED to prepend '(cont)' to file
        self.printDynamics('(cont)'+file,iterSlice=slice(-2,None))
        print(' printing dynamics to %s ...'%('(cont)'+fileY))
        np.save('(cont)'+fileY,self.statesY) # fileY is npy file

    def plotDynamicsWithPeaks(self, node, file, title=None, iterSlice=None, ylimRange=None, withPeakHeight=False):
        # plot time series data with peaks to file
        print(' plotting dynamics with peaks to %s ...'%file)
        if not iterSlice: iterSlice = slice(self.iter)
        t = np.array(self.time_[iterSlice])
        x = self.states_np[node,iterSlice]

        fig = plt.figure(figsize=(12,6))
        plt.xlim(t[0],t[-1])
        if ylimRange: plt.ylim(ylimRange)
        plt.plot(t,x,c='k')
        plt.plot(t[self.peakDict[node]],x[self.peakDict[node]],'rx')
        if withPeakHeight: plt.axhline(y=self.peakHeight,c='gray',ls='--')
        if title: plt.title(title)
        plt.xlabel('time $t$')
        plt.ylabel('states $x_j$')
        plt.grid()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    # DEPRECATED
    def plotPeakDistInTime(self, file, bins, iterSlice=None):
        # plot distribution of peaks to file
        print(' plotting distribution of peaks in time to %s ...'%file)
        if not iterSlice: iterSlice = slice(self.iter)
        t = np.array(self.time_[iterSlice])
        peakIdx = []
        for i in range(self.size): peakIdx.extend(self.peakDict[i])

        fig = plt.figure(figsize=(12,6))
        freq,edge = np.histogram(t[peakIdx],bins=bins)
        plt.scatter(edge[:-1],freq,s=1,c='k')
        plt.xlim(t[0],t[-1])
        plt.xlabel('time $t$')
        plt.ylabel('peak count')
        plt.grid()
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    # DEPRECATED
    def plotRaster(self, file, node, height, iterSlice=None, title=None):
        # plot raster plot to file
        print(' plotting raster plot to %s ...'%file)
        if not iterSlice: iterSlice = slice(self.iter)
        raster = self.states_np[node,iterSlice]>height

        fig = plt.figure(figsize=(12,6))
        plt.imshow(raster,cmap='Greys',aspect='auto')
        if title: plt.title(title)
        plt.xlabel('time $t$')
        plt.ylabel('node index')
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    def plotPeakCountDist(self, file, logProb=False, logPeakCount=False,
        histogram=False, align='mid', bins=10, standardize=False):
        # COMMONLY USED!
        # plot disrtibution of peak count to file
        print(' plotting disrtibution of peak count to %s ...'%file)
        if logPeakCount: peakCount = np.log(self.peakCount[self.peakCount>0])
        else: peakCount = self.peakCount
        mean = peakCount.mean()
        sd = peakCount.std()
        if standardize: peakCount = (peakCount-mean)/sd
        minPeak = np.min(peakCount)
        maxPeak = np.max(peakCount)

        fig = plt.figure()
        x = np.linspace(minPeak,maxPeak,200)
        if standardize: plt.plot(x,norm(loc=0,scale=1).pdf(x),'r--')
        else: plt.plot(x,norm(loc=mean,scale=sd).pdf(x),'r--')
        if histogram: plt.hist(peakCount,color='k',align=align,density=True,bins=bins)
        else:
            density = gaussian_kde(peakCount)
            plt.plot(x,np.log(density(x)) if logProb else density(x),'k')
        plt.xlim(np.max(minPeak,0),maxPeak)
        plt.xlabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file)
        plt.close()

    # DEPRECATED
    def plotPeakCountAgainstDegStren(self, file, logPeakCount=False):
        # plot peak count against strengths to file (node classifications by signs of strengths)
        # LATER: change to log-scale axis ticks
        print(' plotting peak count against strengths to %s ...'%(file+'(*)PeakCountAgainst*.png'))
        if logPeakCount: peakCount = np.log(self.peakCount)
        else: peakCount = self.peakCount
        #======================================================================#
        # excitatory nodes
        fig = plt.figure()
        idx = self.idx_PosOut
        plt.scatter(np.log(self.degrees_in[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('excitatory nodes')
        plt.xlabel('$\log(k_\mathrm{in})$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(PosOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstLogKin.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_PosOut
        plt.scatter(np.log(self.degrees_out[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('excitatory nodes')
        plt.xlabel('$\log(k_\mathrm{out})$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(PosOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstLogKout.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_PosOut
        plt.scatter(self.strengths_out[idx],peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('excitatory nodes')
        plt.xlabel('$s_\mathrm{out}$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(PosOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstSout.png')
        plt.close()
        #======================================================================#
        # inhibitory nodes
        fig = plt.figure()
        idx = self.idx_NegOut
        plt.scatter(np.log(self.degrees_in[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('inhibitory nodes')
        plt.xlabel('$\log(k_\mathrm{in})$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(NegOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstLogKin.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_NegOut
        plt.scatter(np.log(self.degrees_out[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('inhibitory nodes')
        plt.xlabel('$\log(k_\mathrm{out})$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(NegOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstLogKout.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_NegOut
        plt.scatter(abs(self.strengths_out[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('inhibitory nodes')
        plt.xlabel('$|s_\mathrm{out}|$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(NegOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstAbsSout.png')
        plt.close()
        #======================================================================#
        # x-axis: strengths_in
        fig = plt.figure()
        idx = self.idx_PosInPosOut
        plt.scatter(self.strengths_in[idx],peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('excitatory nodes ($s_\mathrm{in}>0$)')
        plt.xlabel('$s_\mathrm{in}$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(PosInPosOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstSin.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_PosInNegOut
        plt.scatter(self.strengths_in[idx],peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('inhibitory nodes ($s_\mathrm{in}>0$)')
        plt.xlabel('$s_\mathrm{in}$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(PosInNegOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstSin.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_NegInPosOut
        plt.scatter(abs(self.strengths_in[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('excitatory nodes ($s_\mathrm{in}<0$)')
        plt.xlabel('$|s_\mathrm{in}|$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(NegInPosOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstAbsSin.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_NegInNegOut
        plt.scatter(abs(self.strengths_in[idx]),peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('inhibitory nodes ($s_\mathrm{in}<0$)')
        plt.xlabel('$|s_\mathrm{in}|$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(NegInNegOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstAbsSin.png')
        plt.close()
        #======================================================================#
        # x-axis: strengths_posIn
        fig = plt.figure()
        idx = self.idx_PosOut
        plt.scatter(self.strengths_posIn[idx],peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('excitatory nodes')
        plt.xlabel('$s^+_\mathrm{in}$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(PosOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstSposin.png')
        plt.close()

        fig = plt.figure()
        idx = self.idx_NegOut
        plt.scatter(self.strengths_posIn[idx],peakCount[idx],s=1,c='k')
        plt.xlim(left=0)
        plt.title('inhibitory nodes')
        plt.xlabel('$s^+_\mathrm{in}$')
        plt.ylabel('log(peak count)' if logPeakCount else 'peak count')
        fig.tight_layout()
        fig.savefig(file+'(NegOut)'+('Log' if logPeakCount else '')+'PeakCountAgainstSposin.png')
        plt.close()

    def plotAutocorr(self, nodes, Nlags, autocorrFile, logAutocorrFile, title=None):
        # plot autocorrelations to file
        print(' plotting autocorrelations & log autocorrelations to %s & %s ...'%(autocorrFile,logAutocorrFile))
        myAutocorrs = []
        lags = list(range(1,Nlags+1))
        colors = list(map(tuple,np.random.rand(self.size,3)))

        fig = plt.figure()
        for i in nodes:
            myAutocorr = [autocorr(np.diff(self.states_np[i]),t) for t in lags]
            myAutocorrs.append(myAutocorr)
            plt.scatter(lags,myAutocorr,s=1,color=colors[i])
        meanAutocorr = np.mean(myAutocorrs,axis=0)
        plt.plot(lags,meanAutocorr,'k--')
        # 99% likelihood bound
        plt.axhline(y=2.33/np.sqrt(self.iter),c='r',ls='--')
        plt.axhline(y=-2.33/np.sqrt(self.iter),c='r',ls='--')
        plt.xlim(0,Nlags)
        if title: plt.title(title)
        plt.xlabel('time lag')
        plt.ylabel('autocorrelation')
        fig.tight_layout()
        fig.savefig(autocorrFile)
        plt.close()

        fig = plt.figure()
        plt.plot(lags,np.log(abs(meanAutocorr)),'k')
        plt.xlim(0,Nlags)
        if title: plt.title(title)
        plt.xlabel('time lag')
        plt.ylabel('log(autocorrelation)')
        fig.tight_layout()
        fig.savefig(logAutocorrFile)
        plt.close()

def autocorr(x,t=1):
    # autocorrelation
    return np.corrcoef(x[:-t],x[t:])[0,1]

# DEPRECATED
def plotQQ(x,y,file,xlab,ylab,title=None):
    # QQ plot of two sets of data
    print(' plotting QQ plot to %s ...'%file)
    percentileX = [(i+1)/(len(y)+1) for i in range(len(y))]

    # distribution params modifiable
    if x=='norm': sortX = norm.ppf(percentileX)
    elif x=='expon': sortX = expon.ppf(percentileX)
    elif x=='lognorm': sortX = lognorm.ppf(percentileX,s=1)
    elif x=='pareto': sortX = pareto.ppf(percentileX,b=0.75)
    else: sortX = np.sort(x)
    sortY = np.sort(y)

    fig = plt.figure()
    plt.scatter(sortX,sortY,s=1,c='k')
    if x=='norm':
        mean = np.mean(y)
        sd = np.std(y)
        normY = mean+sd*sortX
        # plt.plot(sortX,normY,'r--')
    if title: plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    fig.tight_layout()
    fig.savefig(file)
    plt.close()

def plotQuantileFraction(x,file,title=None,lab=None):
    # fraction of sum of values accounted for by quantiles (dominance plot)
    print(' plotting quantile-fraction to %s ...'%file)
    fig = plt.figure()
    for idx,y in enumerate(x):
        n = len(y)
        S = sum(y)
        y[::-1].sort()
        f = [i/n for i in range(n+1)]
        s = [y[0]/S]
        for i in range(1,n): s.append(s[i-1]+y[i]/S)
        # s = [sum(y[:i])/S for i in range(n+1)]
        if not lab: plot(f,s)
        else: plt.plot(f,s,label=lab[idx])
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend()
    if title: plt.title(title)
    fig.tight_layout()
    fig.savefig(file)
    plt.close()
