'''
Conductance-based Spiking Neuron Model based on EM Izhikevich's basic model
@ Frankie Yeung (2021 Mar)
'''
import os,re,glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True) # comment this out if no tex distribution is installed
plt.switch_backend('agg')

class SpikingNeuronModel:
    def __init__(self, plotFolderName='out-plot/', contFolderName='out-cont/', loadFromPrev=False):
        '''
        @ plotFolderName: folder to store time series plots
        @ contFolderName: folder to store (and load, if loadFromPrev==True) continuation files
        '''
        self.plotToFolder = False
        self.loadFromPrev = loadFromPrev
        if plotFolderName:
            self.plotFolderName = plotFolderName
            self.plotToFolder = True
            if not os.path.exists(plotFolderName): os.makedirs(plotFolderName)
        if contFolderName:
            self.contFolderName = contFolderName
            if not os.path.exists(contFolderName): os.makedirs(contFolderName)

    def initNetwork(self, couplingFile):
        '''
        @ couplingFile: a binary .npy file containing coupling strengths
        * load coupling strengths
        * the coupling strength matrix is an N*N square matrix consisting of entries gij
          where gij = directed coupling strength linking from node j to i
        '''
        self.Coupling = np.load(couplingFile)
        self.Adjacency = (self.Coupling!=0)
        self.outDegrees = np.sum(self.Adjacency,axis=0)
        with np.errstate(invalid='ignore'):
            self.outStrengths = np.nan_to_num(np.sum(self.Coupling,axis=0)/self.outDegrees)
        self.idxExcNodes = np.argwhere(self.outStrengths>=0).flatten() # indices for exc nodes
        self.idxInhNodes = np.argwhere(self.outStrengths<0).flatten() # indices for inh nodes
        # excitatory & inhibitory nodes
        self.N_exc = len(self.idxExcNodes)
        self.N_inh = len(self.idxInhNodes)
        self.N = self.Coupling.shape[0] # network size
        self.idxPosCoupling = {i:np.argwhere(self.Coupling[i]>0).flatten() for i in range(self.N)} # exc indices
        self.idxNegCoupling = {i:np.argwhere(self.Coupling[i]<0).flatten() for i in range(self.N)} # inh indices

    def initDynamicalParams(self, dynamicalParamDict={
        'voltageThres_exc': 0,
        'voltageThres_inh': -80,
        'tau_exc': 5,
        'tau_inh': 6,
        'beta': 1
    }):
        # model params
        self.voltageThres_exc = dynamicalParamDict['voltageThres_exc']
        self.voltageThres_inh = dynamicalParamDict['voltageThres_inh']
        self.tau_exc = dynamicalParamDict['tau_exc']
        self.tau_inh = dynamicalParamDict['tau_inh']
        self.beta = dynamicalParamDict['beta']
        # a,b,c,d,Coupling
        if self.loadFromPrev:
            # load continuation files from self.contFolderName
            self.a = np.load(self.contFolderName+'(cont)a.npy')
            self.b = np.load(self.contFolderName+'(cont)b.npy')
            self.c = np.load(self.contFolderName+'(cont)c.npy')
            self.d = np.load(self.contFolderName+'(cont)d.npy')
        else:
            # following params in EM Izhikevich's paper
            # exc nodes obeying one set of a,b,c,d params and inh nodes obeying another set
            rand_exc = np.random.rand(self.N_exc); rand_inh = np.random.rand(self.N_inh)
            self.a = np.zeros(self.N); self.a[self.idxExcNodes],self.a[self.idxInhNodes] = 0.02*np.ones(self.N_exc),0.02+0.08*rand_inh
            self.b = np.zeros(self.N); self.b[self.idxExcNodes],self.b[self.idxInhNodes] = 0.2*np.ones(self.N_exc),0.25-0.05*rand_inh
            self.c = np.zeros(self.N); self.c[self.idxExcNodes],self.c[self.idxInhNodes] = -65+15*rand_exc**2,-65*np.ones(self.N_inh)
            self.d = np.zeros(self.N); self.d[self.idxExcNodes],self.d[self.idxInhNodes] = 8-6*rand_exc**2,2*np.ones(self.N_inh)

    def initDynamics(self, totIter, plotStep=500):
        '''
        @ totIter: total iteration steps to run simulation for
        @ plotStep: time steps between each plotting of intermediate time series files
        * initialization of dynamics
        '''
        self.prevIter = 0
        if self.loadFromPrev:
            stopTimeFile = glob.glob(self.contFolderName+'SimulationStoppedAt_*.txt') # a list of one file name
            self.prevIter = int(re.split('[_.]',stopTimeFile[0])[1])
            print(' previous simulation stopped at t=%d, loading previous files ...'%self.prevIter)
        self.totIter = totIter
        self.plotStep = plotStep
        self.plotTime = list(range(self.prevIter+self.plotStep,self.totIter,self.plotStep))

        #### voltage & spike ####
        '''
        @ self.voltage & self.spike: for *computational* purpose
        @ self.SpikeSeries & self.VoltageSeries: for *recording* purpose
        '''
        if self.loadFromPrev:
            self.voltage = np.load(self.contFolderName+'(cont)v_t=%d.npy'%self.prevIter)
            self.recover = np.load(self.contFolderName+'(cont)r_t=%d.npy'%self.prevIter)
        else:
            self.voltage = -65*np.ones(self.N)
            self.recover = self.b*self.voltage
        self.SpikeSeries = np.zeros((self.N,self.totIter-self.prevIter)) # spike
        self.VoltageSeries = np.zeros((self.N,self.totIter-self.prevIter)) # voltage

        #### spike history ####
        self.spikeTimeHistory = {i:[] for i in range(self.N)}
        if self.loadFromPrev:
            prevSpikeFiles = glob.glob(self.contFolderName+'out-spike-t=*.npy')
            prevSpikeFiles.sort(key=lambda x:int(x[-9:-4])) # x[-9:-4] is time tag
            for file in prevSpikeFiles:
                prevSpikeSeries = np.load(file)
                for i in range(self.N):
                    self.spikeTimeHistory[i] += np.argwhere(prevSpikeSeries[i]).flatten().tolist()
                    ## (optional) truncation of history for speeding up
                    # history = np.argwhere(prevSpikeSeries[i]).flatten()
                    # history = history[history>self.prevIter-20*max(self.tau_exc,self.tau_inh)]
                    # self.spikeTimeHistory[i] += history.tolist()

    def historicalDecayFactorSum(self, idx, t, tau):
        sumForEachIdx = np.array([np.sum(np.exp(-(t-np.array(self.spikeTimeHistory[i]))/tau)) for i in idx])
        return sumForEachIdx

    def runDynamics(self):
        ''' run the spiking neuron model '''
        pbar = tqdm(total=self.totIter)
        pbar.update(self.prevIter)
        for t in range(self.prevIter,self.totIter):
            #### noise ####
            current = np.zeros(self.N)
            current[self.idxExcNodes],current[self.idxInhNodes] = 5*np.random.rand(self.N_exc),2*np.random.rand(self.N_inh)

            #### reset spike ####
            idxSpikingNodes = (self.voltage>=30)
            self.SpikeSeries[:,t-self.prevIter] = idxSpikingNodes
            self.voltage[idxSpikingNodes] = self.c[idxSpikingNodes]
            self.recover[idxSpikingNodes] = self.recover[idxSpikingNodes]+self.d[idxSpikingNodes]
            for i in np.argwhere(idxSpikingNodes).flatten(): self.spikeTimeHistory[i].append(t)

            conductance_exc = self.beta*np.array([np.sum(self.Coupling[i,self.idxPosCoupling[i]]*\
                self.historicalDecayFactorSum(self.idxPosCoupling[i],t,self.tau_exc)) for i in range(self.N)])
            conductance_inh = self.beta*np.array([np.sum(np.abs(self.Coupling[i,self.idxNegCoupling[i]])*\
                self.historicalDecayFactorSum(self.idxNegCoupling[i],t,self.tau_inh)) for i in range(self.N)])
            current += conductance_exc*(self.voltageThres_exc-self.voltage)+conductance_inh*(self.voltageThres_inh-self.voltage)

            self.voltage += 0.5*(.04*self.voltage**2+5*self.voltage+140-self.recover+current)
            self.voltage += 0.5*(.04*self.voltage**2+5*self.voltage+140-self.recover+current)
            self.recover += self.a*(self.b*self.voltage-self.recover)

            idxSpikingNodes = (self.voltage>=30)
            self.voltage[idxSpikingNodes] = 30 # equalize all spikes to 30eV
            self.VoltageSeries[:,t-self.prevIter] = self.voltage

            if t in self.plotTime: self.plotRaster(t=t)
            pbar.update()
        pbar.close()

    # ======================================================================== #
    # plotting & saving functions

    def plotRaster(self, t=None, sortBySpikeCounts=False):
        ''' raster plot up to time t '''
        if not t: t = self.totIter
        print(' plotting raster plot at t=%d ...'%t)
        fig = plt.figure(figsize=(12,6))
        if sortBySpikeCounts: idxSet = np.argsort(np.sum(self.SpikeSeries,axis=1))
        else: idxSet = list(range(self.N))
        for i,idx in enumerate(idxSet):
            spikeTime = self.prevIter+np.argwhere(self.SpikeSeries[idx,:t-self.prevIter]==1).flatten()
            plt.scatter(spikeTime,[i]*len(spikeTime),s=.1,c='k')
        if t<self.totIter: plt.axvline(x=t,c='k',ls='--')
        plt.xlim(self.prevIter,self.totIter)
        plt.ylim(0,self.N)
        plt.xlabel('time $t$')
        plt.ylabel('node index'+(' (sorted)' if sortBySpikeCounts else ''))
        fig.tight_layout()
        fig.savefig((self.plotFolderName if self.plotToFolder else '')+'SpikingModel_raster'+\
            ('Sorted' if sortBySpikeCounts else '')+'_t=%05d.png'%t)
        plt.close()

    def plotTimeSeries(self, Nnode=20, t=None, nodes=None):
        ''' time series plot up to time t '''
        if not t: t = self.totIter
        print(' plotting time series at t=%d ...'%t)
        if not nodes: nodes = list(range(Nnode))
        for i in nodes:
            fig = plt.figure(figsize=(12,6))
            plt.plot(np.arange(self.prevIter,t),self.VoltageSeries[i,:t-self.prevIter],c='k')
            plt.axhline(y=self.c[i],c='k',ls='--',label='reset level $c$')
            plt.xlim(self.prevIter,t)
            plt.xlabel('time $t$')
            plt.ylabel('membrane potential $v$')
            plt.title('(node %d) $a=%.4f,b=%.4f,c=%.4f,d=%.4f$'%\
                (i,self.a[i],self.b[i],self.c[i],self.d[i]))
            plt.legend()
            fig.tight_layout()
            fig.savefig((self.plotFolderName if self.plotToFolder else '')+'SpikingModel_v%04d_t=%05d.png'%(i,t))
            plt.close()

    def saveContFiles(self):
        ''' save continuation files (a,b,c,d,voltage,recover) '''
        # a,b,c,d
        np.save(self.contFolderName+'(cont)a.npy',self.a)
        np.save(self.contFolderName+'(cont)b.npy',self.b)
        np.save(self.contFolderName+'(cont)c.npy',self.c)
        np.save(self.contFolderName+'(cont)d.npy',self.d)
        # states
        np.save(self.contFolderName+'(cont)v_t=%d.npy'%self.totIter,self.voltage)
        np.save(self.contFolderName+'(cont)r_t=%d.npy'%self.totIter,self.recover)
        if self.loadFromPrev: os.remove(self.contFolderName+'SimulationStoppedAt_%d.txt'%self.prevIter)
        open(self.contFolderName+'SimulationStoppedAt_%d.txt'%self.totIter,'a').close()

    def saveTimeSeries(self):
        ''' save time series files (SpikeSeries,VoltageSeries) '''
        # time series
        np.save(self.contFolderName+'out-spike-t=%05dto%05d.npy'%(self.prevIter,self.totIter),self.SpikeSeries)
        np.save(self.contFolderName+'out-voltage-t=%05dto%05d.npy'%(self.prevIter,self.totIter),self.VoltageSeries)

    def saveDynamicsAndPlot(self, NnodeToPlot=20):
        ''' container for saving & plotting '''
        self.saveContFiles()
        self.saveTimeSeries()
        self.plotRaster()
        self.plotTimeSeries(Nnode=NnodeToPlot)

    # ======================================================================== #
    # below are helper functions

    def plotBurstPlot(self, countPeriod):
        ''' burst plot for identifying simultaneous spike activity '''
        print(' plotting burst plot with count period %d ...'%countPeriod)
        spikeCounts = np.sum(self.SpikeSeries,axis=0)
        spikeCountsCumSum = np.cumsum(spikeCounts)
        spikeCountsRolling = spikeCountsCumSum-np.concatenate([[0]*countPeriod,spikeCountsCumSum[:-countPeriod]])
        spikeCountsRolling /= self.N
        fig = plt.figure(figsize=(12,6))
        plt.plot(spikeCountsRolling,c='k')
        plt.xlim(0,self.totIter)
        plt.xlabel('time $t$')
        plt.ylabel('total burst over %d-step rolling window'%countPeriod)
        fig.tight_layout()
        fig.savefig((self.plotFolderName if self.plotToFolder else '')+\
            'spikingModel_burst_w=%d_t=%d.png'%(countPeriod,self.totIter))
        plt.close()
