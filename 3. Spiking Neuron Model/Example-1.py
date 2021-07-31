'''
Example run of SpikingNeuronModel
@ Frankie Yeung (2021 Mar)
'''
import numpy as np
import SpikingNeuronModel as m
np.random.seed(0)
totIter = 40
plotStep = 20
myModel = m.SpikingNeuronModel()
myModel.initNetwork('DIV25_PREmethod.npy')
myModel.initDynamicalParams()
myModel.initDynamics(totIter=totIter,plotStep=plotStep)
myModel.runDynamics()
myModel.saveDynamicsAndPlot()
