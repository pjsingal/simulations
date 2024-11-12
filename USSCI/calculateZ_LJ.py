# Calculating Lennard-Jones collision frequencies (Z_LJ)
import numpy as np


def getZLJ(inputs, T):
    sigmaAM = (inputs['sigmaA']+inputs['sigmaM'])/2
    kb = 1.380649e-23 # J/K or m2kg/s2/K
    epsAM = np.sqrt(inputs['epsAA']*inputs['epsMM'])
    R = 8.314 # J/mol/K
    mAM = inputs['mA']*inputs['mM']/(inputs['mA']+inputs['mM'])
    omega = 0.636+0.567*np.log(kb*T/epsAM)
    Z = sigmaAM**2*omega*np.sqrt(8*R*T/np.pi/mAM)
    return Z