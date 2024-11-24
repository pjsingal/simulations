# Calculating Lennard-Jones collision frequencies (Z_LJ)
import numpy as np
import math

def getZLJ(inputs, T):
    sigmaAM = (inputs['sigmaA']+inputs['sigmaM'])/2
    kb = 1.380649e-23 # J/K or m2kg/s2/K
    epsAM = np.sqrt(inputs['epsAA']*inputs['epsMM'])
    R = 8.314 # J/mol/K
    mAM = inputs['mA']*inputs['mM']/(inputs['mA']+inputs['mM'])
    omega = 0.636+0.567*np.log(kb*T/epsAM)
    Z = sigmaAM**2*omega*np.sqrt(8*R*T/np.pi/mAM)
    return Z


def getZdd(inputs,T):
    f=1 #assumed value of rigidity factor
    muA = inputs['muA']
    muM = inputs['muM']
    kb = 1.380649e-23 # J/K or m2kg/s2/K
    mAM = inputs['mA']*inputs['mM']/(inputs['mA']+inputs['mM'])
    Z =f*math.gamma(1/3)*(muA*muM)**(2/3)*(kb*T)**(-1/6)*(8*np.pi/mAM)**(1/2)
    return Z