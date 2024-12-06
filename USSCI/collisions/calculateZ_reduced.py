# Calculating Lennard-Jones collision frequencies (Z_LJ)
import numpy as np
import math
import yaml

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



def getC6(inputs):
    epsMM = inputs['eps']*kb
    epsAA = inputs['colliders'][0]['eps']*kb
    sigmaMM = inputs['sigma']
    sigmaAA = inputs['colliders'][0]['sigma']
    epsMA = (epsMM*epsAA)**(1/2)
    sigmaMA = (sigmaMM+sigmaAA)/2
    return 4*epsMA*sigmaMA**6

def getM(inputs,T):
    mA = inputs['colliders'][0]['m']
    muM = inputs['mu']*3.335e-30 # C m
    muA = inputs['colliders'][0]['mu']*3.335e-30 # C m
    BA = inputs['Btot']/100*h*c #J
    BM = inputs['colliders'][0]['Btot']/100*h*c #J
    num = mA*(muM*muA)**(2/3)*(BA+BM)/2
    den = hbar*(kb*T)**(2/3)
    return num/den

def getTheta(inputs,T):
    C6 = getC6(inputs)
    muM = inputs['mu']*3.335e-30 # C m
    muA = inputs['colliders'][0]['mu']*3.335e-30 # C m
    return C6*kb*T/(muA*muM)**2

def loadYAML(fName):
    with open(fName) as f:
        return yaml.safe_load(f)

c = 3e8 #m/s
kb = 1.380649e-23 # J⋅K−1 boltzmann constant
h = 6.62607015e10-34 # Planck constant, m2 kg / s
hbar = h/2/np.pi # J*s

data = loadYAML("USSCI/collisions/dipole-dipole-collision-frequencies_reduced.yaml")

M = getM(data['reactions'][0],200)
theta = getTheta(data['reactions'][0],200)
print(M)
print(theta)
print(np.log(theta))