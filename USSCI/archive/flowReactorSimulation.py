import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

model='h2o2.yaml'
fuel='H2'
oxidizer='O2:1.0, N2:3.76'
species='OH' #species to track the mole fraction of
phi=0.375
tau=50e-3 #residence time
T_range = np.linspace(1300,2000,30)
P=1.2 #atm
t_max=50 #max time for each simulation at each temperature

gas = ct.Solution(model)

Xvals = []
for T in T_range:
    gas.set_equivalence_ratio(phi,fuel,oxidizer,basis='mole')
    gas.TP = T, P
    flowReactor = ct.Reactor(gas, energy='on')
    # flowReactor = ct.IdealGasConstPressureReactor(gas)
    # flowReactor = ct.FlowReactor(gas, energy='on')
    reactorNetwork = ct.ReactorNet([flowReactor])
    states = ct.SolutionArray(flowReactor.thermo, extra=['t'])
    t=0
    while t<t_max:
        t=reactorNetwork.step()
        states.append(flowReactor.thermo.state,t=t)
    states.save('states.csv',basis='mole',overwrite=True)
    i_ign=np.gradient(states("h").X.flatten()).argmax()
    tList = states.t.flatten()
    IDT = tList[i_ign]
    #the idx of the species of interest when t equals the residence time
    threshold_idx = np.argmax(tList >= tau+IDT) 
    Xvals.append(states(species).X.flatten()[threshold_idx])

plt.figure()
plt.plot(T_range,Xvals)
plt.xlabel("Temperature")
plt.ylabel("NH3 mole fraction")
plt.show()