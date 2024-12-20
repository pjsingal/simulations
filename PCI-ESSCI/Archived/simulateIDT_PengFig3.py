import sys, os
sys.path.append(os.getcwd()+"cantera/build/python")
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy
import scipy.optimize
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import gridspec
#PENG FIGURE 3
import sys, os
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd 
import time
import numpy as np
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.legend_handler import HandlerTuple
# plt.rcParams['axes.labelsize'] = 18

save_plots = True
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
name = 'IDT_NH3_fig3'

models = {
        #   'mevel':'D:/Research/Models/Mevel/mevel.cti',
          'LMR-R':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR.yaml",            
          'Ar':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allAR.yaml",
          r'H$_2$O':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allH2O.yaml",
          }

colors = ['xkcd:purple','r','b']

def ignitionDelay(states, species):
    # i_ign = states(species).Y.argmax()
    i_ign = np.gradient(states(species).Y.T[0]).argmax()
    return states.t[i_ign]

P = 11
T_list = np.linspace(1000/0.45,1000/0.7,10)

for i, m in enumerate(list(models.keys())):

    estimatedIgnitionDelayTimes = np.ones(len(T_list))
    estimatedIgnitionDelayTimes[:] = 5

    ignitionDelays_RG = np.zeros(len(T_list))
    for j, T in enumerate(T_list):
        gas = ct.Solution(list(models.values())[i])
        gas.TPX = T, P*ct.one_atm, {'NH3':0.0114, 'O2':0.0086, 'Ar':0.9800}
        r = ct.Reactor(contents=gas)
        reactorNetwork = ct.ReactorNet([r])
        timeHistory = ct.SolutionArray(gas, extra=['t'])
        t0 = time.time()
        t = 0
        counter = 1
        while t < estimatedIgnitionDelayTimes[j]:
            t = reactorNetwork.step()
            if counter % 1 == 0:
                timeHistory.append(r.thermo.state, t=t)
            counter += 1
        tau = ignitionDelay(timeHistory, 'oh')
        t1 = time.time()

        # print("Computed Real Gas Ignition Delay: {:.3e} seconds for T={}K. "
        #       "Took {:3.2f}s to compute".format(tau, temperature, t1-t0))
        ignitionDelays_RG[j] = tau
        

    ax.semilogy(1000/T_list, 1e6*ignitionDelays_RG, '-', linestyle='solid', color=colors[i], label=m)


path="graph_reading"
dataset0 = pd.read_csv(path+'/Peng Fig 3/Fig3expData.csv')
ax.plot(dataset0.iloc[:, 0],dataset0.iloc[:, 1],marker='s',color='k', markersize=6, zorder=2, fillstyle='full', linestyle = 'None', label="Peng et al.")
ax.legend(fontsize=15, frameon=False)#, loc='upper right')  
ax.set_ylabel(r'Ignition delay [$\mathdefault{\mu s}$]', fontsize=18)
ax.set_xlabel(r'1000/T [K$^\mathdefault{-1}$]', fontsize=18)
ax.tick_params(axis='both', direction="in", labelsize=15)
ax.tick_params(axis='both', which='minor', direction="in")#, bottom=False)

if save_plots == True:
    plt.savefig(name+'.pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(name+'.png', dpi=1000, bbox_inches='tight')
plt.show()        