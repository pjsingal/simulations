from __future__ import division
from __future__ import print_function
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from joblib import Parallel, delayed
import matplotlib as mpl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--figwidth', type=float, help="figwidth = ")
parser.add_argument('--figheight', type=float, help="figheight = ")
parser.add_argument('--fsz', type=float, help="mpl.rcParams['font.size'] = ", default=8)
parser.add_argument('--fszxtick', type=float, help="mpl.rcParams['xtick.labelsize'] = ", default=7)
parser.add_argument('--fszytick', type=float, help="mpl.rcParams['ytick.labelsize'] = ", default=7)
parser.add_argument('--fszaxlab', type=float, help="mpl.rcParams['axes.labelsize'] = ", default=8)
parser.add_argument('--lw', type=float, help="lw = ", default=0.7)
parser.add_argument('--mw', type=float, help="mw = ", default=0.5)
parser.add_argument('--msz', type=float, help="msz = ", default=2.5)
parser.add_argument('--lgdw', type=float, help="lgdw = ", default=0.6)
parser.add_argument('--lgdfsz', type=float, help="lgdw = ", default=5)
parser.add_argument('--gridsz', type=int, help="gridsz = ", default=10)
parser.add_argument('--dpi', type=int, help="dpi = ", default=1000)
parser.add_argument('--date', type=str, help="sim date = ")
args = parser.parse_args()
lw=args.lw
mw=args.mw
msz=args.msz
dpi=args.dpi
lgdw=args.lgdw
lgdfsz=args.lgdfsz
gridsz=args.gridsz
from matplotlib.legend_handler import HandlerTuple
mpl.rc('font',family='Times New Roman')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = args.fsz
mpl.rcParams['xtick.labelsize'] = args.fszxtick
mpl.rcParams['ytick.labelsize'] = args.fszytick
plt.rcParams['axes.labelsize'] = args.fszaxlab
mpl.rcParams['xtick.major.width'] = 0.5  # Width of major ticks on x-axis
mpl.rcParams['ytick.major.width'] = 0.5  # Width of major ticks on y-axis
mpl.rcParams['xtick.minor.width'] = 0.5  # Width of minor ticks on x-axis
mpl.rcParams['ytick.minor.width'] = 0.5  # Width of minor ticks on y-axis
mpl.rcParams['xtick.major.size'] = 2.5  # Length of major ticks on x-axis
mpl.rcParams['ytick.major.size'] = 2.5  # Length of major ticks on y-axis
mpl.rcParams['xtick.minor.size'] = 1.5  # Length of minor ticks on x-axis
mpl.rcParams['ytick.minor.size'] = 1.5  # Length of minor ticks on y-axis

########################################################################################
models = {
    'Cornell-2024': {
        'submodels': {
            'base': r"chemical_mechanisms/Cornell-2024/cornell-2024.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/cornell-2024_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/cornell-2024_LMRR_allPLOG.yaml",
                    },
    },
}

P=1
fuel={'C2H6':0.9,'CHF3':0.1}
oxidizer={'O2':0.21,'N2':0.79}
phi=2
T_list = np.linspace(850,950,gridsz)
data=['XCH4_90CH4_10NH3.csv','XNO_90CH4_10NH3.csv']
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*3

Xspecies=['CF3-CF3','CF2O']

########################################################################################

# length = 200e-3  # *approximate* PFR length [m]
length = 20e-2  # *approximate* PFR length [m]
diameter=0.0087 # PFR diameter [m]
n_steps = 2000
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
Q_tn = 1000 #nominal gas flow rate @ STP [mL/min]

def getXvsT(T,model,m):
    gas = ct.Solution(models[model]['submodels'][m])
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = T, P*ct.one_atm
    area = np.pi*(diameter/2)**2
    u0 = Q_tn*1e-6/60/area
    n_steps=2000
    mass_flow_rate = u0 * gas.density * area
    flowReactor = ct.IdealGasConstPressureReactor(gas)
    reactorNetwork = ct.ReactorNet([flowReactor])
    # tau=192.097*P*ct.one_atm/1e5*1000/Q_tn/T #residence time formula from Gutierrez-2025
    tau=length/u0
    dt = tau/n_steps
    t1 = (np.arange(n_steps) + 1) * dt
    states = ct.SolutionArray(flowReactor.thermo)
    for n, t_i in enumerate(t1):
        reactorNetwork.advance(t_i)
    # print(f'{t1[-1]}={tau}')
    states.append(flowReactor.thermo.state)
    Xvec=[]
    for species in Xspecies:
        Xvec.append(states(species).X.flatten()[0])
    # print(f'{Xspecies[0]}:{Xvec[0]:.8e}, {Xspecies[1]}:{Xvec[1]:.8e}')
    return Xvec

f, ax = plt.subplots(1,len(Xspecies), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
tic = time.time()
for j,model in enumerate(models):
    print(f'\nModel: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f'Submodel: {m}')
        X_history=[]
        for T in T_list:
            X_history.append(getXvsT(T,model,m))
        for z, species in enumerate(Xspecies):
            Xi_history = [item[z]*1e6 for item in X_history]
            if k==0:
                label=f'{model}'
            else:
                label=None
            ax[z].plot(T_list,np.array(Xi_history)*1e6, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[z].set_ylabel(f'X-{species} [ppm]')
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_xlabel('Temperature [K]')
plt.suptitle(r'Plug-flow reactor: (90% C2H6/10% CHF3)/air, (1atm, phi=1.0)',fontsize=10)
ax[len(Xspecies)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1)  
path=f'USSCI/figures/'+args.date+'/Cornell-2024'
os.makedirs(path,exist_ok=True)
name=f'Fig1_2.png'
plt.savefig(f'{path}/{name}', dpi=500, bbox_inches='tight')
toc = time.time()
print(f'Simulation completed in {toc-tic}s and stored at {path}/{name}\n')