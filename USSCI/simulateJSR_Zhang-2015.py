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
    'Zhang-2015': {
        'submodels': {
            'base': r"chemical_mechanisms/Zhang-2015/zhang-2015_nhexane.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2015_nhexane_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2015_nhexane_LMRR_allPLOG.yaml",
                    },
    },
}

X={'NC6H14':0.00095,'NC6H14':0.00005,'O2':0.019,'N2':0.98} #n-hexane mixed with 3-methylpentane (fio what correct name of the latter species is in mech)
P=10
T_list = np.linspace(500,1100,gridsz)
data=['XCH4_75N2_25H2O.csv','XCO2_75N2_25H2O.csv','XCO_75N2_25H2O.csv']
tau=0.7 # might actually be 0.07s
diameter=0.04 #m
V = 4/3*np.pi*(diameter/2)**2 #JSR volume
t_max=20
reactorTemperature = 6000
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*3

Xspecies=['CH4','CO2','CO']

########################################################################################

def getStirredReactor(gas):
    h = 79.5 # heat transfer coefficient W/m2/K
    K = 2e-5 # pressureValveCoefficient
    reactorRadius = (V*3/4/np.pi)**(1/3) # [m3]
    reactorSurfaceArea =4*np.pi*reactorRadius**2 # [m3]
    fuelAirMixtureTank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)
    env = ct.Reservoir(gas)
    reactor = ct.IdealGasReactor(gas, energy='on', volume=V)
    ct.MassFlowController(upstream=fuelAirMixtureTank,
                          downstream=reactor,
                          mdot=reactor.mass/tau)
    ct.Valve(upstream=reactor,
             downstream=exhaust,
             K=K)
    ct.Wall(reactor, env, A=reactorSurfaceArea, U=h)
    return reactor

def getTemperatureDependence(model,m):
    print(f'Sub-model: {m}')
    gas = ct.Solution(models[model]['submodels'][m])
    stirredReactor = getStirredReactor(gas)
    columnNames = (
        ['pressure'] +
        [stirredReactor.component_name(item)
         for item in range(stirredReactor.n_vars)]
    )
    tempDependence = pd.DataFrame(columns=columnNames)
    concentrations = X
    for T in T_list:
        gas.TPX = T, P*ct.one_atm, concentrations
        stirredReactor = getStirredReactor(gas)
        reactorNetwork = ct.ReactorNet([stirredReactor])
        t = 0
        while t < t_max:
            t = reactorNetwork.step()
        state = np.hstack([stirredReactor.thermo.P,
                        stirredReactor.mass,
                        stirredReactor.volume,
                        stirredReactor.T,
                        stirredReactor.thermo.X])
        concentrations = stirredReactor.thermo.X
        tempDependence.loc[T] = state
    return tempDependence

f, ax = plt.subplots(1,len(Xspecies), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
tic = time.time()
for j,model in enumerate(models):
    print(f'\nModel: {model}')
    tempDependences = Parallel(n_jobs=3)(
        delayed(getTemperatureDependence)(model,m)
        for m in models[model]['submodels']
        )
    for z, species in enumerate(Xspecies):
        for k,m in enumerate(models[model]['submodels']):
            tempDependence=tempDependences[k]
            print(f'Submodel: {m}')
            if k==0:
                label=f'{model}'
            else:
                label=None
            ax[z].plot(tempDependence.index,tempDependence[species]*100, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[z].set_ylabel(f'X-{species} [%]')
        # if j==len(list(models.keys()))-1:
        #     dat = pd.read_csv(f'USSCI/graph-reading/Lavadera-2018/{data[z]}',header=None)
        #     ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Lavadera et al.')
        ax[z].set_xlim([700,1200])
        ax[z].tick_params(axis='both',direction='in')
        ax[z].set_xlabel('Temperature [K]')
plt.suptitle(r'Jet-stirred reactor: 2.31% C3H8/7.69% O2/67.5% N2/22.5% H2O (1.1atm)',fontsize=10)
ax[len(Xspecies)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1)  
path=f'USSCI/figures/'+args.date+'/Zhang-2015'
os.makedirs(path,exist_ok=True)
name=f'Fig8.png'
plt.savefig(f'{path}/{name}', dpi=500, bbox_inches='tight')
toc = time.time()
print(f'Simulation completed in {toc-tic}s and stored at {path}/{name}\n')