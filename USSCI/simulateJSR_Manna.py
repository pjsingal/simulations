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
    'Gutierrez-2025': {
        'submodels': {
            'base': r"chemical_mechanisms/Gutierrez-2025/gutierrez-2025.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/gutierrez-2025_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/gutierrez-2025_LMRR_allPLOG.yaml",
                    },
    },
    'Arunthanayothin-2021': {
        'submodels': {
            'base': r'chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml',
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR_allPLOG.yaml",
                    },
    },
    'Song-2019': {
        'submodels': {
            'base': r"chemical_mechanisms/Song-2019/song-2019.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR_allPLOG.yaml",
                    },
    },
    'Mei-2019': {
        'submodels': {
            'base': r"chemical_mechanisms/Mei-2019/mei-2019.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR_allPLOG.yaml",
                    },
    },
}

P=1.16
fuel={'CH4':0.9,'NH3':0.1}
oxidizer='O2'
diluent='N2'
phi=0.8
fraction={'diluent':0.9}
T_list = np.linspace(900,1180,gridsz)
data=['XCH4_90CH4_10NH3.csv','XNO_90CH4_10NH3.csv']
tau=0.4
t_max=50
V=113e-6
reactorTemperature = 900
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r",'orange']*3

Xspecies=['CH4','NO']

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

def getTemperatureDependence(gas,T):
    gas.TP = T, P*ct.one_atm
    gas.set_equivalence_ratio(phi,fuel,oxidizer,diluent=diluent,fraction=fraction)
    stirredReactor = getStirredReactor(gas)
    reactorNetwork = ct.ReactorNet([stirredReactor])
    states = ct.SolutionArray(stirredReactor.thermo)
    t = 0
    while t < t_max:
        t = reactorNetwork.step()
    states.append(stirredReactor.thermo.state)
    Xvec=[]
    for species in Xspecies:
        Xvec.append(states(species).X.flatten()[0])
    return Xvec

def simulateModel(model):
    print(f'\nModel: {model}')
    X_histories=[]
    for k,m in enumerate(models[model]['submodels']):
        print(f'Submodel: {m}')
        gas = ct.Solution(models[model]['submodels'][m])
        X_history=[]
        for i,T in enumerate(T_list):
            X_history.append(getTemperatureDependence(gas,T))
        X_histories.append(X_history)
    return X_histories




allXhistories=Parallel(n_jobs=len(models))(delayed(simulateModel)(model) for model in models)

f, ax = plt.subplots(1,len(Xspecies), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
tic = time.time()
for j,model in enumerate(models):
    for k,m in enumerate(models[model]['submodels']):
        for z, species in enumerate(Xspecies):
            Xi_history = [item[z] for item in allXhistories[j][k]]
            if k==0:
                label=f'{model}'
            else:
                label=None
            if species=='NO':
                ax[z].plot(T_list,np.array(Xi_history)*1e6, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
                ax[z].set_ylabel(f'X-{species} [ppm]')
            else:
                ax[z].plot(T_list,np.array(Xi_history)*100, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
                ax[z].set_ylabel(f'X-{species} [%]')
            ax[z].set_xlim([900,1250])
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_xlabel('Temperature [K]')
dat = pd.read_csv(f'USSCI/graph-reading/Manna-2024/{data[z]}',header=None)
ax[0].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Manna et al.')
ax[1].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Manna et al.')
plt.suptitle(r'Jet-stirred reactor: (90% CH4/10% NH3)/O2/N2, (1.16atm, phi=0.8)',fontsize=10)
ax[len(Xspecies)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1)  
path=f'USSCI/figures/'+args.date+'/Manna-2024'
os.makedirs(path,exist_ok=True)
name=f'Fig1.png'
plt.savefig(f'{path}/{name}', dpi=500, bbox_inches='tight')
toc = time.time()
print(f'Simulation completed in {toc-tic}s and stored at {path}/{name}\n')