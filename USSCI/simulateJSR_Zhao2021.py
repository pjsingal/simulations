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
import csv
import warnings
import copy

warnings.filterwarnings("ignore", message="NasaPoly2::validate")
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
title='JSR: 0.2% n-butane/13% O2/66.8% N2/20% CO2 (100atm)'
folder='Zhao-2021'
name='Fig3'
exp=False
dataLabel='Zhao et al. (2021)'
# data=['XCH4_75N2_25H2O.csv','XCO2_75N2_25H2O.csv','XCO_75N2_25H2O.csv']
# observables=['C4H10','CO','CO2'] #scroll down to body of code

# X={'C4H10':0.002,'O2':0.13,'N2':0.668,'CO2':0.2} # scroll down to body of code
P=100
T_list = np.linspace(500,1000,gridsz)
Xlim=[500,1000]
tau=0.15 #it actually ranges from 0.234 to 0.13 seconds
V=0.5e-6 #m3
t_max=40

models = {
    # 'Arunthanayothin-2021': { #use NC4H10
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Song-2019': { #use NC4H10
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Song-2019/song-2019.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Aramco-3.0': {
        'submodels': {
            'base': r"chemical_mechanisms/AramcoMech30/aramco30.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allPLOG.yaml",
                    },
    },
    # 'Zhang-2018': { #produce bad results
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2018/zhang-2018_ethanolDME.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2018_ethanolDME_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2018_ethanolDME_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Bugler-2016': { #sim takes forever so probably unstable
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Bugler-2016/bugler-2016.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Zhang-2016': { #produce bad results
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2016/zhang-2016_nheptane.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2016_nheptane_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2016_nheptane_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Zhang-2015': { #produce bad results
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2015/zhang-2015_nhexane.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2015_nhexane_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2015_nhexane_LMRR_allPLOG.yaml",
    #                 },
    # },
}
########################################################################################
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*3
# V = 4/3*np.pi*(diameter/2)**2 #JSR volume

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

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
    gas.TPX = T, P*ct.one_atm, X
    stirredReactor = getStirredReactor(gas)
    reactorNetwork = ct.ReactorNet([stirredReactor])
    states = ct.SolutionArray(stirredReactor.thermo)
    t = 0
    while t < t_max:
        t = reactorNetwork.step()
    states.append(stirredReactor.thermo.state)
    return [states(species).X.flatten()[0] for species in observables]

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    X_history=[getTemperatureDependence(gas,T) for T in T_list]
    for z, species in enumerate(observables):
        Xi_history = [item[z] for item in X_history]
        data = zip(T_list,Xi_history)
        simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/JSR/{m}/{species}'
        os.makedirs(simOutPath,exist_ok=True)
        save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
# observables_default=['C4H10','CO','CO2']
# observables_default=['CH4', 'C2H2','C2H4','C2H6']
# observables_default=['C3H8']
# observables_default=['CH4','C2H6','C3H8','C4H10']
observables_default=['C2H6','C3H8','C4H10']
X_default={'C4H10':0.002,'O2':0.13,'N2':0.668,'CO2':0.2} #case 3
f, ax = plt.subplots(1,len(observables_default), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    # if model=='Arunthanayothin-2021' or model=='Song-2019':
    #     X={'NC4H10':0.002,'O2':0.13,'N2':0.668,'CO2':0.2} #case 3
    #     # observables=['CH4','C2H6','C3H8','NC4H10']
    #     observables=['C2H6','C3H8','NC4H10']
    # else:
    X=X_default
    observables=observables_default
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        flag=False
        # flag=True
        while not flag:
            for z, species in enumerate(observables):
                simFile=f'USSCI/data/{args.date}/{folder}/{model}/JSR/{m}/{species}/{name}.csv'
                if not os.path.exists(simFile):
                    sims=generateData(model,m) 
                    flag=True
            flag=True
        for z, species in enumerate(observables):   
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/JSR/{m}/{species}/{name}.csv'
            sims=pd.read_csv(simFile)
            label = f'{model}' if k == 0 else None
            ax[z].plot(sims.iloc[:,0],sims.iloc[:,1]*1e6, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[z].set_ylabel(f'X-{species} [ppm]')
            if exp and j==len(models)-1 and k==2:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
                ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[z].set_xlim(Xlim)
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_xlabel('Temperature [K]')
        print('  > Data added to plot')
plt.suptitle(f'{title}',fontsize=10)
ax[len(observables_default)-1].legend(fontsize=lgdfsz,frameon=False,loc='lower left', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/JSR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
