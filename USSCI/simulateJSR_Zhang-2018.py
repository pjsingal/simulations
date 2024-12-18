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
title=r'0.2% C$_2$H$_5$OH/2% O$_2$/N$_2$'+f'\n10 atm'
folder='Zhang-2018'
name='Fig10'
exp=False
dataLabel='Zhang et al. (2018)'
data=['ch4.csv', 'c2h2.csv','c2h4.csv','c2h6.csv'] #'c2h2' is just a placeholder, no data exists
# observables=['O2','CO','CO2','C2H4','CH4']
# observables=['C2H5OH','CH3OH', 'C2H6','C2H2','CH3OCH3']
observables=['CH4', 'C2H2','C2H4','C2H6']

X={'CH3CH2OH':0.002,'O2':0.02,'N2':1-0.002-0.02} #ethanol mixed with o2 and n2 bath
# X={'C2H5OH':0.002,'O2':0.02,'N2':1-0.002-0.02} #ethanol mixed with o2 and n2 bath
P=10
T=950
Xlim=[0,0.45]
tau=0.07
diameter=0.04 #m
t_max=50

models = {
    'ThInK 1.0': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10/think.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/think_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/think_LMRR_allPLOG.yaml",
            # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/think_LMRR_allP.yaml",
                    },
    },
    r'ThInK 1.0 (HO2-PLOG)': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10_HO2plog/think_ho2plog.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR_allPLOG.yaml",
            # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR_allP.yaml",
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r",'orange','xkcd:grey','goldenrod']*12
V = 4/3*np.pi*(diameter/2)**2 #JSR volume

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

def getTimeHistory(gas):
    gas.TPX = T, P*ct.one_atm, X
    stirredReactor = getStirredReactor(gas)
    reactorNetwork = ct.ReactorNet([stirredReactor])
    states = ct.SolutionArray(stirredReactor.thermo)
    t = 0
    tvec=[]
    Xvec=[]
    while t < t_max:
        t = reactorNetwork.step()
        states.append(stirredReactor.thermo.state)
        tvec.append(t)
    Xvec=[states(species).X.flatten() for species in observables]
    return tvec, Xvec

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    X_history=getTimeHistory(gas)
    # print(X_history)
    for z, species in enumerate(observables):
        # print(X_history[1])
        Xi_history = X_history[1][z]
        # print(len(X_history[0]))
        data = zip(X_history[0],Xi_history)
        simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/JSR/{m}/{species}'
        os.makedirs(simOutPath,exist_ok=True)
        save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,len(observables), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.25)
for j,model in enumerate(models):
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        flag=False
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
            
            if exp and j==len(models)-1 and z!=1:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
                ax[z].plot(dat.iloc[:,0],dat.iloc[:,1]*1e6,'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[z].set_xlim(Xlim)
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_xlabel('Temperature [K]')
        print('  > Data added to plot')
ax[0].set_ylabel(f'Mole fraction [ppm]')
ax[0].annotate(f'{title}', xy=(0.07, 0.96), xycoords='axes fraction',ha='left', va='top',fontsize=lgdfsz)
ax[0].annotate(r'CH$_4$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[1].annotate(r'C$_2$H$_2$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[2].annotate(r'C$_2$H$_4$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[3].annotate(r'C$_2$H$_6$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1)
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/JSR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
