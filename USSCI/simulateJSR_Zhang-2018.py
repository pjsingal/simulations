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
title='Jet-stirred reactor: 2.31% C3H8/7.69% O2/67.5% N2/22.5% H2O (1.1atm)'
folder='Zhang-2018'
name='Fig8'
exp=False
data=['XCH4_75N2_25H2O.csv','XCO2_75N2_25H2O.csv','XCO_75N2_25H2O.csv']
# observables=['CO2','CH2O','CH4']
observables=['NH3','H2O']

# X={'C2H5OH':0.002,'O2':0.02,'N2':1-0.002-0.02} #ethanol mixed with o2 and n2 bath
X={'H2':0.04,'O2':0.12,'N2':1-0.04-0.12} #ethanol mixed with o2 and n2 bath
P=10
T_list = np.linspace(825,1075,gridsz)
Xlim=[700,1200]
tau=0.07
diameter=0.04 #m
V = 4/3*np.pi*(diameter/2)**2 #JSR volume
t_max=20

models = {
    # 'Zhang-2018': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2018/zhang-2018_ethanolDME.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2018_ethanolDME_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2018_ethanolDME_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Stagni-2023': {
        'submodels': {
            'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allPLOG.yaml",
                    },
    },
    'Alzueta-2023': {
        'submodels': {
            'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*3

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

def generateData(model,m,species):
    print(f'  Generating {species} data')
    tic2 = time.time()
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
    toc2 = time.time()
    data = zip(tempDependence.index,tempDependence[species])
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/{m}/{species}'
    os.makedirs(simOutPath,exist_ok=True)
    save_to_csv(f'{simOutPath}/{name}.csv', data)
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return tempDependence

tic1=time.time()
f, ax = plt.subplots(1,len(observables), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'\nModel: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        for z, species in enumerate(observables):
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/{m}/{species}/{name}.csv'
            if os.path.exists(simFile): #plot the data
                print(f'  {species} data already exists')
                sims=pd.read_csv(simFile)
            else:
                sims=generateData(model,m,species)
            label = f'{model}' if k == 0 else None
            ax[z].plot(sims.iloc[:,0],sims.iloc[:,1], color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[z].set_ylabel(f'X-{species} [%]')
            if exp and j==len(list(models.keys()))-1:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
                ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Lavadera et al.')
            ax[z].set_xlim(Xlim)
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_xlabel('Temperature [K]')
        print(' Data added to plot')
plt.suptitle(f'{title}',fontsize=10)
ax[len(observables)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')