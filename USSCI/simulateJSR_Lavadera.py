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
title=r'2.31% C$_3$H$_8$/7.69% O$_2$/'+f'\n'+r'49.5% N$_2$/40.5% H$_2$O'+f'\n1.1 atm'
folder='Lavadera-2018'
name='Fig3'
exp=True
dataLabel='Lavadera et al. (2018)'
data=['XCH4_55N2_45H2O.csv','XC2H2_55N2_45H2O.csv','XC2H4_55N2_45H2O.csv','XC2H6_55N2_45H2O.csv']
# observables=['O2','H2','CO','CO2','CH4','C2H4','C2H2']
observables=['CH4','C2H2','C2H4','C2H6']

X={'C3H8':0.0231,'O2':0.0769,'N2':0.495,'H2O':0.405} #phi=1.5, circle symbols
P=1.1
T_list = np.linspace(775,1100,gridsz)
Xlim=[773,1100]
tau=0.5
V=113e-6
t_max=50

models = {
    # 'Arunthanayothin-2021': {
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Stagni-2023': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Alzueta-2023': {
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Glarborg-2018': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Merchant-2015': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Merchant-2015/merchant-2015.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/merchant-2015_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Gutierrez-2025': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Gutierrez-2025/gutierrez-2025.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/gutierrez-2025_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/gutierrez-2025_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Bugler-2016': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Bugler-2016/bugler-2016.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Song-2019': {
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
}
########################################################################################
lstyles = ["solid","dashed","dotted"]*6
colors = ['goldenrod',"xkcd:teal","r",'orange','xkcd:grey',"xkcd:purple"]*12
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
f, ax = plt.subplots(len(observables),1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.25)
for j,model in enumerate(models):
    ax[1].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
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
            label = f'{m}'
            ax[z].plot(sims.iloc[:,0],sims.iloc[:,1]*100, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            
            if exp and j==len(models)-1 and k==len(models[model]['submodels'])-1:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
                ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[z].set_xlim(Xlim)
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_ylabel(f'Mole fraction [%]')
            
        print('  > Data added to plot')
ax[3].set_xlabel('Temperature [K]')

ax[0].annotate(f'{title}', xy=(0.07, 0.96), xycoords='axes fraction',ha='left', va='top',fontsize=lgdfsz)
ax[0].annotate(r'CH$_4$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[1].annotate(r'C$_2$H$_2$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[2].annotate(r'C$_2$H$_4$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[3].annotate(r'C$_2$H$_6$', xy=(0.91, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz+2)
ax[1].legend(fontsize=lgdfsz,frameon=False,loc='upper left', handlelength=lgdw,ncol=1)
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/JSR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
