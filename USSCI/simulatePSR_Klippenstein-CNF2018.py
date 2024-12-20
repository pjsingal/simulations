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
title=f'T = 1800 K\nP = 100 atm\n'+r'$\tau$ = 3 ms'
folder='Klippenstein-CNF2018'
name='Fig14_1atm'
exp=False
# dataLabel='Lavadera et al. (2018)'
# data=['XCH4_55N2_45H2O.csv','XC2H2_55N2_45H2O.csv','XC2H4_55N2_45H2O.csv','XC2H6_55N2_45H2O.csv']
observable='NO'

P=1
fuel='CH4'
oxidizerList=['O2:0.21,N2:0.79', 'O2:0.21,N2:0.4,H2O:0.39']
phiList = np.linspace(0.7,1.5,gridsz)
T=1800
Xlim=[0.6,1.6]
tau=3e-3
V=113e-6 #value wasn't reported in his paper
t_max=50

models = {
    'Klippenstein-CNF2018': {
        'submodels': {
            'base': r"chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allP.yaml",
                    },
    },
    # 'Glarborg-2018': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allPLOG.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
    #                 },
    # },
}
########################################################################################
lstyles = ["solid","dashed","dotted", "dashdot"]*6
colors = ["xkcd:purple","r","xkcd:teal",'orange']*3
# V = 4/3*np.pi*(diameter/2)**2 #PSR volume

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
    reactor = ct.IdealGasReactor(gas, energy='off', volume=V)
    ct.MassFlowController(upstream=fuelAirMixtureTank,
                          downstream=reactor,
                          mdot=reactor.mass/tau)
    ct.Valve(upstream=reactor,
             downstream=exhaust,
             K=K)
    ct.Wall(reactor, env, A=reactorSurfaceArea, U=h)
    return reactor

def getPhiDependence(gas,phi):
    gas.TP = T, P*ct.one_atm
    Xvec=[]
    for oxidizer in oxidizerList:
        gas.set_equivalence_ratio(phi,fuel,oxidizer)
        stirredReactor = getStirredReactor(gas)
        reactorNetwork = ct.ReactorNet([stirredReactor])
        states = ct.SolutionArray(stirredReactor.thermo)
        t = 0
        while t < t_max:
            t = reactorNetwork.step()
        states.append(stirredReactor.thermo.state)
        Xvec.append(states(observable).X.flatten()[0])
    return Xvec

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    # print(np.where(np.isnan(gas.forward_rate_constants)))
    X_history=[getPhiDependence(gas,phi) for phi in phiList]
    for z, oxidizer in enumerate(oxidizerList):
        Xi_history = [item[z] for item in X_history]
        data = zip(phiList,Xi_history)
        simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/PSR/{m}/{oxidizer}'
        os.makedirs(simOutPath,exist_ok=True)
        save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
f, ax = plt.subplots(len(oxidizerList),1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.25)
for j,model in enumerate(models):
    print(f'Model: {model}')
    ax[1].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        flag=False
        while not flag:
            for z, oxidizer in enumerate(oxidizerList):
                simFile=f'USSCI/data/{args.date}/{folder}/{model}/PSR/{m}/{oxidizer}/{name}.csv'
                if not os.path.exists(simFile):
                    sims=generateData(model,m) 
                    flag=True
            flag=True
        for z, oxidizer in enumerate(oxidizerList):   
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/PSR/{m}/{oxidizer}/{name}.csv'
            sims=pd.read_csv(simFile)
            label = f'{m}'
            ax[z].plot(sims.iloc[:,0],sims.iloc[:,1]*1e6, color=colors[k], linestyle=lstyles[k], linewidth=lw, label=label)
            
            # if exp and j==len(models)-1 and k==2:
            #     dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
            #     ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[z].set_xlim(Xlim)
            ax[z].tick_params(axis='both',direction='in')
            
            ax[z].set_ylabel(f'NO mole fraction [ppm]')
        print('  > Data added to plot')
ax[0].set_xlabel('Equivalence ratio [-]')
ax[0].annotate(f'{title}', xy=(0.07, 0.96), xycoords='axes fraction',ha='left', va='top',fontsize=lgdfsz+2)
ax[0].annotate(f'CH4/air', xy=(0.09, 0.05), xycoords='axes fraction',ha='left', va='bottom',fontsize=lgdfsz+2)
ax[1].annotate(f'CH4/({oxidizerList[1]})', xy=(0.09, 0.05), xycoords='axes fraction',ha='left', va='bottom',fontsize=lgdfsz+2)
ax[1].legend(fontsize=lgdfsz,frameon=False,loc='upper left', handlelength=lgdw,ncol=1)
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/PSR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
