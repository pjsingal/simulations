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
warnings.filterwarnings("ignore", message=".*discontinuity.*detected.*")
warnings.filterwarnings("ignore", message=".*return _ForkingPickler.loads.*")
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
title='Flow reactor'
folder='Rasmussen-2008'
name='Fig8'
exp=False
override=False
dataLabel='Gutierrez et al. (2025)'
data=['XCH4_90CH4_10NH3.csv','XNO_90CH4_10NH3.csv']
# observables=['NH3','CH3OCH3']
# observables=['NH3','CH3OCH3']
observables=['CO2', 'CO', 'NO', 'NO2']


# X1=502e-6
# X2=440e-6
# X3=0.0148
# X4=145e-6
# X5=6e-6
# Xdil=1-X1-X2-X3-X4-X5
# X={'CO':X1,'H2':X2,'O2':X3,'NO2':X4,'NO':X5,'N2':Xdil}
# P=100/1.01325

X1=518e-6
X2=453e-6
X3=0.0153
X4=36e-6
X5=113e-6
Xdil=1-X1-X2-X3-X4-X5
X={'CO':X1,'H2':X2,'O2':X3,'NO2':X4,'NO':X5,'N2':Xdil}
P=20/1.01325

T_list = np.linspace(598,898,gridsz)
Xlim=[600,900]
Ylim=[0,1.4]
length = 50e-2  # [m]
diameter=0.008 # [m]
n_steps = 3000
Q_std = 3000 #nominal gas flow rate @ STP [mL/min]

models = {
    'Klippenstein-CNF2018': {
        'submodels': {
            'base': r"chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR.yaml",
            # 'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
            #         },
            'LMRR-allPdep': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
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

def getTemperatureDependence(gas,T):
    # Get nonstandard volumetric flow rate:
    Q = Q_std*(1e5/(P*ct.one_atm))*(273.15/T)
    gas.TPX = T, P*ct.one_atm, X
    area = np.pi*(diameter/2)**2
    u0 = Q*1e-6/60/area
    mass_flow_rate = u0 * gas.density * area
    flowReactor = ct.IdealGasConstPressureReactor(gas,energy='off')
    reactorNetwork = ct.ReactorNet([flowReactor])
    tau=length/u0
    dt = tau/n_steps
    t1 = (np.arange(n_steps) + 1) * dt
    states = ct.SolutionArray(flowReactor.thermo)
    for n, t_i in enumerate(t1):
        reactorNetwork.advance(t_i)
    states.append(flowReactor.thermo.state)
    return [states(species).X.flatten()[0] for species in observables]

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    # print(np.where(np.isnan(gas.forward_rate_constants)))
    X_history=[getTemperatureDependence(gas,T) for T in T_list]
    for z, species in enumerate(observables):
        Xi_history = [item[z] for item in X_history]
        data = zip(T_list,Xi_history)
        simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/FR/{m}/{species}'
        os.makedirs(simOutPath,exist_ok=True)
        save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
    return X_history
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,len(observables), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        sims=generateData(model,m) 
        # flag=False
        # while not flag:
        #     for z, species in enumerate(observables):
        #         simFile=f'USSCI/data/{args.date}/{folder}/{model}/FR/{m}/{species}/{name}.csv'
        #         if not os.path.exists(simFile) or override:
        #             sims=generateData(model,m) 
        #             flag=True
        #     flag=True
        for z, species in enumerate(observables):  
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/FR/{m}/{species}/{name}.csv' 
            sims=pd.read_csv(simFile)
            label = f'{model}-{m}'
            ax[z].plot(sims.iloc[:,0],sims.iloc[:,1]*1e6, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[z].set_ylabel(f'X-{species} [ppm]')
            if exp and j==len(models)-1 and k==2:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[z]}',header=None)
                ax[z].plot(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[z].set_xlim(Xlim)
            # ax[z].set_ylim(Ylim)
            ax[z].tick_params(axis='both',direction='in')
            ax[z].set_xlabel('Temperature [K]')
        print('  > Data added to plot')
plt.suptitle(f'{title}',fontsize=10)
ax[len(observables)-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/FR'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
