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
title='18.8% NH3/2.1% CH4/9.1% O2/70% Ar'
folder='Glarborg-2025'
name='Fig9'
exp=False
dataLabel='Dai et al. (2020)'
data=['6a_2phi.csv']

fuel={'NH3':0.7,'H2':0.3}
oxidizer='O2'
diluent='AR'
fraction={'diluent':0.92}
phi=1
# X={'H2':1,'NH3':9,'O2':7.25,'N2':3.4,'Ar':23.85}
P=10
T_list = np.linspace(1181,1500,gridsz)
t_max=300e-5
Xlim=[900,1500]
Ylim=[1,1000]
indicator='oh' # oh, oh*, h, o, pressure

models = {
    'Glarborg-2025': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allPLOG.yaml",
                    },
    },
    # 'Glarborg-2025_NH3PLOG': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO_NH3PLOG.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_NH3PLOG_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_NH3PLOG_LMRR_allPLOG.yaml",
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

def getTimeHistory(gas,T):
    def ignitionDelay(states, species):
        # max_value = np.gradient(states(species).Y.T[0]).max()
        # threshold=max_value/2
        # i_ign = (abs(np.gradient(states(species).Y.T[0]) - threshold)).argmin()
        # i_ign = np.gradient(states(species).X.T[0]).argmax()
        # Rjoule = 38.31446261815324 # [J/K/mol]
        # conc = np.array(states(species).Y.T[0])*P*ct.one_atm/Rjoule/T
        # max_value = conc.max()
        max_value = states(species).X.T[0].max()
        threshold = max_value / 2
        # # # i_ign = (abs(conc - threshold)).argmin()
        i_ign = (abs(np.array(states(species).Y.T[0]) - threshold)).argmin()
        # # print(i_ign)
        # i_ign = states(species).Y.T[0].argmax()
        # # print(i_ign)
        return states.t[i_ign]   
    gas.TP = T, P*ct.one_atm
    gas.set_equivalence_ratio(phi,fuel,oxidizer,diluent=diluent,fraction=fraction)
    r = ct.Reactor(contents=gas,energy='on')
    reactorNetwork = ct.ReactorNet([r])
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    t=0
    while t < t_max:
        t=reactorNetwork.step()
        timeHistory.append(r.thermo.state, t=t)
    return ignitionDelay(timeHistory, indicator)

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    # IDT_list = Parallel(n_jobs=len(T_list))(
    #     delayed(getTimeHistory)(gas,T)
    #     for T in T_list
    # )
    IDT_list=[getTimeHistory(gas,T) for T in T_list]
    data = zip(T_list,IDT_list)
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/IDT/{m}'
    os.makedirs(simOutPath,exist_ok=True)
    save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        simFile=f'USSCI/data/{args.date}/{folder}/{model}/IDT/{m}/{name}.csv'
        # if not os.path.exists(simFile):
        sims=generateData(model,m)  
        sims=pd.read_csv(simFile)
        label = f'{model}' if k == 0 else None
        ax.semilogy(sims.iloc[:,0],sims.iloc[:,1]*1e6, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
        if exp and j==len(models)-1 and k==2:
            dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[0]}',header=None)
            ax.semilogy(dat.iloc[:,0],dat.iloc[:,1]*1e6,'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
        # ax.set_xlim(Xlim)
        # ax.set_ylim(Ylim)
        ax.tick_params(axis='both',direction='in')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel(f'Ignition delay [mu-s]')
        print('  > Data added to plot')
ax.annotate(f'{title}\n60bar', xy=(0.97, 0.95), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz)
ax.legend(fontsize=lgdfsz,frameon=False,loc='lower left', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/IDT'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
