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
title=r'C$_4$H$_10$/air'+f'\nP=20 atm\n$\phi$=2'
folder='Zhao-2021'
name='Fig7a'
exp=True
dataLabel='Zhao et al. (2021)'
data=['7a_2phi.csv']

# fuel='C4H10' # scroll down to body of code
oxidizer={'O2':0.21,'N2':0.79}
phi=2
P=20
# T_list = np.linspace(1191,1470,gridsz)
# Xlim=[1000/1180,1000/1500]
T_list = np.linspace(1440,645,gridsz)
Xlim=[1000/1538,1000/625]
Ylim=[0.01,1200]
indicator='o' # oh, oh*, h, o, pressure

models = {
    'Arunthanayothin-2021': { #use NC4H10
        'submodels': {
            'base': r'chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml',
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR_allPLOG.yaml",
                    },
    },
    # 'Bugler-2016': { # sim takes forever so probably unstable
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Bugler-2016/bugler-2016.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/bugler-2016_LMRR_allPLOG.yaml",
    #                 },
    # },
    'Song-2019': { #use NC4H10
        'submodels': {
            'base': r"chemical_mechanisms/Song-2019/song-2019.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR_allPLOG.yaml",
                    },
    },
    'Aramco-3.0': {
        'submodels': {
            'base': r"chemical_mechanisms/AramcoMech30/aramco30.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allPLOG.yaml",
                    },
    },
    # 'Zhang-2018': { #results are very wrong
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2018/zhang-2018_ethanolDME.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2018_ethanolDME_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2018_ethanolDME_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Zhang-2016': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2016/zhang-2016_nheptane.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2016_nheptane_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2016_nheptane_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Zhang-2015': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Zhang-2015/zhang-2015_nhexane.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2015_nhexane_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/zhang-2015_nhexane_LMRR_allPLOG.yaml",
    #                 },
    # },
}
########################################################################################
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","r"]*12

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getTimeHistory(gas,T):
    def ignitionDelay(states, species):
        i_ign = np.gradient(states(species).Y.T[0]).argmax()
        # i_ign = states(species).Y.T[0].argmax()
        return states.t[i_ign]   
    gas.TP = T, P*ct.one_atm
    gas.set_equivalence_ratio(phi,fuel, oxidizer)
    r = ct.Reactor(contents=gas)
    reactorNetwork = ct.ReactorNet([r])
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    t_end=1
    t=0
    while t < t_end:
        t=reactorNetwork.step()
        timeHistory.append(r.thermo.state, t=t)
    return ignitionDelay(timeHistory, indicator)

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    # IDT_list = Parallel(n_jobs=10)(
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
fuel_default='C4H10'
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    if model=='Arunthanayothin-2021' or model=='Song-2019':
        fuel='NC4H10'
    else:
        fuel=fuel_default
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        simFile=f'USSCI/data/{args.date}/{folder}/{model}/IDT/{m}/{name}.csv'
        if not os.path.exists(simFile):
            sims=generateData(model,m)  
        sims=pd.read_csv(simFile)
        label = f'{model}' if k == 0 else None
        ax.semilogy(np.divide(1000,sims.iloc[:,0]),sims.iloc[:,1]*1e3, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
        if exp and j==len(models)-1 and k==2:
            dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[0]}',header=None)
            ax.semilogy(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
        ax.set_xlim(Xlim)
        # ax.set_ylim(Ylim)
        ax.tick_params(axis='both',direction='in')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel(f'Ignition delay [ms]')
        print('  > Data added to plot')
ax.annotate(f'{title}', xy=(0.97, 0.05), xycoords='axes fraction',ha='right', va='bottom',fontsize=lgdfsz)
ax.legend(fontsize=lgdfsz,frameon=False,loc='upper left', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/IDT'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
