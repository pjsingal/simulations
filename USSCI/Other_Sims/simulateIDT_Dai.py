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
folder='Dai-2020'
name='Fig6'
exp=True
dataLabel='Dai et al. (2020)'
data=['6a_2phi.csv']

X={'NH3':0.188,'CH4':0.021,'O2':0.091,'N2':0,'AR':0.7} #mixture6
# X={'NH3':0.118,'CH4':0,'O2':0.176,'N2':0,'AR':0.706} #m1
# X={'NH3':0.104,'CH4':0.006,'O2':0.178,'N2':0,'AR':0.712} #m2
# X={'NH3':0.04,'CH4':0.04,'O2':0.22,'N2':0.2,'AR':0.5} #m4
P=59.215396 #60bar
T_list = np.linspace(940,1060,gridsz)
Xlim=[940,1060]
indicator='o' # oh, oh*, h, o, pressure

models = {
    'Klippenstein-CNF2018': {
        'submodels': {
            'base': r"chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
                    },
    },
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
    # 'Cornell-2024': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Cornell-2024/cornell-2024.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/cornell-2024_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/cornell-2024_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Arunthanayothin-2021': { #bad
    #     'submodels': {
    #         'base': r'chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml',
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/arunthanayothin-2021_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Gutierrez-2025': { #don't use this here because sims keep breaking
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
    # 'Song-2019': {  #bad
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Song-2019/song-2019.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/song-2019_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Mei-2019': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Mei-2019/mei-2019.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR_allPLOG.yaml",
    #                 },
    # },
    # 'Aramco-3.0': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/AramcoMech30/aramco30.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allPLOG.yaml",
    #                 },
    # },
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
        i_ign = np.gradient(states(species).Y.T[0]).argmax()
        # i_ign = states(species).Y.T[0].argmax()
        return states.t[i_ign]   
    gas.TPX = T, P*ct.one_atm, X
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
        if not os.path.exists(simFile):
            sims=generateData(model,m)  
        sims=pd.read_csv(simFile)
        label = f'{model}' if k == 0 else None
        ax.semilogy(sims.iloc[:,0],sims.iloc[:,1]*1e3, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
        if exp and j==len(models)-1 and k==2:
            dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[0]}',header=None)
            ax.semilogy(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
        ax.set_xlim(Xlim)
        ax.tick_params(axis='both',direction='in')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel(f'Ignition delay [ms]')
        print('  > Data added to plot')
ax.annotate(f'{title}\n60bar', xy=(0.97, 0.95), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz)
ax.legend(fontsize=lgdfsz,frameon=False,loc='lower left', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/IDT'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
