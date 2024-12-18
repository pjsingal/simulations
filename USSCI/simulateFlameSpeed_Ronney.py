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
fuel='NH3'
oxidizer={'O2':1,'N2':3.76}
phi_list = np.linspace(0.6,1.8,gridsz)
P=1
T = 296 #unburned gas temperature
Xlim=[0.6,1.8]
Ylim=[0,12]
width=0.03
title=f'{fuel}/air ({P} atm, {T} K)'
folder='Ronney-1986'
exp=True
dataLabel='Ronney (1986)'
data='760torr.csv'
name=f'{fuel}_{P}atm{T}K'

models = {
    # 'Glarborg-2025': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allPLOG.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allP.yaml",
    #                 },
    # },
    'Glarborg-2025-original': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2025-original/glarborg-2025-original.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-original_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-original_LMRR_allPLOG.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-original_LMRR_allP.yaml",
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted","dashdot"]*6
colors = ["xkcd:purple",'orange',"xkcd:teal", 'xkcd:grey',"goldenrod",'r', 'b']*12

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getFlameSpeed(gaz,phi):
    gaz.set_equivalence_ratio(phi,fuel,oxidizer)
    flame = ct.FreeFlame(gaz,width=width)
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.1)
    # flame.soret_enabled = True
    # flame.transport_model = 'multicomponent'
    flame.solve(loglevel=1, auto=True)
    return flame.velocity[0]*100 # cm/s

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    gas.TP = T, P*ct.one_atm
    # mbr=[getFlameSpeed(gas,phi) for phi in phi_list]
    mbr = Parallel(n_jobs=len(phi_list))(
        delayed(getFlameSpeed)(gas,phi)
        for phi in phi_list
    )
    data = zip(phi_list,mbr)
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/FlameSpeed/{m}'
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
    ax.plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        simFile=f'USSCI/data/{args.date}/{folder}/{model}/FlameSpeed/{m}/{name}.csv'
        if not os.path.exists(simFile):
            sims=generateData(model,m)  
        sims=pd.read_csv(simFile)
        label = f'{m}'
        ax.plot(sims.iloc[:,0],sims.iloc[:,1], color=colors[k], linestyle=lstyles[k], linewidth=lw, label=label)
        if exp and k==3:
            dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data}')
            NH3_list = np.divide(dat.iloc[:,0],100)
            ox_frac_list = np.subtract(1,NH3_list)
            O2_list = np.multiply(ox_frac_list, 0.21)
            phi_list = np.divide(np.divide(NH3_list,O2_list),np.divide(4,3))
            ax.plot(phi_list,dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylim)
        ax.tick_params(axis='both',direction='in')
        ax.set_xlabel(r'Equivalence Ratio')
        ax.set_ylabel(r'Burning velocity [cm $\rm s^{-1}$]')
        print('  > Data added to plot')
ax.annotate(f'{title}', xy=(0.05, 0.97), xycoords='axes fraction',ha='left', va='top',fontsize=lgdfsz+1)

ax.legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/FlameSpeed'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
