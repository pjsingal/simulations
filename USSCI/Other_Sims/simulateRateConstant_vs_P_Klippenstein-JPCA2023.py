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


folder='Klippenstein-JPCA2023'
name='Fig3_vs_P'

# H,2,N,O, ,+, ,M, ,<,=,>, ,H, ,+, ,H,N,O, ,+, ,M

P_list = np.linspace(1,100,gridsz)
T=2000
reactionList=['NH2 + O <=> H + HNO','H + HNO <=> H2NO','H + HNO <=> HNOH', 'NH + OH <=> H + HNO']
Xlim=[0,60]
Ylim=[1e-12,1e-9]
X={'H2O2':1}

models = {
    'Glarborg-2025': {
        'submodels': {
            'base': r"chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/glarborg-2025-HNNO_LMRR_allPLOG.yaml",
                    },
    },
    # 'Klippenstein-CNF2018': {
    #     'submodels': {
    #         'base': r"chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR.yaml",
    #         'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/klippenstein-CNF2018_LMRR_allPLOG.yaml",
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


def getRateConstant(gas,T,P,reaction):
    gas.TPX = T, P*ct.one_atm,X
    return gas.forward_rate_constants[gas.reaction_equations().index(reaction)]*1000/6.02e23

def generateData(model,m,reaction):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    # save_to_csv('data.csv', gas.reaction_equations())
    # print(gas.reaction_equations())
    k_list=[getRateConstant(gas,T,P,reaction) for P in P_list]
    data = zip(P_list,k_list)
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/RC_vs_P/{m}/{reaction}'
    os.makedirs(simOutPath,exist_ok=True)
    save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,len(reactionList), figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)

import matplotlib.ticker as ticker


for i, reaction in enumerate(reactionList):
    ax[i].xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[i].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    print(reaction)
    title=f'{reaction}\n{T}K'
    for j,model in enumerate(models):
        print(f'Model: {model}')
        for k,m in enumerate(models[model]['submodels']):
            print(f' Submodel: {m}')
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/RC_vs_P/{m}/{reaction}/{name}.csv'
            # if not os.path.exists(simFile):
            sims=generateData(model,m,reaction)  
            sims=pd.read_csv(simFile)
            print(sims.iloc[:,0][sims.iloc[:,1].argmax()])
            label = f'{model}' if k == 0 else None
            ax[i].semilogy(sims.iloc[:,0],sims.iloc[:,1], color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            ax[i].set_xlim(Xlim)
            ax[i].set_ylim(Ylim)
            ax[i].tick_params(axis='both',direction='in')
            ax[i].set_xlabel('Pressure [atm]')
            print('  > Data added to plot')
    ax[i].annotate(f'{title}', xy=(0.97, 0.95), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz)
ax[0].legend(fontsize=lgdfsz,frameon=False,loc='lower left', handlelength=lgdw,ncol=1)
ax[0].set_ylabel(f'rate constant [cm^3/molec/s]')
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/RC'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')


