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
title=[f'3% H2/1.5% O2/Ar\n17atm',
       f'3% H2/1.5% O2/N2\n13atm',
       f'3% H2/1.5% O2/\n9% H2O/Ar\n15atm',
       f'3% H2/1.5% O2/\n20% CO2/Ar\n12atm']
folder='Shao-2019'
name='Fig8'
exp=True
dataLabel='Shao et al. (2019)'
data=['1.csv','2.csv','3.csv','4.csv']

X_list=[{'H2':0.03, 'O2':0.015, 'Ar':1-0.03-0.015},
        {'H2':0.03, 'O2':0.015, 'N2':1-0.03-0.015},
        {'H2':0.03, 'O2':0.015, 'H2O':0.09, 'Ar':1-0.03-0.015-0.09},
        {'H2':0.03, 'O2':0.015, 'CO2':0.20, 'Ar':1-0.2-0.03-0.015}]
# X_list=[{'H2':0.03, 'O2':0.015, 'Ar':1-0.03-0.015},
#         {'H2':0.03, 'O2':0.015, 'N2':1-0.03-0.015},
#         {'H2':0.03, 'O2':0.015, 'H2O':0.09, 'Ar':1-0.03-0.015-0.09}]
# P=33
P_list=[17,13,15,12]
T_range_list = [np.linspace(1100,1300,gridsz),
                np.linspace(1100,1300,gridsz),
                np.linspace(1200,1400,gridsz),
                np.linspace(1100,1300,gridsz)]
# T_range_list = [np.linspace(1100,1300,gridsz),
#                 np.linspace(1100,1300,gridsz),
#                 np.linspace(1200,1400,gridsz)]
Xlim=[1100,1400]
indicator='oh' # oh, oh*, h, o, pressure

models = {
    'ThInK 1.0': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10/think.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/think_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/think_LMRR_allPLOG.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/think_LMRR_allP.yaml",
                    },
    },
    r'ThInK 1.0 (HO2-PLOG)': {
        'submodels': {
            'base': r"chemical_mechanisms/ThinkMech10_HO2plog/think_ho2plog.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR_allPLOG.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/think_ho2plog_LMRR_allP.yaml",
                    },
    },
}
########################################################################################
lstyles = ["solid","dashed","dotted","dashdot"]*6
colors = ["xkcd:purple","xkcd:teal","r",'orange']

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getTimeHistory(gas,T,P,X):
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

#Filter species
def filteredModel(model):
    #Filter species
    all_species = ct.Species.list_from_file(model)
    species=[]
    for S in all_species:
        comp = S.composition
        if 'C' in comp:
            continue
        species.append(S)
    species_names = {S.name for S in species}
    #Filter reactions
    ref_phase = ct.Solution(thermo='ideal-gas',kinetics='gas',species=all_species)
    all_reactions = ct.Reaction.list_from_file(model,ref_phase)
    reactions=[]
    for R in all_reactions:
        if not all(reactant in species_names for reactant in R.reactants):
            continue
        if not all(product in species_names for product in R.products):
            continue
        reactions.append(R)
    gas = ct.Solution(name='reducedThink',thermo='ideal-gas',kinetics='gas',transport_model='mixture-averaged',species=species,reactions=reactions)
    # gas.write_yaml("reducedThink.yaml")
    return gas

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    # gas = filteredModel(models[model]['submodels'][m])
    gas = ct.Solution(models[model]['submodels'][m])
    for i,T_list in enumerate(T_range_list):
        # IDT_list = Parallel(n_jobs=len(T_list))(
        #     delayed(getTimeHistory)(gas,T,P_list[i],X_list[i])
        #     for T in T_list
        # )
        IDT_list = [getTimeHistory(gas,T,P_list[i],X_list[i]) for T in T_list]
        data = zip(T_list,IDT_list)
        simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/IDT/{i}/{m}'
        os.makedirs(simOutPath,exist_ok=True)
        save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')
print(folder)
tic1=time.time()
f, ax = plt.subplots(1,4, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    print(f'Model: {model}')
    ax[2].plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        for i,T_list in enumerate(T_range_list):
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/IDT/{i}/{m}/{name}.csv'
            if not os.path.exists(simFile):
                sims=generateData(model,m)  
            sims=pd.read_csv(simFile)
            label = f'{m}'
            ax[i].semilogy(sims.iloc[:,0],sims.iloc[:,1]*1e6, color=colors[j], linestyle=lstyles[k], linewidth=lw, label=label)
            if exp and j==len(models)-1 and k==2:
                dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[i]}')
                ax[i].semilogy(dat['T'],dat['IDT'],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
            ax[i].set_xlim(Xlim)
            ax[i].tick_params(axis='both',direction='in')
            ax[i].set_xlabel('Temperature [K]')
            ax[i].set_ylabel(f'Ignition delay [mu-s]')
            ax[i].annotate(f'{title[i]}', xy=(0.95, 0.96), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz)
        print('  > Data added to plot')
# plt.suptitle(f'{title}',fontsize=10)
ax[2].legend(fontsize=lgdfsz,frameon=False,loc='lower left', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/IDT'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
