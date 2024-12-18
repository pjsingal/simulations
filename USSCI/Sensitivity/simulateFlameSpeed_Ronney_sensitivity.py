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
phi=1.1
P=1
T = 296 #unburned gas temperature
Xlim=[0.6,3]
Ylim=[0,120]
width=0.03
title=f'({fuel}/air ({P} atm, {T} K'+r'$\phi$='+f'{phi})'
folder='Ronney-1986'
name=f'{fuel}_{P}atm{T}K{phi}phi_sensitivity'
cutoff=20
barWidth = 0.6
threshold = 0.01

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
patterns = ['','x','','x','','x',''] 

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def getSensitivity(gas):
    gas.set_equivalence_ratio(phi,fuel,oxidizer)
    f = ct.FreeFlame(gas,width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.1)
    f.solve(loglevel=1, auto=True)
    Su0=f.velocity[0] # m/s
    sensitivities = pd.DataFrame(index=gas.reaction_equations(), columns=["base_case"])
    dk = 0.1
    for r in range(gas.n_reactions):
        gas.set_multiplier(1.0)
        gas.set_multiplier(1+dk,r)
        f.solve(loglevel=0, refine_grid=False, auto=False)
        Su = f.velocity[0]
        sensitivities.iloc[r,0]=(Su-Su0)/(Su0*dk)
    gas.set_multiplier(1.0)
    sensitivities_subset = sensitivities[sensitivities["base_case"].abs() > threshold]
    return sensitivities_subset

def generateData(model,m):
    print(f'  Conducting sensitivity analysis')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    gas.TP = T, P*ct.one_atm
    # mbr=getFlameSpeed(gas,phi)
    sensitivities = getSensitivity(gas)
    data = zip(sensitivities.index,sensitivities["base_case"])
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
    reactionList=[]
    print(f'Model: {model}')
    # ax.plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        simFile=f'USSCI/data/{args.date}/{folder}/{model}/FlameSpeed/{m}/{name}.csv'
        if not os.path.exists(simFile):
            sims=generateData(model,m)  
        sims=pd.read_csv(simFile,header=None)
        reactionList += sims.iloc[:,0].to_list()
    reactionList = list(set(reactionList))
    masterDict={}
    for i, reaction in enumerate(reactionList):
        masterDict[reaction] = []
    for k,m in enumerate(models[model]['submodels']):  
        for i, reaction in enumerate(reactionList):
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/FlameSpeed/{m}/{name}.csv'
            sims=pd.read_csv(simFile,header=None)
            sims_sorted = sims.sort_values(by=1, ascending=False)
            rxns = sims_sorted.iloc[:,0].to_list()
            vals = sims_sorted.iloc[:,1].to_list()
            flag = 0
            for z, rxn, in enumerate(rxns):
                if rxn==reaction:
                    masterDict[reaction].append(vals[z])
                    flag=1
            if flag == 0: # the master list reaction does not exist in the model list
                masterDict[reaction].append(0)
    sumDict = {}
    for key in masterDict.keys():
        sumDict[key]=abs(sum(masterDict[key]))
    sumDict = dict(sorted(sumDict.items(), key=lambda item: item[1],reverse=True))
    weighted_reactionList = list(sumDict.keys())
    modelData=[]
    for k,m in enumerate(models[model]['submodels']):
        modelDict={}
        for i, reaction in enumerate(weighted_reactionList):
            simFile=f'USSCI/data/{args.date}/{folder}/{model}/FlameSpeed/{m}/{name}.csv'
            sims=pd.read_csv(simFile,header=None)
            sims_sorted = sims.sort_values(by=1, ascending=False)
            rxns = sims_sorted.iloc[:,0].to_list()
            vals = sims_sorted.iloc[:,1].to_list()
            flag = 0
            for z, rxn, in enumerate(rxns):
                if rxn==reaction:
                    modelDict[reaction]=vals[z]
                    flag=1
            if flag == 0: # the master list reaction does not exist in the model list
                modelDict[reaction]=0
        modelData.append(modelDict)
    num_submodels = len(list(models[model]['submodels'].keys()))
    offsets = np.linspace(-barWidth * (num_submodels - 1) / 2, barWidth * (num_submodels - 1) / 2, num_submodels)
    for k,m in enumerate(models[model]['submodels']):
        mdl_dict = modelData[k]
        rxns = list(mdl_dict.keys())[:cutoff]
        vals = list(mdl_dict.values())[:cutoff]
        y = np.arange(len(rxns)) # the label locations
        new_y = [5*i for i in y]
        label = f'{m}'
        ax.barh(new_y+offsets[k], vals, barWidth, label=label,color=colors[k],hatch=patterns[k]) 
        ax.set_xlabel(r"Sensitivity: $\frac{\partial\:\ln{S_{u}}}{\partial\:\ln{k}}$")
        ax.set_title(f'{model} {title}',fontsize=lgdfsz)
        ax.set_yticks(new_y, rxns, fontsize=6)
        ax.invert_yaxis()
        print('  > Data added to plot')

ax.legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
pos = ax.get_position()
ax.set_position([pos.x0*1.35, pos.y0, pos.width, pos.height])
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/FlameSpeed'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
