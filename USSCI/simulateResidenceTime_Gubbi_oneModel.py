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
# title=r'Gubbi Residence Time using Alzueta-2023'
title=r'NH$_3$/air'+'\n'+r'$\phi$=1.22'+'\n'+r'T$_{NH_3}$=300 K'+'\n'+r'T$_{air}$=650 K'+'\n'+r'P=20 bar'
folder='Gubbi-2023_oneModel'
name='Fig3'
exp=False
dataLabel='Liu et al. (2019)'
data=['4c_2phi.csv']

fuel='NH3'
phi=2
P=20/1.013 #20 bar
T_fuel = 300
T_air = 650
phi = 1.22
cutoff=int(1)
Xlim=[0.1,1000]
Ylim=[4,1000]
indicator='NH2' # oh, oh*, h, o, pressure
width=0.05

models = {
    'Alzueta-2023': {
        'submodels': {
            # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
            'Ar':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allAR.yaml",
            r'H$_2$O':"chemical_mechanisms/Alzueta-2023/alzuetamechanism_LMRR_allH2O.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml",
            'LMRR-allPLOG': f"USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allPLOG.yaml",
                    },
    },
}
########################################################################################
lstyles = ["dashed","dashed","solid","dotted"]*6
colors = ['r', 'b',"xkcd:purple",'orange',"xkcd:teal", 'xkcd:grey',"goldenrod"]*12
linewidth = [lw*0.5,lw*0.5,lw,lw]

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
def getFlame(gaz,P):
    largeur=width
    flame = ct.FreeFlame(gaz,width=largeur)
    flame.set_refine_criteria(ratio=3, slope=0.05, curve=0.05)
    # flame.set_refine_criteria(ratio=5, slope=0.5, curve=0.5)
    flame.soret_enabled = True
    flame.transport_model = 'multicomponent'
    flame.solve(loglevel=0, auto=True)
    etat = {
        'T [K]': flame.T,
        # 'P': flame.P,
        'vel [m/s]': flame.velocity,
        # 'X': flame.X,
        'dist [m]': flame.grid #list of distances [m]
    }
    for species in gaz.species_names:
        etat[f"X_{species}"] = flame.X[gaz.species_names.index(species)]
    S_b = list(etat['vel [m/s]'])[-1] # [m/s]
    Rjoule = 8.31446261815324 # [J/K/mol]
    full_time = [(distance)/S_b for distance in etat['dist [m]']] # [s]
    conc = np.multiply(flame.X[gaz.species_names.index(indicator)], np.divide(P*ct.one_atm,np.multiply(Rjoule,etat['T [K]'])))
    distance_index=np.argmax(conc)
    tau_f = np.array(etat['dist [m]'][distance_index])/S_b
    etat['tau [ms]']=[(ft - tau_f)*1000 for ft in full_time] # [ms]
    return etat

def setTPX(gas,submodel):
    def cp(T,P,X):
        gas_stream = ct.Solution(submodel)
        gas_stream.TPX = T, P*ct.one_atm, {X:1}
        return gas_stream.cp_mole # [J/kmol/K]
    cp_fuel = cp(T_fuel,P,fuel) # [J/kmol/K]
    cp_o2 = cp(T_air,P,'O2') # [J/kmol/K]
    cp_n2 = cp(T_air,P,'N2') # [J/kmol/K]
    x_fuel = (phi*(1/0.75)*0.21)/(1+phi*(1/0.75)*0.21)
    x_o2 = 0.21*(1-x_fuel)
    x_n2 = 0.79*(1-x_fuel)
    # x_air=1-x_fuel
    T_mix = (x_fuel*cp_fuel*T_fuel+(x_o2*cp_o2+x_n2*cp_n2)*T_air)/(x_fuel*cp_fuel+ x_o2*cp_o2 + x_n2*cp_n2)
    gas.TPX = T_mix, P*1e5,{'NH3':x_fuel,'O2':x_o2,'N2':x_n2}

def getXNOdry(X_NO,X_O2):
    X_O2dry = 0.15 # 15% O2 dry
    return np.multiply(np.multiply(X_NO,np.divide(0.21-X_O2dry,np.subtract(0.21,X_O2))),1e6) # [ppm, 15% O2 dry]

def generateData(model,m):
    print(f'  Generating species data')
    tic2 = time.time()
    gas = ct.Solution(models[model]['submodels'][m])
    setTPX(gas,models[model]['submodels'][m])
    state=getFlame(gas,P)
    XNO_dry = getXNOdry(list(state['X_NO']),list(state['X_O2']))[:cutoff*(-1)]
    tau = list(state['tau [ms]'])[:cutoff*(-1)]
    data = zip(tau,XNO_dry)
    simOutPath = f'USSCI/data/{args.date}/{folder}/{model}/ResidenceTime/{m}'
    os.makedirs(simOutPath,exist_ok=True)
    save_to_csv(f'{simOutPath}/{name}.csv', data)
    toc2 = time.time()
    print(f'  > Simulated in {round(toc2-tic2,2)}s')

print(folder)
tic1=time.time()
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight))
plt.subplots_adjust(wspace=0.3)
for j,model in enumerate(models):
    ax.plot(0, 0, '.', color='white',markersize=0.1,label=f'{model}') 
    print(f'Model: {model}')
    for k,m in enumerate(models[model]['submodels']):
        print(f' Submodel: {m}')
        simFile=f'USSCI/data/{args.date}/{folder}/{model}/ResidenceTime/{m}/{name}.csv'
        if not os.path.exists(simFile):
            sims=generateData(model,m)  
        sims=pd.read_csv(simFile)
        label = f'{m}'
        ax.loglog(np.multiply(P,sims.iloc[:,0]),sims.iloc[:,1], color=colors[k], linestyle=lstyles[k], linewidth=linewidth[k], label=label)
        if exp and j==len(models)-1 and k==2:
            dat = pd.read_csv(f'USSCI/graph-reading/{folder}/{data[0]}',header=None)
            ax.loglog(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label=dataLabel)
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylim)
        ax.tick_params(axis='both',direction='in')
        ax.set_xlabel(r'P*$\tau$ [bar*ms]')
        ax.set_ylabel(r'NO [ppm, 15% O$_2$ dry]')
        print('  > Data added to plot')
ax.annotate(f'{title}', xy=(0.97, 0.97), xycoords='axes fraction',ha='right', va='top',fontsize=lgdfsz+1)

ax.legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw,ncol=1) 
toc1=time.time()
outPath=f'USSCI/figures/{args.date}/{folder}/ResidenceTime'
os.makedirs(outPath,exist_ok=True)
name=f'{name}.png'
plt.savefig(f'{outPath}/{name}', dpi=500, bbox_inches='tight')
print(f'Figure generated in {round(toc1-tic1,3)}s')
