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
import matplotlib as mpl
import argparse

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
models = {
    # 'Stagni-2023': {
    #     # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2023_LMRR_allP.yaml",
    #             },
    # 'Alzueta-2023': {
    #     # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
    #             },
    # 'Glarborg-2018': {
    #     # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
    #             },
    'Aramco-3.0': {
        # 'base': r"chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
                },
}

T_list = [np.linspace(800,1100,gridsz),np.linspace(750,850,gridsz),np.linspace(750,825,gridsz),np.linspace(750,800,gridsz)]
dilution = 0.94
codiluentList = ['NH3', 'H2O']
codiluentPercentList = [0, 0.20, 0.40]
H2Percent = 0.3
O2Percent = 0.3
Tin_list = [600, 750, 850, 1000] # Kelvin
P_list = [1.2, 25, 50, 100] # [atm]
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
########################################################################################

def getStirredReactor(gas,V,tau,K,h):
    reactorRadius = (V*3/4/np.pi)**(1/3) # [m3]
    reactorSurfaceArea =4*np.pi*reactorRadius**2 # [m3]
    fuelAirMixtureTank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)
    env = ct.Reservoir(gas)
    reactor = ct.IdealGasReactor(gas, energy='on', volume=V)
    ct.MassFlowController(upstream=fuelAirMixtureTank,
                          downstream=reactor,
                          mdot=reactor.mass/tau)
    ct.Valve(upstream=reactor,
             downstream=exhaust,
             K=K)
    ct.Wall(reactor, env, A=reactorSurfaceArea, U=h)
    return reactor

def getTemperatureDependence(gas,V,tau,K,h,T_list,P,X,t_max):
    stirredReactor = getStirredReactor(gas,V,tau,K,h)
    columnNames = (
        ['pressure'] +
        [stirredReactor.component_name(item)
         for item in range(stirredReactor.n_vars)]
    )
    tempDependence = pd.DataFrame(columns=columnNames)
    for T in T_list:
        gas.TPX = T, P*ct.one_atm, X
        stirredReactor = getStirredReactor(gas,V,tau,K,h)
        reactorNetwork = ct.ReactorNet([stirredReactor])
        t = 0
        while t < t_max:
            t = reactorNetwork.step()
        state = np.hstack([stirredReactor.thermo.P,
                        stirredReactor.mass,
                        stirredReactor.volume,
                        stirredReactor.T,
                        stirredReactor.thermo.X])
        tempDependence.loc[T] = state
    return tempDependence

def generatePlot(model,codiluent,TP_list,val1,option):
    f, ax = plt.subplots(len(P_list), 3, figsize=(args.figwidth,args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    for z, val2 in enumerate(TP_list):
        if option=='P':
            print(f'Pressure: {val2}atm')
        if option=='T':
            print(f'Temperature: {val2}K')
        for k,m in enumerate(models[model]):
            print(f'Submodel: {m}')
            for i, codiluentPercent in enumerate(codiluentPercentList):
                print(f'{codiluent}: {codiluentPercent}%')
                X = {'H2': H2Percent, 'O2': O2Percent, 'AR': dilution*(1-codiluentPercent), codiluent: dilution*codiluentPercent}
                gas = ct.Solution(list(models[model].values())[k])
                if option=='P': # TP_list contains pressures and TP is a temperature
                    newT_list = T_list[z]
                    T, P = val1, val2
                if option=='T': # TP_list contains temperatures and TP is a pressure
                    newT_list = T_list[0]
                    T, P = val2, val1
                gas.TPX = T, P*ct.one_atm, X
                tau = 0.5 # residence time [s]
                V = 0.000113 #30.5*(1e-2)**3 reactor volume [m3]
                h = 79.5 # heat transfer coefficient W/m2/K
                K = 2e-5 # pressureValveCoefficient
                t_max = 50  # max simulation time [s]
                tempDependence = getTemperatureDependence(gas,V,tau,K,h,newT_list,P,X,t_max)
                ax[z,0].plot(tempDependence.index,np.subtract(tempDependence['temperature'],tempDependence.index),color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {codiluentPercent}% {codiluent}')   
                ax[z,1].plot(tempDependence.index,tempDependence['O2']*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {codiluentPercent}% {codiluent}')   
                ax[z,2].plot(tempDependence.index,tempDependence['H2']*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {codiluentPercent}% {codiluent}')
        ax[z,1].set_title(f"JSR {model} ({P}atm, {T}K, {H2Percent}% H2/{O2Percent}% O2, {dilution}% {codiluent}/Ar)",fontsize=args.fsz)
        ax[z,0].tick_params(axis='both',direction='in')
        ax[z,1].tick_params(axis='both',direction='in')
        ax[z,2].tick_params(axis='both',direction='in')
        ax[z,0].set_ylabel(r'$\Delta$ T [K]')
        ax[z,1].set_ylabel('O$_2$ mole fraction [%]')
        ax[z,2].set_ylabel('H$_2$ mole fraction [%]')
        print('Subplot added to figure!')
    ax[len(P_list)-1,2].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
    ax[len(P_list)-1,1].set_xlabel('Temperature [K]')
    path=f'USSCI/figures/'+args.date+'/JSR'
    os.makedirs(path,exist_ok=True)
    if option=='P':
        plt.savefig(path+f'/{model}_{codiluent}_P_dependence.png', dpi=500, bbox_inches='tight')
        print(f'Simulation has been stored at {path}/{model}_{codiluent}_P_dependence.png\n')
    if option=='T':
        plt.savefig(path+f'/{model}_{codiluent}_Tin_dependence.png', dpi=500, bbox_inches='tight')
        print(f'Simulation has been stored at {path}/{model}_{codiluent}_Tin_dependence.png\n')

for model in models:
    print(f'Model: {model}')
    if model == 'Aramco-3.0':
        newCodiluentList=['H2O']
    else:
        newCodiluentList=codiluentList
    for codiluent in newCodiluentList:
        print(f'Codiluent: {codiluent}')
        generatePlot(model,codiluent,P_list,1000,'P')
        generatePlot(model,codiluent,Tin_list,1.2,'T')