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

models = {
    'Alzueta-2023': {
        # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
        'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
        'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
                },
    'Stagni-2020': {
        # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
                },
    'Glarborg-2018': {
        # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
                },
}

T_list = np.linspace(800,1050,gridsz)
dilution = 0.94
codiluentList = ['NH3', 'H2O']
codiluentPercentList = [0, 0.20, 0.40]
H2Percent = 0.3
O2Percent = 0.3
lines =['-','--','-','-','-']
reactorTemperature = 1000 # Kelvin
reactorPressureList = np.multiply([1.2, 10, 25, 50],ct.one_atm)
residenceTime = 0.5 # tau [s]
reactorVolume = 0.000113 #30.5*(1e-2)**3  # m3
heatTransferCoefficient = 79.5 # W/m2/K
reactorRadius = np.cbrt(reactorVolume*3/4/np.pi) # [m3]
reactorSurfaceArea = 4*np.pi*np.square(reactorRadius) # m3
pressureValveCoefficient = 2e-5
maxPressureRiseAllowed = 0.01
maxSimulationTime = 50  # seconds
tempDependence = []


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

# colors = ["xkcd:grey"]*3 + ["xkcd:teal"]*3 + ["orange"]*3 + ['r']*3 + ['b']*3 + ['xkcd:purple']*3
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
# lstyles = ["solid"]*3 + ["dashed"]*3 + ["dotted"]*3

f, ax = plt.subplots(len(reactorPressureList), 3, figsize=(args.figwidth,args.figheight))
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.3)

for n, model in enumerate(models):
    for codiluent in codiluentList:
        for z, reactorPressure in enumerate(reactorPressureList):
            for k,m in enumerate(models[model]):
                for i, codiluentPercent in enumerate(codiluentPercentList):
                    X = {'H2': H2Percent, 'O2': O2Percent, 'AR': dilution*(1-codiluentPercent), codiluent: dilution*codiluentPercent}
                    gas = ct.Solution(list(models[model].values())[k])
                    gas.TPX = reactorTemperature, reactorPressure, X
                    fuelAirMixtureTank = ct.Reservoir(gas)
                    exhaust = ct.Reservoir(gas)
                    env = ct.Reservoir(gas)
                    stirredReactor = ct.IdealGasReactor(gas, energy='on', volume=reactorVolume)
                    massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                                                downstream=stirredReactor,
                                                                mdot=stirredReactor.mass/residenceTime)
                    pressureRegulator = ct.Valve(upstream=stirredReactor,
                                                downstream=exhaust,
                                                K=pressureValveCoefficient)
                    w2 = ct.Wall(stirredReactor, env, A=reactorSurfaceArea, U=heatTransferCoefficient)
                    reactorNetwork = ct.ReactorNet([stirredReactor])
                    columnNames = [stirredReactor.component_name(item) for item in range(stirredReactor.n_vars)]
                    columnNames = ['pressure'] + columnNames
                    timeHistory = pd.DataFrame(columns=columnNames)
                    tic = time.time()
                    t = 0
                    counter = 1
                    while t < maxSimulationTime:
                        t = reactorNetwork.step()
                        if(counter%10 == 0):
                            state = np.hstack([stirredReactor.thermo.P, stirredReactor.mass, 
                                        stirredReactor.volume, stirredReactor.T, stirredReactor.thermo.X])
                            timeHistory.loc[t] = state
                        counter += 1
                    toc = time.time()
                    pressureDifferential = timeHistory['pressure'].max()-timeHistory['pressure'].min()
                    if(abs(pressureDifferential/reactorPressure) > maxPressureRiseAllowed):
                        print("WARNING: Non-trivial pressure rise in the reactor. Adjust K value in valve")
                    tempDependence.append(pd.DataFrame(columns=timeHistory.columns))
                    tempDependence[i].index.name = 'Temperature'
                    for j,T in enumerate(T_list): #temperature in T:
                        reactorTemperature = T #temperature  # Kelvin
                        gas.TPX = reactorTemperature, reactorPressure, X
                        timeHistory = pd.DataFrame(columns=columnNames)
                        fuelAirMixtureTank = ct.Reservoir(gas)
                        exhaust = ct.Reservoir(gas)
                        env = ct.Reservoir(gas)
                        # gas.TPX = reactorTemperature, reactorPressure, concentrations
                        stirredReactor = ct.IdealGasReactor(gas, energy='on', volume=reactorVolume)
                        massFlowController = ct.MassFlowController(upstream=fuelAirMixtureTank,
                                                                downstream=stirredReactor,
                                                                mdot=stirredReactor.mass/residenceTime)
                        pressureRegulator = ct.Valve(upstream=stirredReactor, 
                                                    downstream=exhaust, 
                                                    K=pressureValveCoefficient)
                        w2 = ct.Wall(stirredReactor, env, A=reactorSurfaceArea, U=heatTransferCoefficient)
                        reactorNetwork = ct.ReactorNet([stirredReactor])
                        tic = time.time()
                        t = 0
                        while t < maxSimulationTime:
                            t = reactorNetwork.step()
                        state = np.hstack([stirredReactor.thermo.P, 
                                        stirredReactor.mass, 
                                        stirredReactor.volume, 
                                        stirredReactor.T, 
                                        stirredReactor.thermo.X])
                        toc = time.time()
                        concentrations = stirredReactor.thermo.X
                        tempDependence[i].loc[T] = state
                    if k==0:
                        ax[z,0].plot(tempDependence[i].index, np.subtract(tempDependence[i]['temperature'],tempDependence[i].index), color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {codiluentPercent}% {codiluent}')   
                        ax[z,1].plot(tempDependence[i].index, tempDependence[i]['O2']*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {codiluentPercent}'+r'% NH$_3$')   
                        ax[z,2].plot(tempDependence[i].index, tempDependence[i]['H2']*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} {codiluentPercent}'+r'% NH$_3$') 
                    if k==1:
                        ax[z,0].plot(tempDependence[i].index, np.subtract(tempDependence[i]['temperature'],tempDependence[i].index), color=colors[i], marker='x',fillstyle='none',linestyle='none',markersize=msz,markeredgewidth=mw, label=f'{m} {codiluentPercent}% {codiluent}')   
                        ax[z,1].plot(tempDependence[i].index, tempDependence[i]['O2']*100, color=colors[i], marker='x',fillstyle='none',linestyle='none',markersize=msz,markeredgewidth=mw, label=f'{m} {codiluentPercent}% {codiluent}')   
                        ax[z,2].plot(tempDependence[i].index, tempDependence[i]['H2']*100, color=colors[i], marker='x',fillstyle='none',linestyle='none',markersize=msz,markeredgewidth=mw, label=f'{m} {codiluentPercent}% {codiluent}') 
            ax[z,1].set_title(f"{model} ({float(reactorPressure/ct.one_atm)}atm, {int(reactorTemperature)}K, {H2Percent}% H2/{O2Percent}% O2/{dilution*codiluentPercent}% {codiluent}/{dilution*(1-codiluentPercent)}% Ar)")
            ax[z,0].tick_params(axis='both',direction='in')
            ax[z,1].tick_params(axis='both',direction='in')
            ax[z,2].tick_params(axis='both',direction='in')
            ax[z,0].set_ylabel(r'$\Delta$ T [K]')
            ax[z,1].set_ylabel('O$_2$ mole fraction [%]')
            ax[z,2].set_ylabel('H$_2$ mole fraction [%]')
        ax[len(reactorPressureList)-1,2].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
        ax[len(reactorPressureList)-1,1].set_xlabel('Temperature [K]')
        path=f'USSCI/figures/'+args.date+'/JSR'
        os.makedirs(path,exist_ok=True)
        plt.savefig(path+f'/{model}__{codiluent}_Pdependence.png', dpi=500, bbox_inches='tight')