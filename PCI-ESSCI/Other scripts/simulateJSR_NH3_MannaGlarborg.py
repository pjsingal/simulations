#%%
from __future__ import division
from __future__ import print_function
import sys, os
sys.path.append("C:/Users/pjsin/Documents/cantera/build/python")
import pandas as pd
import numpy as np
import time
import cantera as ct
import os.path
from os import path
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

args = parser.parse_args()
lw=args.lw
mw=args.mw
msz=args.msz
dpi=args.dpi
lgdw=args.lgdw
lgdfsz=args.lgdfsz
gridsz=args.gridsz

mpl.rc('font',family='Times New Roman')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = args.fsz
mpl.rcParams['xtick.labelsize'] = args.fszxtick
mpl.rcParams['ytick.labelsize'] = args.fszytick
from matplotlib.legend_handler import HandlerTuple
plt.rcParams['axes.labelsize'] = args.fszaxlab
# mpl.rcParams['xtick.major.width'] = 0.5  # Width of major ticks on x-axis
# mpl.rcParams['ytick.major.width'] = 0.5  # Width of major ticks on y-axis
# mpl.rcParams['xtick.minor.width'] = 0.5  # Width of minor ticks on x-axis
# mpl.rcParams['ytick.minor.width'] = 0.5  # Width of minor ticks on y-axis
# mpl.rcParams['xtick.major.size'] = 2.5  # Length of major ticks on x-axis
# mpl.rcParams['ytick.major.size'] = 2.5  # Length of major ticks on y-axis
# mpl.rcParams['xtick.minor.size'] = 1.5  # Length of minor ticks on x-axis
# mpl.rcParams['ytick.minor.size'] = 1.5  # Length of minor ticks on y-axis

save_plots = True
# figsize=(3.5,7)
f, ax = plt.subplots(1,1, figsize=(args.figwidth, args.figheight)) 

import matplotlib.ticker as ticker
# plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.3)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
# ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
# ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
# ax.annotate('(d)', xy=(0.95, 0.95), xycoords='axes fraction',ha='right', va='top')
# ax[1].annotate('(e)', xy=(0.95, 0.95), xycoords='axes fraction',ha='right', va='top')
# ax[2].annotate('(f)', xy=(0.95, 0.95), xycoords='axes fraction',ha='right', va='top')
# lw=0.7
# mw=0.5
# msz=3.5
# dpi=1000
# lgdw=1
# lgdfsz=6
# gridsz=50
# lw=0.7
# mw=0.5
# msz=3.5
# dpi=1000
# lgdw=0.6
# lgdfsz=7
# gridsz=50

# name = 'JSR_NH3_V1'
# colors = ['r','b']
# models = {
#           'Ar':"test/data/alzuetamechanism_LMRR_allAR.yaml",
#           r'H$_2$O':"test/data/alzuetamechanism_LMRR_allH2O.yaml",           
#           }
# name = 'JSR_NH3_V2'
# colors = ['r','b',"xkcd:grey"]
# models = {
#           'Ar':"test/data/alzuetamechanism_LMRR_allAR.yaml",
#           r'H$_2$O':"test/data/alzuetamechanism_LMRR_allH2O.yaml",
#           'Alzueta':"test/data/alzuetamechanism.yaml",            
#           }
name = 'JSR_NH3_MannaGlarborg'
colors = ['r','b','xkcd:purple']
models = {
          r'This study (Glarborg)':"test\\data\\glarborg-2018.yaml",
          }
T_list = np.linspace(900,2000,gridsz)
P = 1.2
tau = 0.25

# NH3_con_list = [0.02, 0.05, 0.075, 0.10]
# NH3_con_list = [0, 0.05, 0.10, 0.15, 0.20]
diluent = 0.92
# NH3percent_list = [0, 0.02, 0.05, 0.075, 0.10]
NH3percent_list = [0.10]


lines =['-','--','-','-','-']


##############################################################################################################################

reactorTemperature = 1000  # Kelvin
reactorPressure = P*ct.one_atm  # in atm. This equals 1.06 bars
residenceTime = tau  # s
reactorVolume = 0.000113 #30.5*(1e-2)**3  # m3
reactorRadius = np.cbrt(reactorVolume*3/4*np.pi) # m3
reactorSurfaceArea = 4*np.pi*np.square(reactorRadius) # m3
pressureValveCoefficient = 0.01
maxPressureRiseAllowed = 0.01
maxSimulationTime = 50  # seconds
heatTransferCoefficient = 7949.6
heatTransferCoefficient = 7.9496*2.2
tempDependence = []

##############################################################################################################################


path="G:\\Mon disque\\Columbia\\Burke Lab\\01 Mixture Rules Project\\Graph Reading\\8 SP H2O X vs T (JSR) (Sabia)\\"
        
H2_10_data = pd.read_csv(path+'Fig9_GlarborgModel.csv') 
ax.plot(H2_10_data.iloc[:, 0],H2_10_data.iloc[:, 1],marker='o',fillstyle='none',zorder=2,linestyle='none',color='k',markersize=msz,markeredgewidth=mw, label="Graph-read model (Glarborg)")
ax.set_title("Fig. 9, Manna/Sabia/de Joannon (2020)")

for k,m in enumerate(models):
    for i, NH3percent in enumerate(NH3percent_list):
        Ar = diluent
        NH3 = (1-diluent)
        reactants = {'AR': Ar, 'NH3':NH3}    
        
        # for j, T in enumerate(T_list):
        
        concentrations = reactants
        gas = ct.Solution(list(models.values())[k])
        gas.TPX = reactorTemperature, reactorPressure, concentrations 
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
        # reactorNetwork.rtol = 1.0e-6
        # reactorNetwork.atol = 1.0e-15
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
        
        inletConcentrations = concentrations
        
        for j,T in enumerate(T_list): #temperature in T:
            reactorTemperature = T #temperature  # Kelvin
            gas.TPX = reactorTemperature, reactorPressure, inletConcentrations
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

        
        # if i == 0:
            
        #     chopped_range = [860,925]
            
        #     chopped_index = [abs(np.subtract(tempDependence[i].index,chopped_range[0])).argmin(), abs(np.subtract(tempDependence[i].index,chopped_range[1])).argmin()]
        #     ax.plot(list(tempDependence[i].index)[:chopped_index[0]], list(np.subtract(tempDependence[i]['temperature'],tempDependence[i].index))[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
        #     ax[1].plot(list(tempDependence[i].index)[:chopped_index[0]], list(tempDependence[i]['O2']*100)[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
        #     ax[2].plot(list(tempDependence[i].index)[:chopped_index[0]], list(tempDependence[i]['H2']*100)[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
            
        #     ax.plot(list(tempDependence[i].index)[chopped_index[1]:], list(np.subtract(tempDependence[i]['temperature'],tempDependence[i].index))[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        #     ax[1].plot(list(tempDependence[i].index)[chopped_index[1]:], list(tempDependence[i]['O2']*100)[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        #     ax[2].plot(list(tempDependence[i].index)[chopped_index[1]:], list(tempDependence[i]['H2']*100)[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        # elif i == 1:
            
        #     chopped_range = [900,950]
            
        #     chopped_index = [abs(np.subtract(tempDependence[i].index,chopped_range[0])).argmin(), abs(np.subtract(tempDependence[i].index,chopped_range[1])).argmin()]
    
        #     ax.plot(list(tempDependence[i].index)[:chopped_index[0]], list(np.subtract(tempDependence[i]['temperature'],tempDependence[i].index))[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
        #     ax[1].plot(list(tempDependence[i].index)[:chopped_index[0]], list(tempDependence[i]['O2']*100)[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
        #     ax[2].plot(list(tempDependence[i].index)[:chopped_index[0]], list(tempDependence[i]['H2']*100)[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
            
        #     ax.plot(list(tempDependence[i].index)[chopped_index[1]:], list(np.subtract(tempDependence[i]['temperature'],tempDependence[i].index))[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        #     ax[1].plot(list(tempDependence[i].index)[chopped_index[1]:], list(tempDependence[i]['O2']*100)[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        #     ax[2].plot(list(tempDependence[i].index)[chopped_index[1]:], list(tempDependence[i]['H2']*100)[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
                
        # elif i == 2:
            
        #     chopped_range = [825,975]        
        #     chopped_index = [abs(np.subtract(tempDependence[i].index,chopped_range[0])).argmin(), abs(np.subtract(tempDependence[i].index,chopped_range[1])).argmin()]

        #     ax.plot(list(tempDependence[i].index)[:chopped_index[0]], list(np.subtract(tempDependence[i]['temperature'],tempDependence[i].index))[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
        #     ax[1].plot(list(tempDependence[i].index)[:chopped_index[0]], list(tempDependence[i]['O2']*100)[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
        #     ax[2].plot(list(tempDependence[i].index)[:chopped_index[0]], list(tempDependence[i]['H2']*100)[:chopped_index[0]], color=colors[i], linestyle=lines[i], label=str(NH3percent*100)+'% H$_2$O')   
            
        #     ax.plot(list(tempDependence[i].index)[chopped_index[1]:], list(np.subtract(tempDependence[i]['temperature'],tempDependence[i].index))[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        #     ax[1].plot(list(tempDependence[i].index)[chopped_index[1]:], list(tempDependence[i]['O2']*100)[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
        #     ax[2].plot(list(tempDependence[i].index)[chopped_index[1]:], list(tempDependence[i]['H2']*100)[chopped_index[1]:], color=colors[i], linestyle=lines[i])   
                                            
        # else:
        # ax.plot(tempDependence[i].index, np.subtract(tempDependence[i]['temperature'],tempDependence[i].index), color=colors[k], linestyle='solid',linewidth=lw, label=m) 
        ax.plot(tempDependence[i].index, tempDependence[i]['H2']*100, color=colors[k], linestyle='solid',linewidth=lw, label=m) 

            
     
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('H$_2$ concentration [%]')
ax.tick_params(axis='both',direction='in')
ax.legend(fontsize=lgdfsz,frameon=False,loc='upper left', handlelength=lgdw)

# ax.set_xlim([780,1070])

if save_plots == True:
    plt.savefig('burkelab_SimScripts/figures/'+name+'.pdf', dpi=500, bbox_inches='tight')
    plt.savefig('burkelab_SimScripts/figures/'+name+'.png', dpi=500, bbox_inches='tight')
# plt.show()     