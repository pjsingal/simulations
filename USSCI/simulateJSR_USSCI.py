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
    'Stagni-2020': {
        # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
                },
    # 'Alzueta-2023': {
    #     # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #     'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
    #     'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
    #             },
    # 'Glarborg-2018': {
    #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
    #             },
    # 'Aramco-3.0': {
    #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
    #             },
}

# fuels = {
#     'Stagni-2020':['H2','NH3'],
#     # 'Alzueta-2023': ['H2','NH3'],
#     # 'Glarborg-2018':['H2','NH3'],
#     # 'Aramco-3.0':['C2H2','CH3OH','C4H10']
# }

models = {
    'Stagni-2020': {
        'submodels': {
            # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
                    },
        'fuels': ['H2','NH3'],
        'oxidizer':'O2',
        'diluent':'He:1',
        'fraction':'diluent:0.9795',
        'phi_list':[0.01, 0.08],
        'QOI':['NO','NH3'],
        'tau': 1.5, # from Bartok / Glarborg Fig. 7
        'P_list':[800/750,10],
        'T_range': np.linspace(800,1210,gridsz),
        'xlim':[[600,1210],[600,1210],[600,1210],[600,1210]],
        't_max':1,
    },
    # 'Alzueta-2023': {
    #     'submodels': {
    #         # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    #         'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
    #         'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
    #                 },
    #     'fuels': ['H2','NH3'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'T_range':[[600,2500],
                #    [600,2500]],
    #     't_max':23200e-6,
    #     'xlim': [[60,160], [12500,23000],[60,160], [5000,9500]]
    # },
    # 'Glarborg-2018': {
    #     'submodels': {
    #         # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/glarborg-2018_LMRR_allP.yaml",
    #                 },
    #     'fuels': ['H2','NH3'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'T_range':[[600,2500],
                #    [600,2500]],
    #     't_max':9100e-6,
    #     'xlim': [[60,200], [6000,9000],[60,200], [3400,4800]]
    # },
    # 'Aramco-3.0': {
    #     'submodels': {
    #         # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #         'LMRR': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR.yaml",
    #         'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/aramco30_LMRR_allP.yaml",
    #                 },
    #     'fuels': ['C2H2','CH3OH','C4H10'],
    #     'oxidizer':'O2:1.0, N2:3.76',
    #     'phi_list':[0.5, 2],
    #     'P_list':[20,40],
    #     'T_range':[[600,2500],
                #    [600,2500]],
    #     't_max':410e-6,
    #     'xlim': [[-10,200], [-10,180],[100,380], [0,300], [0,100], [0,220]],
    # }
}

reactorTemperature = 1000
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3

########################################################################################

def getStirredReactor(gas,tau):
    V = 161e-6 # from Bartok
    h = 79.5 # heat transfer coefficient W/m2/K
    K = 0.01 # pressureValveCoefficient
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

def getTemperatureDependence(gas,T_list,P,phi,oxidizer,fuel,t_max,tau,diluent=None,fraction=None):
    gas.TP = reactorTemperature, P*ct.one_atm
    if diluent:
        gas.set_equivalence_ratio(phi,fuel,oxidizer,diluent=diluent,fraction=fraction,basis='mole')
    else:
        gas.set_equivalence_ratio(phi,fuel,oxidizer,basis='mole')
    stirredReactor = getStirredReactor(gas,tau)
    columnNames = (
        ['pressure'] +
        [stirredReactor.component_name(item)
         for item in range(stirredReactor.n_vars)]
    )
    tempDependence = pd.DataFrame(columns=columnNames)
    for T in T_list:
        gas.TP = T, P*ct.one_atm
        gas.set_equivalence_ratio(phi,fuel,oxidizer,basis='mole')
        stirredReactor = getStirredReactor(gas,tau)
        reactorNetwork = ct.ReactorNet([stirredReactor])
        # reactorNetwork.max_time_step=1e-4  # Set a smaller maximum time step
        # reactorNetwork.rtol = 1e-6             # Reduce relative tolerance
        # reactorNetwork.atol = 1e-12            # Reduce absolute tolerance
        t = 0
        counter=1
        while t < t_max:
            t = reactorNetwork.step()
            if not counter % 50:
                state = np.hstack([
                        stirredReactor.thermo.P,
                        stirredReactor.mass,
                        stirredReactor.volume,
                        stirredReactor.T,
                        stirredReactor.thermo.X,
                        ])
            counter+=1
        tempDependence.loc[T] = state
    return tempDependence


for model in models:
    print(f'Model: {model}')
    f1, ax1 = plt.subplots(len(models[model]['P_list']), len(models[model]['fuels']), figsize=(args.figwidth,args.figheight))
    f2, ax2 = plt.subplots(len(models[model]['P_list']), len(models[model]['fuels']), figsize=(args.figwidth,args.figheight))
    f3, ax3 = plt.subplots(len(models[model]['P_list']), len(models[model]['fuels']), figsize=(args.figwidth,args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    QOI1=models[model]['QOI'][0]
    QOI2=models[model]['QOI'][1]
    f1.suptitle(f'JSR '+r'$\Delta$T [K]'+f': {model}', fontsize=10)
    f2.suptitle(f'JSR '+f'X$_{QOI1}$ [%]'+f': {model}', fontsize=10)
    f3.suptitle(f'JSR '+f'X$_{QOI2}$ [%]'+f': {model}', fontsize=10)
    counter=0
    for z, P in enumerate(models[model]['P_list']):
        print(f'Pressure: {P}atm')
        for w, fuel in enumerate(models[model]['fuels']):
            print(f'Fuel: {fuel}')
            # T_list = np.linspace(models[model]['T_range'][z+w+counter][0],models[model]['T_range'][z+w+counter][1],gridsz)
            T_list=models[model]['T_range']
            for k,m in enumerate(models[model]['submodels']):
                print(f'Submodel: {m}')
                gas = ct.Solution(list(models[model]['submodels'].values())[k])
                for i, phi in enumerate(models[model]['phi_list']):
                    print(r'$\phi$: '+f'{phi}')
                    if models[model].get('diluent') is not None:
                        tempDependence = getTemperatureDependence(gas,T_list,P,phi,models[model]['oxidizer'],fuel,models[model]['t_max'],models[model]['tau'],diluent=models[model]['diluent'],fraction=models[model]['fraction'])
                    else:
                        tempDependence = getTemperatureDependence(gas,T_list,P,phi,models[model]['oxidizer'],fuel,models[model]['t_max'],models[model]['tau'])
                    ax1[z,w].plot(tempDependence.index,np.subtract(tempDependence['temperature'],tempDependence.index),color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')   
                    ax2[z,w].plot(tempDependence.index,tempDependence[QOI1]*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')   
                    ax3[z,w].plot(tempDependence.index,tempDependence[QOI2]*100, color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')
                ax1[z,w].set_title(f"{fuel}/air ({P}atm)",fontsize=8)
                ax1[z,w].tick_params(axis='both',direction='in')
                ax1[len(models[model]['P_list'])-1,w].set_xlabel('Temperature [K]')
                ax2[z,w].set_title(f"{fuel}/air ({P}atm)",fontsize=8)
                ax2[z,w].tick_params(axis='both',direction='in')
                ax2[len(models[model]['P_list'])-1,w].set_xlabel('Temperature [K]')
                ax3[z,w].set_title(f"{fuel}/air ({P}atm)",fontsize=8)
                ax3[z,w].tick_params(axis='both',direction='in')
                ax3[len(models[model]['P_list'])-1,w].set_xlabel('Temperature [K]')
            ax1[z,0].set_ylabel(r'$\Delta$ T [K]')
            ax1[z,w].set_xlim(models[model]['xlim'][z+w+counter])
            ax2[z,0].set_ylabel(f'{QOI1} mole fraction [%]')
            ax2[z,w].set_xlim(models[model]['xlim'][z+w+counter])
            ax3[z,0].set_ylabel(f'{QOI2} mole fraction [%]')
            ax3[z,w].set_xlim(models[model]['xlim'][z+w+counter])
        counter+=len(models[model]['fuels'])-1
    ax1[len(models[model]['P_list'])-1,len(models[model]['fuels'])-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
    ax2[len(models[model]['P_list'])-1,len(models[model]['fuels'])-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
    ax3[len(models[model]['P_list'])-1,len(models[model]['fuels'])-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
    path=f'USSCI/figures/'+args.date+'/JSR'
    os.makedirs(path,exist_ok=True)
    f1.savefig(path+f'/{model}_deltaT.png', dpi=500, bbox_inches='tight')
    f2.savefig(path+f'/{model}_XO2.png', dpi=500, bbox_inches='tight')
    f3.savefig(path+f'/{model}_XH2.png', dpi=500, bbox_inches='tight')
    plt.close(f1)
    plt.close(f2)
    plt.close(f3)
    print(f'Simulations have been stored in {path}\n')