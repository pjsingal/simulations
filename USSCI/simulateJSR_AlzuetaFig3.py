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
    # 'Stagni-2020': {
    #     # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
    #             },
    'Alzueta-2023': {
        # 'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
        'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
        'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
                },
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


# fuel='CH4'
T_list = np.linspace(900,1500,gridsz)
# oxidizer={'O2':1, 'N2': 3.76}

X_CO=206e-6
X_NH3=951e-6
X_O2=910e-6
X_HONO=1e-6
X_Ar=X_CO-X_NH3-X_O2-X_HONO

X = {'CO':X_CO,'NH3':X_NH3,'O2':X_O2,'HONO':X_HONO,'Ar':X_Ar}

# T_list = np.linspace(700,1400,gridsz)
# phi_list = [0.91,1.07,1.27]
P = 1 # [atm]
reactorTemperature = 1000
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k",'r']*3
########################################################################################

def getStirredReactor(gas,V,K,h,T):
    reactorRadius = (V*3/4/np.pi)**(1/3) # [m3]
    reactorSurfaceArea =4*np.pi*reactorRadius**2 # [m3]
    fuelAirMixtureTank = ct.Reservoir(gas)
    exhaust = ct.Reservoir(gas)
    env = ct.Reservoir(gas)
    reactor = ct.IdealGasReactor(gas, energy='on', volume=V)
    tau = 180/T
    ct.MassFlowController(upstream=fuelAirMixtureTank,
                          downstream=reactor,
                          mdot=reactor.mass/tau)
    ct.Valve(upstream=reactor,
             downstream=exhaust,
             K=K)
    ct.Wall(reactor, env, A=reactorSurfaceArea, U=h)
    return reactor

def getTemperatureDependence(gas,V,K,h,T_list,P,t_max):
    stirredReactor = getStirredReactor(gas,V,K,h,reactorTemperature)
    columnNames = (
        ['pressure'] +
        [stirredReactor.component_name(item)
         for item in range(stirredReactor.n_vars)]
    )
    tempDependence = pd.DataFrame(columns=columnNames)
    # concentrations = gas.X
    for T in T_list:
        gas.TP = T, P*ct.one_atm
        stirredReactor = getStirredReactor(gas,V,K,h,T)
        reactorNetwork = ct.ReactorNet([stirredReactor])
        # reactorNetwork.max_time_step=1e-4  # Set a smaller maximum time step
        # reactorNetwork.rtol = 1e-6             # Reduce relative tolerance
        # reactorNetwork.atol = 1e-12            # Reduce absolute tolerance
        t = 0
        while t < t_max:
            t = reactorNetwork.step()
        state = np.hstack([stirredReactor.thermo.P,
                        stirredReactor.mass,
                        stirredReactor.volume,
                        stirredReactor.T,
                        stirredReactor.thermo.X])
        tempDependence.loc[T] = state
        # concentrations = stirredReactor.thermo.X
    return tempDependence


for model in models:
    print(f'Model: {model}')
    f2, ax2 = plt.subplots(1, 1, figsize=(args.figwidth,args.figheight))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'JSR '+r'$\Delta$T [K]'+f': {model}', fontsize=10)
    for k,m in enumerate(models[model]):
        print(f'Submodel: {m}')
        gas = ct.Solution(list(models[model].values())[k])
        gas.TPX = reactorTemperature, P*ct.one_atm, X
        # tau = 3e-3 # from Bartok / Glarborg Fig. 7
        V = 0.000113 #30.5*(1e-2)**3 reactor volume [m3]
        h = 79.5 # heat transfer coefficient W/m2/K
        K = 0.01 # pressureValveCoefficient
        t_max = 50  # max simulation time [s]
        tempDependence = getTemperatureDependence(gas,V,K,h,T_list,P,t_max)
        ax2.plot(tempDependence.index,tempDependence['CO']*1e6/10, color=colors[0], linestyle=lstyles[k], linewidth=lw, label=f'CO/10')   
        ax2.plot(tempDependence.index,tempDependence['NH3']*1e6, color=colors[1], linestyle=lstyles[k], linewidth=lw, label=f'NH3')   
        ax2.plot(tempDependence.index,tempDependence['NO']*1e6, color=colors[2], linestyle=lstyles[k], linewidth=lw, label=f'NO')   
        ax2.plot(tempDependence.index,tempDependence['N2']*1e6, color=colors[3], linestyle=lstyles[k], linewidth=lw, label=f'N2')   
        ax2.set_title(f"{m} ({P}atm)",fontsize=8)
        ax2.tick_params(axis='both',direction='in')
        ax2.set_xlabel('Temperature [K]')
    ax2.set_ylabel('Mole fraction [ppm]')
    ax2.legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
    path=f'USSCI/figures/'+args.date+'/JSR'
    os.makedirs(path,exist_ok=True)
    f2.savefig(path+f'/{model}_AlzuetaFig3.png', dpi=500, bbox_inches='tight')
    plt.close(f2)
    print(f'Simulations have been stored in {path}\n')