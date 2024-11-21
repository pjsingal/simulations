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
from joblib import Parallel, delayed

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

#######################################################################
# Input Parameters
#######################################################################

models = {
    'Stagni-2020': {
        'submodels': {
            # 'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
            'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
            'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
                    },
        'fuel': 'NH3',
        'oxidizer':'O2:1.0, N2:3.76',
        'phi_list':[0.5, 2],
        'QOI':['NH3','O2'],
        'tau': 50e-3,
        'V': 85e-6, #[m3]
        'P_list':[20,40],
        'T_range': np.linspace(1000,1500,gridsz),
        'xlim':[[1100,1600],[1100,1600],[1100,1600],[1100,1600]],
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


length = 1.5e-7  # *approximate* PFR length [m]
u_0 = .006  # inflow velocity [m/s]
area = 1.e-4  # cross-sectional area [m**2]
n_steps = 2000
lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3


def getXvsT(gas,model,T,P,phi,species):
    oxidizer=vals['oxidizer']
    tau=vals['tau']
    fuel=vals['fuel']
    gas.TP = T, P
    gas.set_equivalence_ratio(phi,fuel,oxidizer,basis='mole')
    mass_flow_rate = u_0 * gas.density * area
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    t_total = length / u_0
    dt = t_total / n_steps
    t = (np.arange(n_steps) + 1) * dt #time vector
    z = np.zeros_like(t) #spatial vector
    y = np.zeros_like(t) #vector of [unsure]
    states = ct.SolutionArray(r.thermo)
    for n, t_i in enumerate(t):
        sim.advance(t_i)
        y[n] = mass_flow_rate / area / r.thermo.density
        z[n] = z[n - 1] + y[n] * dt
        states.append(r.thermo.state)
    ## time history of species X='H2', for example
    # timeHistory=t
    # states.X[:, gas.species_index('H2')]
    # find the point where t equals the residence time
    idx = np.argmin(np.abs(t-tau))
    return states.X[:, gas.species_index(species)][idx]

for model in models:
    print(f'Model: {model}')
    vals=models[model]
    T_list=vals['T_range']
    f, ax = plt.subplots(len(vals['P_list']), len(vals['QOI']), figsize=(args.figwidth,args.figheight))
    plt.subplots_adjust(wspace=0.3)
    counter=0
    f.suptitle(f'FR : {model} ({vals['fuel']}/air)', fontsize=10)
    for z, P in enumerate(vals['P_list']):
        print(f'Pressure: {P}atm')
        for w, QOI in enumerate(vals['QOI']):
            print(f'QOI: {QOI}')
            for k,m in enumerate(vals['submodels']):
                print(f'Submodel: {m}')
                gas = ct.Solution(list(vals['submodels'].values())[k])
                for i, phi in enumerate(vals['phi_list']):
                    print(r'$\phi$: '+f'{phi}')
                    X_history = Parallel(n_jobs=-1)(
                        delayed(getXvsT)(gas,vals,T,P,phi,QOI)
                        for T in T_list
                    )
                    ax[z,w].plot(T_list,X_history,color=colors[i], linestyle=lstyles[k], linewidth=lw, label=f'{m} '+r'$\phi$='+f'{phi}')
                    # dat = pd.read_csv(f'USSCI/graph-reading/{model}/FR/{P}bar_{phi}phi.csv',header=None)
                    # ax[z,w].semilogy(dat.iloc[:,0],dat.iloc[:,1],'o',fillstyle='none',linestyle='none',color=colors[k],markersize=msz,markeredgewidth=mw,label=r'$\phi$='+f'{phi}', zorder=120)
                ax[z,w].set_title(f"{QOI} ({P}atm)",fontsize=8)
                ax[z,w].tick_params(axis='both',direction='in')
                ax[len(vals['P_list'])-1,w].set_xlabel('Temperature [K]')
            ax[z,0].set_ylabel(f'mole fraction [-]')
            ax[z,w].set_xlim(vals['xlim'][z+w+counter])
        counter+=len(vals['QOI'])-1
        ax[len(vals['P_list'])-1,len(vals['QOI'])-1].legend(fontsize=lgdfsz,frameon=False,loc='best', handlelength=lgdw)
        path=f'USSCI/figures/'+args.date+'/FR'
        os.makedirs(path,exist_ok=True)
        f.savefig(path+f'/{model}.png', dpi=500, bbox_inches='tight')
        plt.close(f)
        print(f'Simulations have been stored in {path}\n')