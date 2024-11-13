from __future__ import division
from __future__ import print_function
import os
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib as mpl
import numpy as np

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

import matplotlib.ticker as ticker

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
# figsize=(3.8,2)
f, ax = plt.subplots(1, 1, figsize=(args.figwidth, args.figheight)) 
name = 'ShockTubeSpeciesProfile_H2O' #os.path.splitext(os.path.basename(__file__))[0]


model = 'aramco30'
refSpecies='H2O'
X = {'C4H6':0.01, 'O2': 0.11, 'AR': 0.88}
def plotXvsTime(fname,pltlabel,pltcolour,lstyle='solid',zorder_value=10):
    
    # gas = ct.Solution('test/data/Burke_H2_ArBath.yaml')
    gas = ct.Solution(fname)
    gas.TPX = 1196, 2.127*101325, X
    r = ct.Reactor(contents=gas,energy="on")
    reactorNetwork = ct.ReactorNet([r]) # this will be the only reactor in the network
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    estIgnitDelay = 0.1
    t = 0
    counter = 1
    while t < estIgnitDelay:
        t = reactorNetwork.step()
        if counter % 10 == 0:
            timeHistory.append(r.thermo.state, t=t)
        counter += 1
    tConv = 1e6 #time conversion factor (1e6 converts to microseconds)
    timeShift=0 # [seconds]
    shiftedTime = tConv*(timeHistory.t - timeShift)
    moleFrac = timeHistory(refSpecies).X 
    ax.plot(shiftedTime, moleFrac*100, color=pltcolour,label=pltlabel,linestyle=lstyle,linewidth=lw,zorder=zorder_value)
    
print(f'Generating plot for {model}...')
plotXvsTime(f'USSCI/factory_mechanisms/{args.date}/{model}_LMRR.yaml',f'{model}-LMRR',"r",lstyle="dotted",zorder_value=70)
print(f'{model}-LMRR added to plot!')
plotXvsTime(f'USSCI/factory_mechanisms/{args.date}/{model}_LMRR_allP.yaml',f'{model}-LMRR-allP',"r",lstyle="solid",zorder_value=80)
print(f'{model}-LMRR-allP added to plot!')
# plotXvsTime(f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',"Alzueta-LMRR","b",lstyle="solid",zorder_value=70)
# plotXvsTime(f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',"Alzueta-LMRR-allP","b",lstyle="solid",zorder_value=80)
    
ax.legend(fontsize=lgdfsz,handlelength=lgdw, frameon=False, loc='lower right')  
ax.set_ylabel(r'$\rm H_2O$ mole fraction [%]')
ax.set_xlabel(r'Time [$\mathdefault{\mu s}$]')
ax.tick_params(axis='both', direction="in")#, labelsize=7)
ax.set_xlim([0.0001,299.999])
ax.set_ylim([2.25,2.75])

# import matplotlib.ticker as ticker
# ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
# ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
# ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

X_str = "_".join(f"{key}{value}" for key, value in X.items())
path=f'USSCI/figures/{args.date}/shock-tube'
os.makedirs(path,exist_ok=True)
plt.savefig(path+f'/{model}_{X_str}.png', dpi=500, bbox_inches='tight')
print(f'Plot saved to {path}/{model}_{X_str}.png')