import sys, os
sys.path.append("C:/Users/pjsin/Documents/cantera/build/python")
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd 
import time
import numpy as np
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
mpl.rcParams['xtick.major.width'] = 0.5  # Width of major ticks on x-axis
mpl.rcParams['ytick.major.width'] = 0.5  # Width of major ticks on y-axis
mpl.rcParams['xtick.minor.width'] = 0.5  # Width of minor ticks on x-axis
mpl.rcParams['ytick.minor.width'] = 0.5  # Width of minor ticks on y-axis
mpl.rcParams['xtick.major.size'] = 2.5  # Length of major ticks on x-axis
mpl.rcParams['ytick.major.size'] = 2.5  # Length of major ticks on y-axis
mpl.rcParams['xtick.minor.size'] = 1.5  # Length of minor ticks on x-axis
mpl.rcParams['ytick.minor.size'] = 1.5  # Length of minor ticks on y-axis

f, ax = plt.subplots(1, 4, figsize=(args.figwidth, args.figheight)) 

# figsize = (6.5, 1.5) # (width, height)
# lw=0.7
# mw=0.5
# msz=2.5
# dpi=1000
# lgdw=0.6
# lgdfsz=5
# gridsz=10

save_plots = True

import matplotlib.ticker as ticker
plt.subplots_adjust(wspace=0.25)
ax[0].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
ax[1].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
ax[2].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
ax[3].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))

ax[0].xaxis.set_major_locator(ticker.MultipleLocator(50))
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(50))
ax[2].xaxis.set_major_locator(ticker.MultipleLocator(50))
ax[3].xaxis.set_major_locator(ticker.MultipleLocator(50))

ax[0].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0e}"))
ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
ax[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
ax[2].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
ax[2].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
ax[3].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
ax[3].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))

f.text(0.5, 0, r'Temperature [K]', ha='center', va='center',fontsize=args.fszaxlab)

name = 'IDT_shao'
path=os.getcwd()

# models = {    
#           'Alzueta':"test/data/alzuetamechanism.yaml",            
#           'Ar':"test/data/alzuetamechanism_LMRR_allAR.yaml",
#           r'H$_2$O':"test/data/alzuetamechanism_LMRR_allH2O.yaml",
#           'LMR-R':"test/data/alzuetamechanism_LMRR.yaml", 
#           }
models = {    
          'Alzueta':"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism.yaml",  
          r"$\epsilon_{0,NH_3}(300K)$":"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism_epsNH3_T=300K.yaml",  
          r"$\epsilon_{0,NH_3}(2000K)$":"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism_epsNH3_T=2000K.yaml",            
          'Ar':"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism_LMRR_allAR.yaml",
          r'H$_2$O':"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism_LMRR_allH2O.yaml",
        #   'LMR-R':"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism_LMRR.yaml",  
          'LMR-R':"C:\\Users\\pjsin\\Documents\\cantera\\test\\data\\alzuetamechanism_LMRR_extraColliders.yaml",
          }
colors = ["xkcd:grey", "xkcd:teal", "orange", 'r', 'b', 'xkcd:purple']
# colors = ['r', 'b', 'xkcd:purple']
# colors = ["xkcd:grey", "orange", 'r', 'b']
lstyles = ["solid","solid","solid","solid","solid","solid","solid","solid"]
# lstyles = [(7.5, (1, 10)),(0, (5, 5)),(0, (3, 1, 1, 1, 1, 1)),"solid","solid","solid"]
# lstyles = ["solid",(7.5, (5, 10)),(0, (5, 10)),"solid","solid","solid"]


def ignitionDelay(states, species):
    # i_ign = states(species).Y.argmax()
    i_ign = np.gradient(states(species).Y.T[0]).argmax()
    return states.t[i_ign]

################################################################################################

path='G:\\Mon disque\\Columbia\\Burke Lab\\01 Mixture Rules Project\\Graph Reading\\'
df = pd.read_csv(path+'Shao_IDT\\1.csv')
p_df = df['P']
T_df = df['T']
IDT_df = df['IDT']
ax[0].semilogy(T_df,IDT_df,'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Shao et al.', zorder=120)
T_list = np.linspace(1100,1300,gridsz)#[::-1]
for k, m in enumerate(models):
    estimatedIgnitionDelayTimes = np.ones(len(T_list))
    estimatedIgnitionDelayTimes[:] = 0.05
    ignitionDelays_RG = np.zeros(len(T_list))
    for j, T in enumerate(T_list):
        gas = ct.Solution(list(models.values())[k])
        gas.TPX = T, 17*ct.one_atm, {'H2':0.03, 'O2':0.015, 'Ar':1-0.03-0.015}
        r = ct.Reactor(contents=gas)
        reactorNetwork = ct.ReactorNet([r])
        timeHistory = ct.SolutionArray(gas, extra=['t'])
        t0 = time.time()
        t = 0
        counter = 1
        while t < estimatedIgnitionDelayTimes[j]:
            t = reactorNetwork.step()
            if counter % 1 == 0:
                timeHistory.append(r.thermo.state, t=t)
            counter += 1
        tau = ignitionDelay(timeHistory, 'oh')
        t1 = time.time()
        ignitionDelays_RG[j] = tau
    if colors[k] == 'xkcd:purple':
        zorder_value = 100
    elif colors[k] == 'r' or colors[k] == 'b':
        zorder_value = 95
    elif colors[k] == "xkcd:grey":
        zorder_value = 94
    else:   
        zorder_value = k  # Default z-order for other lines
    ax[0].semilogy(T_list, 1e6*ignitionDelays_RG, '-', linestyle=lstyles[k], linewidth=lw, color=colors[k], label=m, zorder=zorder_value)
    

# ax[0].legend(fontsize=lgdfsz, frameon=False, loc='upper right',handlelength=lgdw)
ax[0].set_ylabel(r'Ignition delay [$\mathdefault{\mu s}$]')
# ax[0, 0].set_xlabel(r'Temperature [K]', fontsize=18)
ax[0].tick_params(axis='both', direction="in")
ax[0].tick_params(axis='both', which='minor', direction="in")#, bottom=False)
ax[0].annotate('(a)', xy=(0.2, 0.1), xycoords='axes fraction', ha='right', va='top')

################################################################################################

df = pd.read_csv(path+'\\Shao_IDT\\2.csv')
p_df = df['P']
T_df = df['T']
IDT_df = df['IDT']
T_list = np.linspace(1100,1300,gridsz)#[::-1]
ax[1].semilogy(T_df,IDT_df,'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Shao et al.', zorder=120)
for k, m in enumerate(models):
    estimatedIgnitionDelayTimes = np.ones(len(T_list))
    estimatedIgnitionDelayTimes[:] = 0.05
    ignitionDelays_RG = np.zeros(len(T_list))
    for j, T in enumerate(T_list):
        gas = ct.Solution(list(models.values())[k])
        gas.TPX = T, 13*ct.one_atm, {'H2':0.03, 'O2':0.015, 'N2':1-0.03-0.015}
        r = ct.Reactor(contents=gas)
        reactorNetwork = ct.ReactorNet([r])
        timeHistory = ct.SolutionArray(gas, extra=['t'])
        t0 = time.time()
        t = 0
        counter = 1
        while t < estimatedIgnitionDelayTimes[j]:
            t = reactorNetwork.step()
            if counter % 1 == 0:
                timeHistory.append(r.thermo.state, t=t)
            counter += 1
        tau = ignitionDelay(timeHistory, 'oh')
        t1 = time.time()
        ignitionDelays_RG[j] = tau
    # if lstyles[k] != "solid":
    #     zorder_value = 100
    # elif lstyles[k] == "solid":
    #     zorder_value=1
    if colors[k] == 'xkcd:purple':
        zorder_value = 100
    elif colors[k] == 'r' or colors[k] == 'b':
        zorder_value = 95
    elif colors[k] == "xkcd:grey":
        zorder_value = 94
    else:   
        zorder_value = k  # Default z-order for other lines
    ax[1].semilogy(T_list, 1e6*ignitionDelays_RG, '-', linestyle=lstyles[k],linewidth=lw, color=colors[k], label=m, zorder=zorder_value)
    

# ax[1].legend(fontsize=lgdfsz, frameon=False, loc='upper right',handlelength=lgdw) 
# ax[0,1].set_ylabel(r'Ignition delay [$\mathdefault{\mu s}$]', fontsize=18)
# ax[0,1].set_xlabel(r'Temperature [K]', fontsize=18)
# ax[0,1].legend(fontsize=10, frameon=False, loc='upper right')  
ax[1].tick_params(axis='both', direction="in")
ax[1].tick_params(axis='both', which='minor', direction="in")#, bottom=False)
ax[1].annotate('(b)', xy=(0.2, 0.1), xycoords='axes fraction', ha='right', va='top')

################################################################################################

df = pd.read_csv(path+'\\Shao_IDT\\3.csv')
p_df = df['P']
T_df = df['T']
IDT_df = df['IDT']
H2O_df = df['H2O']
T_list = np.linspace(1200,1400,gridsz)#[::-1]
ax[2].semilogy(T_df,IDT_df,'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Shao et al.', zorder=120)
for k, m in enumerate(models):
    estimatedIgnitionDelayTimes = np.ones(len(T_list))
    estimatedIgnitionDelayTimes[:] = 0.05
    ignitionDelays_RG = np.zeros(len(T_list))
    for j, T in enumerate(T_list):
        gas = ct.Solution(list(models.values())[k])
        gas.TPX = T, 15*ct.one_atm, {'H2':0.03, 'O2':0.015, 'H2O':0.09, 'Ar':1-0.03-0.015-0.09}
        r = ct.Reactor(contents=gas)
        reactorNetwork = ct.ReactorNet([r])
        timeHistory = ct.SolutionArray(gas, extra=['t'])
        t0 = time.time()
        t = 0
        counter = 1
        while t < estimatedIgnitionDelayTimes[j]:
            t = reactorNetwork.step()
            if counter % 1 == 0:
                timeHistory.append(r.thermo.state, t=t)
            counter += 1
        tau = ignitionDelay(timeHistory, 'oh')
        t1 = time.time()
        ignitionDelays_RG[j] = tau

    # if lstyles[k] != "solid":
    #     zorder_value = 100
    # elif lstyles[k] == "solid":
    #     zorder_value=1
    if colors[k] == 'xkcd:purple':
        zorder_value = 100
    elif colors[k] == 'r' or colors[k] == 'b':
        zorder_value = 95
    elif colors[k] == "xkcd:grey":
        zorder_value = 94
    else:   
        zorder_value = k  # Default z-order for other lines
    ax[2].semilogy(T_list, 1e6*ignitionDelays_RG, '-', linestyle=lstyles[k],linewidth=lw, color=colors[k], label=m, zorder=zorder_value)

# ax[2].legend(fontsize=lgdfsz, frameon=False, loc='upper right',handlelength=lgdw)
# ax[2].set_ylabel(r'Ignition delay [$\mathdefault{\mu s}$]')
# ax[2].set_xlabel(r'Temperature [K]')
ax[2].tick_params(axis='both', direction="in")
ax[2].tick_params(axis='both', which='minor', direction="in")#, bottom=False)
ax[2].annotate('(c)', xy=(0.2, 0.1), xycoords='axes fraction',ha='right', va='top')

################################################################################################

df = pd.read_csv(path+'\\Shao_IDT\\4.csv')
p_df = df['P']
T_df = df['T']
IDT_df = df['IDT']
T_list = np.linspace(1100,1300,gridsz)#[::-1]
ax[3].semilogy(T_df,IDT_df,'o',fillstyle='none',linestyle='none',color='k',markersize=msz,markeredgewidth=mw,label='Shao et al.', zorder=120)
for k, m in enumerate(models):
    estimatedIgnitionDelayTimes = np.ones(len(T_list))
    estimatedIgnitionDelayTimes[:] = 0.05
    ignitionDelays_RG = np.zeros(len(T_list))
    for j, T in enumerate(T_list):
        gas = ct.Solution(list(models.values())[k])
        gas.TPX = T, 12*ct.one_atm, {'H2':0.03, 'O2':0.015, 'CO2':0.20, 'Ar':1-0.2-0.03-0.015}
        r = ct.Reactor(contents=gas)
        reactorNetwork = ct.ReactorNet([r])
        timeHistory = ct.SolutionArray(gas, extra=['t'])
        t0 = time.time()
        t = 0
        counter = 1
        while t < estimatedIgnitionDelayTimes[j]:
            t = reactorNetwork.step()
            if counter % 1 == 0:
                timeHistory.append(r.thermo.state, t=t)
            counter += 1
        tau = ignitionDelay(timeHistory, 'oh')
        t1 = time.time()
        ignitionDelays_RG[j] = tau
    # if lstyles[k] != "solid":
    #     zorder_value = 100
    # elif lstyles[k] == "solid":
    #     zorder_value=1
    if colors[k] == 'xkcd:purple':
        zorder_value = 100
    elif colors[k] == 'r' or colors[k] == 'b':
        zorder_value = 95
    elif colors[k] == "xkcd:grey":
        zorder_value = 94
    else:   
        zorder_value = k  # Default z-order for other lines
    ax[3].semilogy(T_list, 1e6*ignitionDelays_RG, '-', linestyle=lstyles[k],linewidth=lw, color=colors[k], label=m, zorder=zorder_value)

ax[3].legend(fontsize=lgdfsz, frameon=False, loc='upper right',handlelength=lgdw)
# ax[1,1].set_ylabel(r'Ignition delay [$\mathdefault{\mu s}$]', fontsize=18)
# ax[3].set_xlabel(r'Temperature [K]')
ax[3].tick_params(axis='both', direction="in")
ax[3].tick_params(axis='both', which='minor', direction="in")#, bottom=False)
ax[3].annotate('(d)', xy=(0.2, 0.1), xycoords='axes fraction',ha='right', va='top')

# ax[0].set_xlim([1000.1,1499.99])
# ax[1].set_xlim([1000.1,1499.99])
# ax[2].set_xlim([1000.1,1499.99])
# ax[3].set_xlim([1000.1,1499.99])

plt.subplots_adjust(hspace=0.3)

# plt.subplots_adjust(top=0.98)
if save_plots == True:
    plt.savefig('burkelab_SimScripts/figures/'+name+'_ESSCI.pdf', dpi=1000, bbox_inches='tight')
    plt.savefig('burkelab_SimScripts/figures/'+name+'_ESSCI.png', dpi=1000, bbox_inches='tight')
    plt.savefig('burkelab_SimScripts/figures/'+name+'_ESSCI.svg', dpi=1000, bbox_inches='tight')
# plt.show()     