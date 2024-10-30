import os
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import csv
parser = argparse.ArgumentParser()
mpl.rc('font',family='Times New Roman')
f, ax = plt.subplots(1, 3, figsize=(6.5, 2.5)) 
plt.subplots_adjust(wspace=0.3)
parser.add_argument('--date', type=str, help="sim date = ")
parser.add_argument('--slopeVal', type=float, help="slope value = ")
parser.add_argument('--curveVal', type=float, help="curve value = ")
args = parser.parse_args()


lstyles = ["solid","dashed","dotted"]*6
colors = ["xkcd:purple","xkcd:teal","k"]*3
models = {
    'Alzueta-2023': {
        'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
        # 'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
        # 'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
                },
    'Mei-2019': {
        'base': r'chemical_mechanisms/Mei-2019/mei-2019.yaml',
        # 'LMRR': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR.yaml',
        # 'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR_allP.yaml',
                },
    'Zhang-2017': {
        'base': r"chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
        # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR.yaml",
        # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR_allP.yaml",
                },
    # 'Otomo-2018': {
    #     'base': r"chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR_allP.yaml",
    #             },
    # 'Stagni-2020': {
    #     'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    #     'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
    #     'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
    #             },
    # 'Han-2021': {
    #     'base': r"chemical_mechanisms/Han-2021/han-2021.yaml",
    #     # 'LMRR': f"USSCI/factory_mechanisms/{args.date}/han-2021_LMRR.yaml",
    #     # 'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/han-2021_LMRR_allP.yaml",
    #             },
}

P = 20 # bar

expData=['1bar','10bar','20bar']
expData_eq=['1bar_eq','10bar_eq','20bar_eq']

T_fuel = 300
T_air = 650
phi = 1.22

lw=1
mkrw=0.5
mkrsz=3

# widths=[1.6,0.15,0.05]
# widths_eq=[3,1.6,1]
widths=0.05
widths_eq=1

def save_state_to_csv(filename, etat):
    headers = list(etat.keys())
    data = zip(*etat.values())
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

def getFlame(gaz,largeur):
    flame = ct.FreeFlame(gaz,width=largeur)
    flame.set_refine_criteria(ratio=3, slope=args.slopeVal, curve=args.curveVal)
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
    conc = np.multiply(flame.X[gaz.species_names.index("NH2")], np.divide(P*1e5,np.multiply(Rjoule,etat['T [K]'])))
    distance_index=np.argmax(conc)
    tau_f = np.array(etat['dist [m]'][distance_index])/S_b
    etat['tau [ms]']=[(ft - tau_f)*1000 for ft in full_time] # [ms]
    return etat

for z,n in enumerate(models):
    print(f"{n}")
    for k,m in enumerate(models[n]):
        def cp(T,P,X):
            gas_stream = ct.Solution(list(models[n].values())[k])
            gas_stream.TPX = T, P*1e5, {X:1}
            return gas_stream.cp_mole # [J/kmol/K]
        cp_fuel = cp(T_fuel,P,'NH3') # [J/kmol/K]
        cp_o2 = cp(T_air,P,'O2') # [J/kmol/K]
        cp_n2 = cp(T_air,P,'N2') # [J/kmol/K]
        x_fuel = (phi*(1/0.75)*0.21)/(1+phi*(1/0.75)*0.21)
        x_o2 = 0.21*(1-x_fuel)
        x_n2 = 0.79*(1-x_fuel)
        # x_air=1-x_fuel
        T_mix = (x_fuel*cp_fuel*T_fuel+(x_o2*cp_o2+x_n2*cp_n2)*T_air)/(x_fuel*cp_fuel+ x_o2*cp_o2 + x_n2*cp_n2)
        # print(f"Getting mixture state...")
        mix = ct.Solution(list(models[n].values())[k])
        mix.TPX = T_mix, P*1e5,{'NH3':x_fuel,'O2':x_o2,'N2':x_n2}
        state = getFlame(mix,widths)
        path=f'USSCI/data/residence-time/'+args.date
        os.makedirs(path,exist_ok=True)
        if k==0:
            mix_eq = ct.Solution(list(models[n].values())[k])
            mix_eq.TPX = T_mix, P*1e5,{'NH3':x_fuel,'O2':x_o2,'N2':x_n2}
            state_eq = getFlame(mix_eq,widths_eq)
            csv_filename =path+f'/{m}_{list(models[n].keys())[k]}_eq.csv'
            save_state_to_csv(csv_filename,state_eq)
            print(f"Equilibrium state saved to CSV.")
        csv_filename =path+f'//{m}_{list(models[n].keys())[k]}.csv'
        save_state_to_csv(csv_filename,state)
        print(f"Transient state saved to CSV.")
print("Simulation complete!")