import os
import cantera as ct
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gridsz', type=int, help="gridsz = ", default=10)
parser.add_argument('--date', type=str, help="sim date = ",default='May28')
parser.add_argument('--slopeVal', type=float, help="slope value = ")
parser.add_argument('--curveVal', type=float, help="curve value = ")
args = parser.parse_args()

############# CHANGE THESE ####################################################################################
gridsz = args.gridsz
date=args.date

# fuel_list = np.linspace(0.14,0.5,gridsz) #fuel mole fractions
# fuel_list = np.linspace(0.14,0.4,gridsz)
# alpha_list = [1.0,0.8,0.6,0.4,0.2,0.0]
# a_st = [0.75,0.7,0.65,0.6,0.55,0.5]
p_list = np.linspace(1,20,gridsz)

def widthFit(p):
   return round(float(2.0717*np.exp(-0.2586*p)),4)
widths = []
for p in p_list:
   widths.append(widthFit(p))

# alpha_list = [1.0,0.6]
# a_st = [0.75,0.65]
alpha_list = [1.0,0.8,0.6,0.4,0.2,0.0]
a_st = [0.75,0.7,0.65,0.6,0.55,0.5]

fratio=3
fslope=args.slopeVal #should be low enough that the results don't depend on the value. 0.02 for both is a good place to start. Try 0.01 and 0.05 and see if there are any differences
fcurve=args.curveVal
models = {
    'Alzueta-2023': {
        'base': r'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
        'LMRR': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR.yaml',
        'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/alzuetamechanism_LMRR_allP.yaml',
                },
    'Mei-2019': {
        'base': r'chemical_mechanisms/Mei-2019/mei-2019.yaml',
        'LMRR': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR.yaml',
        'LMRR-allP': f'USSCI/factory_mechanisms/{args.date}/mei-2019_LMRR_allP.yaml',
                },
    'Zhang-2017': {
        'base': r"chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/zhang-2017_LMRR_allP.yaml",
                },
    'Otomo-2018': {
        'base': r"chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/otomo-2018_LMRR_allP.yaml",
                },
    'Stagni-2020': {
        'base': r"chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
        'LMRR': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR.yaml",
        'LMRR-allP': f"USSCI/factory_mechanisms/{args.date}/stagni-2020_LMRR_allP.yaml",
                },
}
###############################################################################################################


phi_list = [0.8,1.0,1.22,1.4]
# width = 3  # m
loglevel = 1  # amount of diagnostic output (0 to 8)

T_fuel = 300
T_air = 650

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def cp(T,P,X,model):
  gas_stream = ct.Solution(model)
  gas_stream.TPX = T, P*1e5, {X:1}
  return gas_stream.cp_mole # [J/kmol/K]

for phi in phi_list:
  for x, alpha in enumerate(alpha_list):
      for z,n in enumerate(models):
        for k,m in enumerate(models[n]):
          mbr = []
          for i, p in enumerate(p_list):
            gas = ct.Solution(models[n][m])
            cp_fuel = cp(T_fuel,p,'NH3',models[n][m]) # [J/kmol/K]
            cp_o2 = cp(T_air,p,'O2',models[n][m]) # [J/kmol/K]
            cp_n2 = cp(T_air,p,'N2',models[n][m]) # [J/kmol/K]
            
            x_fuel = (phi*(1/0.75)*0.21)/(1+phi*(1/0.75)*0.21)
            x_o2 = 0.21*(1-x_fuel)
            x_n2 = 0.79*(1-x_fuel)
            T_mix = (x_fuel*cp_fuel*T_fuel+(x_o2*cp_o2+x_n2*cp_n2)*T_air)/(x_fuel*cp_fuel+ x_o2*cp_o2 + x_n2*cp_n2)
            
            gas.TPX= T_mix, p*1e5, {'NH3':x_fuel,'O2':x_o2,'N2':x_n2}
            f = ct.FreeFlame(gas, width=widths[i])
            f.set_refine_criteria(ratio=fratio, slope=fslope, curve=fcurve)
            f.transport_model = 'multicomponent'
            f.soret_enabled = True
            f.solve(loglevel=loglevel, auto=True)
            mbr.append(f.velocity[0] * 100) # cm/s

          # Save phi_list and mbr to CSV
          path=f'USSCI/data/flame-speed/Gubbi/vsP/'+args.date
          os.makedirs(path,exist_ok=True)
          csv_filename =path+f'/{n}_{m}_{phi}phi_{alpha}alpha.csv'
          data = zip(p_list, mbr)
          save_to_csv(csv_filename, data)