from LMRRfactory import makeYAML

models = {
    'Alzueta': 'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    'Mei': 'chemical_mechanisms/Mei-2019/mei-2019.yaml',
    # 'Glarborg': "chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
    'Zhang': "chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
    'Otomo': "chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
    'Stagni': "chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    # 'Shrestha': "chemical_mechanisms/Shrestha-2021/shrestha-2021.yaml",
    'Han': "chemical_mechanisms/Han-2021/han-2021.yaml"
    }

for m in models.keys():
    makeYAML(mechInput=models[m],
             outputPath='USSCI/factory_mechanisms',
             allPdep=False,date='Oct24')

# import numpy as np
# T_list=np.linspace(200,2000,100)
# P_list=np.logspace(-1,2,num=10)
# mF.convertToTroe(T_list,P_list)