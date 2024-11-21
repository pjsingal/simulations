import LMRRfactory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str)
parser.add_argument('--allPdep', type=str, default='False')
args = parser.parse_args()

allPdep = args.allPdep
date = args.date

models = {
    # 'AramcoMech30': 'chemical_mechanisms/AramcoMech30/aramco30.yaml',
    # 'Alzueta': 'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    # 'Mei': 'chemical_mechanisms/Mei-2019/mei-2019.yaml',
    # 'Glarborg': "chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
    'Goldsmith': "chemical_mechanisms/Goldsmith-2012/goldsmith-2012.yaml",
    # 'Zhang': "chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
    # 'Otomo': "chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
    # 'Stagni': "chemical_mechanisms/Stagni-2020/stagni-2020.yaml",
    # 'Shrestha': "chemical_mechanisms/Shrestha-2021/shrestha-2021.yaml",
    # 'Han': "chemical_mechanisms/Han-2021/han-2021.yaml"
    }


if allPdep == 'True':
    for m in models.keys():
        LMRRfactory.makeYAML(mechInput=models[m],
                            outputPath=f"USSCI/factory_mechanisms/{date}",
                            allPdep=True)
        LMRRfactory.makeYAML(mechInput=models[m],
                            outputPath=f"USSCI/factory_mechanisms/{date}",
                            allPdep=False)
else:
    for m in models.keys():
        LMRRfactory.makeYAML(mechInput=models[m],
            outputPath=f"USSCI/factory_mechanisms/{date}",
            allPdep=False)