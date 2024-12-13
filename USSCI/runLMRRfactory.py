import LMRRfactory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str)
parser.add_argument('--allPdep', type=str, default='False')
parser.add_argument('--allPLOG', type=str, default='False')
args = parser.parse_args()

allPdep = args.allPdep
allPLOG = args.allPLOG
date = args.date

models = {
    'Bugler-2016': "chemical_mechanisms/Bugler-2016/bugler-2016.yaml",
    'Song-2019': "chemical_mechanisms/Song-2019/song-2019.yaml",
    'Arunthanayothin-2021': "chemical_mechanisms/Arunthanayothin-2021/arunthanayothin-2021.yaml",
    'Gutierrez-2025': "chemical_mechanisms/Gutierrez-2025/gutierrez-2025.yaml",
    'Cornell-2024': "chemical_mechanisms/Cornell-2024/cornell-2024.yaml",
    'Jian-2024': "chemical_mechanisms/Jian-2024/jian-2024.yaml",
    # 'AramcoMech30-updated': 'chemical_mechanisms/AramcoMech30-updated/aramco30-updated.yaml',
    # 'AramcoMech30': 'chemical_mechanisms/AramcoMech30/aramco30.yaml',
    # 'ThinkMech10': 'chemical_mechanisms/ThinkMech10/think.yaml',
    # 'ThinkMech10_HO2plog': 'chemical_mechanisms/ThinkMech10_HO2plog/think_ho2plog.yaml',
    'Alzueta': 'chemical_mechanisms/Alzueta-2023/alzuetamechanism.yaml',
    'Glarborg': "chemical_mechanisms/Glarborg-2018/glarborg-2018.yaml",
    'Merchant': "chemical_mechanisms/Merchant-2015/merchant-2015.yaml",
    'Zhang-2017': "chemical_mechanisms/Zhang-2017/zhang-2017.yaml",
    'Zhang-2015': "chemical_mechanisms/Zhang-2015/zhang-2015_nhexane.yaml",
    'Zhang-2016': "chemical_mechanisms/Zhang-2016/zhang-2016_nheptane.yaml",
    'Zhang-2018': "chemical_mechanisms/Zhang-2018/zhang-2018_ethanolDME.yaml",
    # 'Otomo': "chemical_mechanisms/Otomo-2018/otomo-2018.yaml",
    'Stagni': "chemical_mechanisms/Stagni-2023/stagni-2023.yaml",
    'Klippenstein-CNF2018': "chemical_mechanisms/Klippenstein-CNF2018/klippenstein-CNF2018.yaml",
    'Glarborg-2025-HNNO': "chemical_mechanisms/Glarborg-2025-HNNO/glarborg-2025-HNNO.yaml",
    }

for m in models.keys():
    LMRRfactory.makeYAML(mechInput=models[m],
                        outputPath=f"USSCI/factory_mechanisms/{date}")
    if allPdep == 'True':
        LMRRfactory.makeYAML(mechInput=models[m],
                            outputPath=f"USSCI/factory_mechanisms/{date}",
                            allPdep=True)
    if allPLOG == 'True':
        LMRRfactory.makeYAML(mechInput=models[m],
                            outputPath=f"USSCI/factory_mechanisms/{date}",
                            allPLOG=True)