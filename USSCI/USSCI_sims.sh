# Make executable: chmod +x ./USSCI/USSCI_sims.sh
# ./USSCI/USSCI_sims.sh
fsz=8
fszxtick=7
fszytick=7
fszaxlab=9
lw=0.7
mw=0.5
msz=3.2
lgdw=1.0
lgdfsz=7

# python startup.py &

date='Dec17'

# python USSCI/runLMRRfactory.py --date $date --allPdep 'True' --allPLOG 'True'

# python USSCI/simulateIDT_Beigzadeh_H2O.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date



# python USSCI/simulateFR_Cornell.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 25 --date $date

# python USSCI/simulateJSR_Zhang-2015.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/simulateJSR_Zhang-2016.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateIDT_Shao.py \
# --figwidth 4 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateFR_Gutierrez.py \
# --figwidth 7 --figheight 1.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 25 --date $date

# python USSCI/simulateJSR_Manna.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/simulateIDT_Glarborg2025.py \ #SANDBOX PARAMETERS, NOT REAL
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

## HYDROCARBONS

# python USSCI/simulateIDT_Dai.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/Other_Sims/simulateIDT_UBurke.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 6 --date $date
 
# python USSCI/Other_Sims/simulateIDT_Liu.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateJSR_Zhao2021.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 25 --date $date


# python USSCI/simulateIDT_Beigzadeh_CO2.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date



# ## NITROGEN CHEMISTRY
# # python USSCI/simulateFlameSpeed_Burke.py \
# # --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# # --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateJSR_Zhao2024.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 25 --date $date





# python USSCI/simulateST_Shao2019.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --date $date


# python USSCI/simulateFR_Jian.py \
# --figwidth 2.5 --figheight 3.85 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateResidenceTime_Gubbi_oneModel.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateRateConstant_vs_T_Klippenstein-JPCA2023.py \
# --figwidth 8 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 45 --date $date

# python USSCI/simulateRateConstant_vs_P_Klippenstein-JPCA2023.py \
# --figwidth 8 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 45 --date $date


# python USSCI/simulateJSR_SabiaNH3.py \
# --figwidth 2.5 --figheight 5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateJSR_Cornell2022.py \
# --figwidth 2.5 --figheight 5.65 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 25 --date $date

# python USSCI/simulateJSR_Klippenstein-JPCA2023.py \
# --figwidth 2.5 --figheight 5.55 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 45 --date $date

# python USSCI/simulatePSR_Klippenstein-CNF2018.py \
# --figwidth 2.5 --figheight 5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 25 --date $date

# python USSCI/simulateFR_Rasmussen.py \
# --figwidth 2.5 --figheight 5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 25 --date $date

# python USSCI/simulateResidenceTime_Gubbi_multiModel.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateFlameSpeed_ThinkMech10_H2.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/simulateFlameSpeed_ThinkMech10_CH4_lowP.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/simulateFlameSpeed_ThinkMech10_CH4_hiP.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/simulateFlameSpeed_ThinkMech10_CH2O.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date


# python USSCI/simulateIDT_Shao2019.py \
# --figwidth 7.5 --figheight 2.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date


# SENSITIVITY ANALYSIS
# python USSCI/Sensitivity/simulateFlameSpeed_ThinkMech10_H2_sensitivity.py \
# --figwidth 5.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/Sensitivity/simulateFlameSpeed_ThinkMech10_CH2O_sensitivity.py \
# --figwidth 5.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/Sensitivity/simulateFlameSpeed_ThinkMech10_CH4_lowP_sensitivity.py \
# --figwidth 5.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/Sensitivity/simulateFlameSpeed_ThinkMech10_CH4_hiP_sensitivity.py \
# --figwidth 5.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 20 --date $date

# python USSCI/Sensitivity/simulateFR_Jian_sensitivity.py \
# --figwidth 5.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/Sensitivity/simulateFR_Rasmussen_sensitivity.py \
# --figwidth 7.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateFlameSpeed_Ronney.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/Sensitivity/simulateFlameSpeed_Ronney_sensitivity.py \
# --figwidth 5.5 --figheight 5.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateJSR_Zhang-2018.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 15 --date $date

# python USSCI/simulateIDT_Xia.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateJSR_SabiaH2O.py \
# --figwidth 6.5 --figheight 2.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 6.5 --gridsz 10 --date $date


# python USSCI/simulateIDT_ThinkMech10_C2H2.py \
# --figwidth 7.5 --figheight 2.5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateIDT_ThinkMech10_CH3OH.py \
# --figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 10 --date $date

# python USSCI/simulateJSR_Lavadera.py \
# --figwidth 2.5 --figheight 6.66667 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw 2 --lgdfsz 5.5 --gridsz 25 --date $date

python USSCI/Other_Sims/simulateIDT_Zhao.py \
--figwidth 2.5 --figheight 2.35 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
--lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.5 --gridsz 16 --date $date