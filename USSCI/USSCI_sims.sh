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

date='Nov4'

# python USSCI/runLMRRfactory.py --date $date --allPdep 'True'

python USSCI/simulateIDT_Shao_USSCI.py \
--figwidth 7 --figheight 6.66667 --fsz $fszaxlab --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
--lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 9 --date $date &

python USSCI/simulateshocktubeShao_USSCI.py \
--figwidth 7 --figheight 2.5 --fsz $fszaxlab --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
--lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 3 --date $date &

python USSCI/simulateJSR_NH3_USSCI.py \
--figwidth 10 --figheight 5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
--lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.8 --gridsz 50 --date $date

python USSCI/simulateJSR_H2O_USSCI.py \
--figwidth 10 --figheight 5 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
--lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 5.8 --gridsz 50 --date $date

# python USSCI/simulateresidencetimeGubbi_USSCI.py --date $date --slopeVal 0.05 --curveVal 0.05

# python USSCI/simulateflamespeedGubbi_vsP_USSCI.py --date $date --slopeVal 0.05 --curveVal 0.05 --gridsz 50

# python USSCI/simulateflamespeedGubbi_vsPhi_USSCI.py --date $date --slopeVal 0.05 --curveVal 0.05 --gridsz 20

# python USSCI/simulateresidencetimeGubbi_FromData.py \
# --figwidth 6.5 --figheight 2.72 --fsz $fszaxlab --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz $lgdfsz --date $date --slopeVal 0.05 --curveVal 0.05 --xscale 'log' --yscale 'log'&
