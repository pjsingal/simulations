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

date='Nov23'

# python USSCI/runLMRRfactory.py --date $date --allPdep 'True' --allPLOG 'True'

# python USSCI/simulateFR_phi_USSCI.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 30 --date $date

# python USSCI/simulateST_USSCI.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --date $date

# python USSCI/simulateIDT_USSCI.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 20 --date $date

# python USSCI/simulateIDT_USSCI_diluent.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 20 --date $date

# python USSCI/simulateIDT_Beigzadeh.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 30 --date $date

# python USSCI/simulateIDT_Dai.py \
# --figwidth 6 --figheight 6 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 30 --date $date

# python USSCI/simulateJSR_Lavadera.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 15 --date $date

# python USSCI/simulateJSR_Manna.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 20 --date $date

# python USSCI/simulateIDT_UBurke.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 10 --date $date

# python USSCI/simulateIDT_Shao.py \
# --figwidth 4 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 10 --date $date

# python USSCI/simulateFR_Gutierrez.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 15 --date $date

# python USSCI/simulateFR_Cornell.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 25 --date $date

# python USSCI/simulateJSR_Zhang-2015.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 15 --date $date

# python USSCI/simulateJSR_Zhang-2016.py \
# --figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
# --lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 15 --date $date

python USSCI/simulateJSR_Zhang-2018.py \
--figwidth 7 --figheight 3 --fsz $fszxtick --fszxtick $fszxtick --fszytick $fszytick --fszaxlab $fszaxlab \
--lw $lw --mw $mw --msz $msz --lgdw $lgdw --lgdfsz 7 --gridsz 15 --date $date

# screen -S Alzueta -dm