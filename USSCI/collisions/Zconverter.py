# Collision frequency converter

# Converting values from Matsugi-2021

#T=750
Z = [8.72, 6.26, 30.3, 14.9, 16.7, 22.4, 53.1]
r_ZLJ = [1.48,1.52,3.00,3.18,3.56,3.77,8.92]
r_Zcap = [1.03,0.88,1.79,1.87,2.06,2.12,5.32]

ZLJ750 = []
Zcap750 = []

for i, val in enumerate(Z):
    ZLJ750.append(round(val/r_ZLJ[i],2))
    Zcap750.append(round(val/r_Zcap[i],2))

#T=1500
Z = [10.8,6.65,33.2,18.2,19.9,27.9,84.0]
r_ZLJ = [1.43,1.29,2.60,3.09,3.39,3.78,13.0]
r_Zcap = [1.13,0.83,1.74,2.04,2.19,2.35,7.50]

ZLJ1500 = []
Zcap1500 = []

for i, val in enumerate(Z):
    ZLJ1500.append(round(val/r_ZLJ[i],2))
    Zcap1500.append(round(val/r_Zcap[i],2))


ZLJ_list = []
Zcap_list = []

for i, val in enumerate(Z):
    ZLJ_list.append([ZLJ750[i],ZLJ1500[i]])
    Zcap_list.append([Zcap750[i],Zcap1500[i]])


print("ZLJ: ", ZLJ_list)
print("Zcap: ", Zcap_list)