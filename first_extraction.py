#Import modules
import h5py as h5
import numpy as np

filename = "GRChombo_BBSsol02_A147A147q100d12p000_Res40.h5" #Pick you favourite filename
f = h5.File(filename, "r") #Read the file

for key, val in f.attrs.items():
        print("    %s: %s" % (key, val))

#Print the data available in the group for the amplitude (22)-mode
for dset in f["amp_l2_m2"].keys():      
    print (dset)

for l in range(2,8):
        for m in range(-l,l+1):
            mode_str = "amp_l{}_m{}".format(l,m)
            if mode_str in f:
                print("Data available for mode l={}, m={}".format(l,m))

                


time_p = np.array(f["phase_l2_m2"]["X"]) #time array for the amplitude of the (22)-mode 
phase = np.array(f["phase_l2_m2"]["Y"]) #amplitude array for the (22)-mode

time_p_21 = np.array(f["phase_l2_m1"]["X"]) #time array for the amplitude of the (22)-mode 
phase_21 = np.array(f["phase_l2_m1"]["Y"]) #amplitude array for the (22)-mode


time_a = np.array(f["amp_l2_m2"]["X"]) #time array for the amplitude of the (22)-mode 
amp = np.array(f["amp_l2_m2"]["Y"]) #amplitude array for the (22)-mode

time_a_21 = np.array(f["amp_l2_m1"]["X"]) #time array for the amplitude of the (22)-mode 
amp_21 = np.array(f["amp_l2_m1"]["Y"]) #amplitude

#interpolate amp to get same shape as phase, then get same shape
amp = np.interp(time_p, time_a, amp)
amp_21 = np.interp(time_p_21, time_a_21, amp_21)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(time_p, amp*np.cos(phase)) 
plt.plot(time_p_21, amp_21*np.cos(phase_21))
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.savefig("test_plot.png")
