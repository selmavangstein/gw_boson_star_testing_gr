from matplotlib import pyplot as plt
import numpy

filename = 'A17A17q100d17p000/outdir/H1_time_injection_waveform.dat'
numpy.loadtxt(filename)
data = numpy.loadtxt(filename)
time = data[:,0]
strain = data[:,1]
plt.plot(time, strain)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.savefig('A17A17q100d17p000/outdir/testplot.png')