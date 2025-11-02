#!/usr/bin/env python
"""
Example 
"""

import bilby 
from bilby.core.prior import Uniform
from bilby.gw.utils import asd_from_freq_series

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg") 
import matplotlib.pyplot as plt
import h5py
import lalsimulation as lalsim
import lal


#some inverse fourier transform helper 
def infft(frequency_domain_strain, sampling_frequency, length=None):
    """ Inverse FFT for use in conjunction with nfft.

    Parameters
    ----------
    frequency_domain_strain: array_like
        Single-sided, normalised FFT of the time-domain strain data (in units
        of strain / Hz).
    sampling_frequency: int, float
        Sampling frequency of the data.
    length: float
        length of the transformed axis of the output.
    """

    time_domain_strain_norm = np.fft.irfft(frequency_domain_strain, n=length)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain


#interpolating to fit our time array
def nr_injection(time):
    """
    This function produces the amplitude for a given 
    NR-derived signal at any given time for a given data file.
    
    Parameters
    ----------
    time : array-like
        A time, or an array of times, at which the amplitudes should be returned.
    datafile : str
        The path to the data file containing the injection.
    """
    
    hp = np.interp(time, times, h_p.data.data)
    hc = np.interp(time, times, h_c.data.data)
    
    return {"plus": hp, "cross": hc}

# Specify the output directory and the name of the simulation.
# bilby setup stuff
outdir = "outdir"
label = "phenomXP"
bilby.core.utils.setup_logger(outdir=outdir, label=label)
    
# Set sampling frequency of the data segment that we're going to inject the signal into
# just be aware of aliasing issues if too low
sampling_frequency = 4096.0

# Set the binary parameters of the NR inejection waveform
# this is the angle and distance to the source
inclination = 1.0471975512
luminosity_distance = 250.
distance = luminosity_distance * lal.PC_SI * 1.0e6
phiRef = 0.0

mtotal = 40.0 # set total mass scale - sim only has relative


# our relevant file path - create separate folder system
filepath = 'chombo/GRChombo_BBSsol02_A147A147q100d12p000_Res40.h5'
#filepath = 'chombo/GRChombo_BBSsol02_A17A17q100d17p000_Res40.h5'
#filepath = 'grav_wave_boson_star_testing_gr/GRChombo_BBSsol02_A147A147q100d12p000_Res40.h5'
f = h5py.File(filepath, 'r')
    
# setting up dict to feed parameters into waveforms
params = lal.CreateDict()
lalsim.SimInspiralWaveformParamsInsertNumRelData(params, filepath)

# Metadata parameters masses

# extract masses and convert to different units
m1 = f.attrs['mass1'] #code units
m2 = f.attrs['mass2']

mass_1 = m1 * mtotal / (m1 + m2) #solar masses
mass_2 = m2 * mtotal / (m1 + m2)

# Choose extrinsic parameters

m1SI = mass_1 * lal.MSUN_SI #in kg
m2SI = mass_2 * lal.MSUN_SI

deltaT = 1.0/sampling_frequency #cadence

# we need to set the lowest trustable frequency - set as lowest simulated frequency, scaled by the chosen mass
f_lower = f.attrs['f_lower_at_1MSUN']/mtotal  # this choice generates the whole NR waveforms from the beginning
fRef = 0   #beginning of the waveform
fStart = f_lower

# Spins
# we only have nonspinning stuff, so dont worry abt it
spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(fRef, mtotal, filepath)

s1x = spins[0]
s1y = spins[1]
s1z = spins[2]
s2x = spins[3]
s2y = spins[4]
s2z = spins[5]

f.close()

# just the specific model to use
approx = lalsim.NR_hdf5

#Which modes to inject? - we only have l=2 modes
inject_l_modes=[2]

# set up array for the modes
ModeArray = lalsim.SimInspiralCreateModeArray()

print('\nInjecting a subset of l modes (all |m|<l modes are being injected): ')
# actually extracting the modes and putting them into the array
for mode in inject_l_modes:
    print('l={}; '.format(mode))
    lalsim.SimInspiralModeArrayActivateAllModesAtL(ModeArray, mode)

# putting the mode array into the params dict for use later
lalsim.SimInspiralWaveformParamsInsertModeArray(params, ModeArray)

#Generate plus and cross polarisations
# try an plot later
# si units
# this is the data from the merger in our direction
h_p, h_c = lalsim.SimInspiralChooseTDWaveform(m1SI, m2SI, s1x, s1y, s1z,
                s2x, s2y, s2z, distance, inclination, phiRef, np.pi/2., 0.0, 0.0, 
                deltaT, fStart, fRef, params, approx)
#Time array    
times = np.arange(len(h_p.data.data))*h_p.deltaT

plt.figure()
plt.plot(times, h_p.data.data, label="h_p")
plt.plot(times, h_c.data.data, label="h_c")
plt.savefig("hphc_plot.png")
#plt.show()
#plot the complex strain as well
#plt.plot(times, h_p.data.data + 1j*h_c.data.data, label="h_p + h_c") # did not work
plt.figure()
plt.plot(times, h_p.data.data + h_c.data.data, label="h_p + h_c")
plt.savefig("hphc_tot_plot.png")


amplitude = []

#Fill in amplitude to find the peak 
# we need to find the peak time to set as coalescence time for analysis later
for i in range(len(h_p.data.data)):
    amp = np.sqrt(h_p.data.data[i] * h_p.data.data[i] + h_c.data.data[i] * h_c.data.data[i])
    amplitude.append(amp)
    
peak_id = amplitude.index(max(amplitude))
peak = times[peak_id]

print("I am at peak at value: ", peak)

hplus = h_p.data.data[peak_id]
hcross = h_c.data.data[peak_id]
print("hplus at peak: ", hplus)
print("hcross at peak: ", hcross)

# the phase when they merge
phase_merger = np.arctan2(-hcross,hplus) + np.pi

print("arctan(-hcross/hplus): ", np.arctan2(-hcross,hplus))
print("Phase at coalescence: ", phase_merger)

# Set the duration of the data segment that we're
# going to inject the signal into
duration = times[-1]
minimum_frequency = fStart

print("Duration of the signal equals: ", duration)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We first establish a  dictionary of parameters that includes all of the different waveform
# parameters, including masses (mass_1, mass_2),  spins (a, tilt, phi), etc.
# this is to create the signal propagating through the detectors
injection_parameters = dict(
    mass_1=mass_1,
    mass_2=mass_2,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=luminosity_distance,
    theta_jn=inclination,
    psi=np.pi/2,
    phase=phase_merger,
    geocent_time=peak,
    ra=1.375,
    dec=-1.2108)

# create signal as it reaches the detectors / propagate signal to detectors
# call the waveform_generator to create our waveform model.
waveform = bilby.gw.waveform_generator.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency,
    time_domain_source_model=nr_injection, start_time=0.0)

# get out the strain in the time and frequency domains
time_domain = waveform.time_domain_strain(parameters=injection_parameters)
time_array = waveform.time_array

fr_domain = waveform.frequency_domain_strain(parameters=injection_parameters)
fr_array = waveform.frequency_array

#Some plotting routines for sanity check and visuals
plt.figure(figsize=(10,8))
plt.plot(fr_array, fr_domain["plus"], label="h_p")
plt.plot(fr_array, fr_domain["cross"], label="h_c")
plt.xlabel("Frequency")
plt.ylabel("Strain")
plt.yscale('log')
plt.legend()
plt.savefig("outdir/waveform_frequency_domain.pdf")

plt.figure(figsize=(10,8))
plt.plot(time_array, time_domain["plus"], label="h_p from Bilby")
plt.plot(time_array, time_domain["cross"], label="h_c from Bilby")
plt.xlim(time_array[0], time_array[-1])
plt.xlabel("Time")
plt.ylabel("Strain")
plt.legend()
plt.savefig("outdir/waveform_time_domain.pdf")

# inject the signal into three interferometers and adds noise
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=0.)
ifos.inject_signal(waveform_generator=waveform,
                   parameters=injection_parameters, raise_error=False);

ifos.plot_data(label="post_injection")


# Generate some intermmediate plot files
# these are different versions of datafiles, prob wont need most of them
start_time = 0
end_time = duration

for ifo in ifos:
    time_idxs = (
                (ifo.time_array >= start_time) &
                (ifo.time_array <= end_time)
            ) #returns a boolean array of interferometer time

    frequency_idxs = np.where(ifo.frequency_mask)[0]
    frequency_idxs = frequency_idxs[::max(1, len(frequency_idxs) // 4000)]
    plot_times = ifo.time_array[time_idxs]
    plot_frequencies = ifo.frequency_array[frequency_idxs]

    frequency_window_factor = (
                np.sum(ifo.frequency_mask)
                / len(ifo.frequency_mask)
            ) # create a window factor using a masking array for limiting the frequency band 
    hf_d = asd_from_freq_series(
                ifo.frequency_domain_strain[frequency_idxs],
                1 / ifo.strain_data.duration)
    ht_d = np.fft.irfft(ifo.whitened_frequency_domain_strain
                            * np.sqrt(np.sum(ifo.frequency_mask))
                            / frequency_window_factor, n=len(time_idxs)
                        )[time_idxs]
    
    # Save data stream here
    np.savetxt("outdir/"+ifo.name+"_time_data_stream.dat", np.column_stack([plot_times, ht_d]), delimiter='   ')

    np.savetxt("outdir/"+ifo.name+"_frequency_data_stream.dat", np.column_stack([plot_frequencies, ifo.frequency_domain_strain[frequency_idxs]]), delimiter='   ')

    np.savetxt("outdir/"+ifo.name+"_frequency_asd_data_stream.dat", np.column_stack([plot_frequencies, hf_d]), delimiter='   ')

    hf_inj = waveform.frequency_domain_strain(injection_parameters)
    hf_inj_det = ifo.get_detector_response(hf_inj, injection_parameters)
    ht_inj_det = infft(hf_inj_det * np.sqrt(2. / ifo.sampling_frequency) /
                            ifo.amplitude_spectral_density_array,
                            sampling_frequency, len(time_idxs))[time_idxs]

    # Save injections to file 
    np.savetxt("outdir/"+ifo.name+"_time_injection_waveform.dat", np.column_stack([plot_times, ht_inj_det]), delimiter='   ')

    np.savetxt("outdir/"+ifo.name+"_frequency_injection_waveform.dat", np.column_stack([plot_frequencies, hf_inj_det[frequency_idxs]]), delimiter='   ')

    np.savetxt("outdir/"+ifo.name+"_frequency_asd_injection_waveform.dat", np.column_stack([plot_frequencies, asd_from_freq_series(hf_inj_det[frequency_idxs], 1 / ifo.strain_data.duration)]), delimiter='   ')
