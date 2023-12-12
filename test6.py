import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


ts = np.load('/content/drive/MyDrive/SSEF 2024/output.npy', allow_pickle=True)


dt = 1. / 2.048e6
xtime = np.arange(len(ts)) * dt


myfi = plt.figure(figsize=[15, 5])
plt.plot(xtime, np.abs(ts))
plt.xlabel('Signal (Arb)')
plt.xlabel('Time since start (s)')


maxme = np.argmax(np.abs(ts))
t_max = xtime[maxme]
print(t_max)


window = 4e-3


myfi = plt.figure(figsize=[15, 5])
plt.plot(xtime, np.abs(ts))
plt.ylabel('Signal (Arb)')
plt.xlabel('Time since start (s)')
t0 = t_max - window / 2.
t1 = t_max + window / 2.
plt.xlim([t0, t1])
plt.title('Zoomed-In View around Maximum')


fs = 2.048e6
sub = slice(int(t0 * fs), int(t1 * fs))

dt = 1. / fs
xtime = np.arange(len(ts)) * dt
myfi, axs = plt.subplots(2, 1, figsize=[15, 10], sharex=True)
axs[0].plot(xtime[sub] - t0, np.abs(ts)[sub])

_, _, _, sgimg = axs[1].specgram(ts[sub], Fs=fs, vmin=-100, vmax=-70)
axs[0].set_ylabel('Signal (Arb)')
axs[1].set_ylabel('Spectrum (Hz)')
axs[1].set_xlabel('Time since start (s)')
plt.colorbar(sgimg, orientation='horizontal', ax=axs[1])


plt.close(myfi)



filtered_data = signal.savgol_filter(np.abs(ts), 5, 2)



locs, props = signal.find_peaks(filtered_data, height=0.25, distance=10000, width=1)



def plot_pulse(ts, time, window):
    t0 = time - window/2.
    t1 = time + window/2.
    fs = 2.048e6
    sub = slice(int(t0 * fs), int(t1 * fs))
    dt = 1. / fs
    xtime = np.arange(len(ts)) * dt
    myfi, axs = plt.subplots(2, 1, figsize=[15, 10], sharex=True)
    axs[0].plot(xtime[sub] - t0, np.abs(ts[sub]))  # Fix here
    _, _, _, sgimg = axs[1].specgram(ts[sub], Fs=fs, vmin=-100, vmax=-70)  # Fix here
    axs[0].set_ylabel('Signal (Arb)')
    axs[1].set_ylabel('Spectrum (Hz)')
    axs[1].set_xlabel('Time since start (s)')
    plt.colorbar(sgimg, orientation='horizontal', ax=axs[1])


    plt.close(myfi)


for loc in locs:
    plot_pulse(ts, xtime[loc], 1e-3)

plt.show()
