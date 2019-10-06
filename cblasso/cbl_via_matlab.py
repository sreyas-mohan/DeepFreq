import sys
from tqdm import tqdm
import numpy as np
sys.path.append('/usr/local/lib/python3.5/dist-packages/')
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'sdp')


def find_freq_sdp(signals, f):
    nfreq = (f > -0.5).sum(axis=1)
    fest = -np.ones((f.shape[0], f.shape[1]))
    for i in tqdm(range(len(nfreq))):
        freqs = eng.find_freq_cbl(matlab.double(list(signals[i]), is_complex=True), int(nfreq[i]))
        freqs = np.asarray(freqs)
        if len(freqs.shape) == 2:
            freqs = freqs[:, 0]
        else:
            freqs = np.array([freqs])
        num_spikes = min(len(freqs), nfreq[i])
        fest[i, :num_spikes] = freqs
    return fest


def find_freq_sdp_thresh(signals, threshold, max_freq=10):
    fest = -np.ones((signals.shape[0], max_freq))
    for i in tqdm(range(signals.shape[0])):
        freqs = eng.find_freq_cbl_thresh(matlab.double(list(signals[i]), is_complex=True), threshold)
        freqs = np.asarray(freqs)
        if len(freqs.shape) == 2:
            freqs = freqs[:, 0]
        else:
            freqs = np.array([freqs])
        num_spikes = min(len(freqs), max_freq)
        fest[i, :num_spikes] = freqs[:num_spikes]
    return fest
