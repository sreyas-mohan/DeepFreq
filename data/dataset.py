import torch
import numpy as np
import torch.utils.data as data_utils
from data import fr, data
from .noise import noise_torch


def load_dataloader(num_samples, signal_dim, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                    kernel_type, kernel_param, batch_size, xgrid):
    clean_signals, f, nfreq = data.gen_signal(num_samples, signal_dim, max_n_freq, min_sep, distance=distance,
                                              amplitude=amplitude, floor_amplitude=floor_amplitude,
                                              variable_num_freq=True)
    frequency_representation = fr.freq2fr(f, xgrid, kernel_type, kernel_param)

    clean_signals = torch.from_numpy(clean_signals).float()
    f = torch.from_numpy(f).float()
    frequency_representation = torch.from_numpy(frequency_representation).float()
    dataset = data_utils.TensorDataset(clean_signals, frequency_representation, f)
    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_dataloader_fixed_noise(num_samples, signal_dim, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                                kernel_type, kernel_param, batch_size, xgrid, snr, noise):
    clean_signals, f, nfreq = data.gen_signal(num_samples, signal_dim, max_n_freq, min_sep, distance=distance,
                                              amplitude=amplitude, floor_amplitude=floor_amplitude,
                                              variable_num_freq=True)
    frequency_representation = fr.freq2fr(f, xgrid, kernel_type, kernel_param)

    clean_signals = torch.from_numpy(clean_signals).float()
    f = torch.from_numpy(f).float()
    frequency_representation = torch.from_numpy(frequency_representation).float()

    noisy_signals = noise_torch(clean_signals, snr, noise)
    dataset = data_utils.TensorDataset(noisy_signals, clean_signals, frequency_representation, f)
    return data_utils.DataLoader(dataset, batch_size=batch_size)


def make_train_data(args):
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    if args.kernel_type == 'triangle':
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param = args.gaussian_std / args.signal_dim
    return load_dataloader(args.n_training, signal_dim=args.signal_dim, max_n_freq=args.max_n_freq,
                           min_sep=args.min_sep, distance=args.distance, amplitude=args.amplitude,
                           floor_amplitude=args.floor_amplitude, kernel_type=args.kernel_type,
                           kernel_param=kernel_param, batch_size=args.batch_size, xgrid=xgrid)


def make_eval_data(args):
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    if args.kernel_type == 'triangle':
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param = args.gaussian_std / args.signal_dim
    return load_dataloader_fixed_noise(args.n_validation, signal_dim=args.signal_dim, max_n_freq=args.max_n_freq,
                                       min_sep=args.min_sep, distance=args.distance, amplitude=args.amplitude,
                                       floor_amplitude=args.floor_amplitude, kernel_type=args.kernel_type,
                                       kernel_param=kernel_param, batch_size=args.batch_size, xgrid=xgrid,
                                       snr=args.snr, noise=args.noise)
