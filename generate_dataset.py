import os
import argparse
import torch
import numpy as np
from data import load_dataloader
from data import noise
from data.data import gen_signal
import json



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the data will be written.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument("--n_test", default=1000, type=int,
                        help="Number of signals")
    parser.add_argument("--signal_dimension", default=50, type=int,
                        help="Dimension of sinusoidal signal")
    parser.add_argument("--minimum_separation", default=1., type=float,
                        help="Minimum distance between spikes, normalized by 1/signal_dim")
    parser.add_argument("--max_freq", default=10, type=int,
                        help="Maximum number of frequency, the distribution is uniform between 1 and max_freq")
    parser.add_argument("--distance", default="normal", type=str,
                        help="Distribution type of the inter-frequency distance")
    parser.add_argument("--amplitude", default="normal_floor", type=str,
                        help="Distribution type of the spike amplitude")
    parser.add_argument("--floor_amplitude", default=0.1, type=float,
                        help="Minimum spike amplitude (only used for the normal_floor distribution)")
    parser.add_argument('--dB', nargs='+', default=['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50'],
                        help='additional dB levels')

    parser.add_argument("--numpy_seed", default=105, type=int,
                        help="Numpy seed")
    parser.add_argument("--torch_seed", default=94, type=int,
                        help="Numpy seed")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite to overcome.".format(args.output_dir))
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'data.args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    s, f, nfreq = gen_signal(
        num_samples=args.n_test,
        signal_dim=args.signal_dimension,
        num_freq=args.max_freq,
        min_sep=args.minimum_separation,
        distance=args.distance,
        amplitude=args.amplitude,
        floor_amplitude=args.floor_amplitude,
        variable_num_freq=True)

    np.save(os.path.join(args.output_dir, 'infdB'), s)
    np.save(os.path.join(args.output_dir, 'f'), f)

    eval_snrs = [np.exp(np.log(10) * float(x) / 10) for x in args.dB]

    for k, snr in enumerate(eval_snrs):
        noisy_signals = noise.noise_torch(torch.tensor(s), snr, 'gaussian').cpu()
        np.save(os.path.join(args.output_dir, '{}dB'.format(float(args.dB[k]))), noisy_signals)

