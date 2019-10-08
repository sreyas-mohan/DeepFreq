import os
import argparse
import json
import warnings
import numpy as np
import torch
from data import fr, loss
from data.source_number import aic_arr, sorte_arr, mdl_arr
import util
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.family'] = 'serif'
params = {
    'font.size': 8,
    'legend.fontsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 11,
    'text.usetex': True,
    'text.latex.unicode': True,
    'figure.figsize': [7, 4]  # instead of 4.5, 4.5
}
matplotlib.rcParams.update(params)
plt.style.use('seaborn-deep')
palette = sns.color_palette('deep', 10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True,
                        help='The data dir. Should contain the .npy files for the tested dB and the frequency file.')
    parser.add_argument('--fr_path', default=None, type=str, required=True,
                        help='Frequency representation module path.')
    parser.add_argument('--counter_path', default=None, type=str,
                        help='Counter module path. If None only the frequency representation module is tested')
    parser.add_argument('--psnet_path', default=None, type=str,
                        help='Path of the psnet.')
    parser.add_argument('--psnet_counter_path', default=None, type=str,
                        help='Path of the counter module associated with the psnet.')
    parser.add_argument('--cblasso_dir', default='test_dataset/cblasso_results', type=str,
                        help='Directory containing CBLasso performance on test data')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='The output directory where the results will be written.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the content of the output directory')

    args = parser.parse_args()

    fr_module, _, _, _, _ = util.load(args.fr_path, 'fr')
    fr_module.cpu()
    fr_module.eval()
    xgrid = np.linspace(-0.5, 0.5, fr_module.fr_size, endpoint=False)

    counter_module = None
    if args.counter_path is not None:
        counter_module, _, _, _, _ = util.load(args.counter_path, 'counter')
        counter_module.cpu()
        counter_module.eval()

    psnet = None
    psnet_counter_model = None
    psnet_grid = None
    if args.psnet_path is not None:
        psnet, _, _, _, _ = util.load(args.psnet_path, 'fr')
        psnet.cpu()
        psnet.eval()
        psnet_grid = np.linspace(-0.5, 0.5, psnet.fr_size, endpoint=False)
        if args.psnet_counter_path is not None:
            psnet_counter_model, _, _, _, _ = util.load(args.psnet_counter_path, 'counter')
            psnet_counter_model.cpu()
            psnet_counter_model.eval()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        raise ValueError('Output directory ({}) already exists and is not empty. Use --overwrite to overcome.'.format(
            args.output_dir))
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'test.args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    music_fnr_arr, model_fnr_arr, periodogram_fnr_arr = [], [], []
    psnet_fnr_arr, psnet_counter_acc = [], []
    model_chamfer, music_aic_chamfer, music_mdl_chamfer = [], [], []
    psnet_chamfer = []
    counter_acc, mdl_acc, aic_acc, sorte_acc = [], [], [], []

    assert os.path.exists(args.data_dir), 'Data directory does not exist'

    with open(os.path.join(args.data_dir, 'data.args'), 'r') as f:
        data_args = json.load(f)
    signal_dim = data_args['signal_dimension']
    num_test = data_args['n_test']

    dB = [float(x) for x in data_args['dB']]
    f = np.load(os.path.join(args.data_dir, 'f.npy'))

    nfreq = np.sum(f >= -0.5, axis=1)

    for k in range(len(dB)):

        data_path = os.path.join(args.data_dir, str(dB[k]) + 'dB.npy')
        if not os.path.exists(data_path):
            warnings.warn('{:.1f}dB data not in data directory.'.format(dB[k]))

        noisy_signals = np.load(data_path)
        noisy_signals = torch.tensor(noisy_signals)

        with torch.no_grad():

            # Evaluate FNR of the frequency representation module
            model_fr_torch = fr_module(noisy_signals)
            model_fr = model_fr_torch.cpu().numpy()
            f_model = fr.find_freq(model_fr, nfreq, xgrid)
            model_fnr_arr.append(100 * loss.fnr(f_model, f, signal_dim) / num_test)

            # Evaluate accuracy of the counter module
            if args.counter_path is not None:
                model_counter = counter_module(model_fr_torch)
                model_counter = model_counter.view(model_counter.size(0))
                model_estimate = torch.round(model_counter).cpu().numpy()
                model_err = 1 - (model_estimate == nfreq).sum() / num_test
                counter_acc.append(100 * model_err)
                f_model_counter = fr.find_freq(model_fr, model_estimate, xgrid, 50)
                model_chamfer.append(loss.chamfer(f_model_counter, f) / num_test)

            # Evalute FNR of the PSnet
            if psnet is not None:
                psnet_fr_torch = psnet(noisy_signals)
                psnet_fr = psnet_fr_torch.cpu().numpy()
                f_psnet = fr.find_freq(psnet_fr, nfreq, psnet_grid)
                psnet_fnr_arr.append(100 * loss.fnr(f_psnet, f, signal_dim) / num_test)

                # Evaluate accuracy of the counter associated with the PSnet
                if psnet_counter_model is not None:
                    psnet_counter = psnet_counter_model(psnet_fr_torch)[:, 0]
                    psnet_counter = psnet_counter.view(psnet_counter.size(0))
                    psnet_counter_estimate = torch.round(psnet_counter).cpu().numpy()
                    psnet_counter_err = 1 - (psnet_counter_estimate == nfreq).sum() / num_test
                    psnet_counter_acc.append(100. * psnet_counter_err)
                    f_psnet_counter = fr.find_freq(psnet_fr, psnet_counter_estimate, psnet_grid, 50)
                    psnet_chamfer.append(loss.chamfer(f_psnet_counter, f) / num_test)

        noisy_signals = noisy_signals.cpu().numpy()
        noisy_signals_c = noisy_signals[:, 0] + 1j * noisy_signals[:, 1]
        music_fr = fr.music(noisy_signals_c, xgrid, nfreq, 25)
        periodogram = fr.periodogram(noisy_signals_c, xgrid)
        f_music = fr.find_freq(music_fr, nfreq, xgrid)
        f_periodogram = fr.find_freq(periodogram, nfreq, xgrid)

        music_fnr_arr.append(100 * loss.fnr(f_music, f, signal_dim) / num_test)
        periodogram_fnr_arr.append(100 * loss.fnr(f_periodogram, f, signal_dim) / num_test)

        if args.counter_path is not None:
            aic_counter = aic_arr(noisy_signals_c, 22)
            mdl_counter = mdl_arr(noisy_signals_c, 25)
            sorte_counter = sorte_arr(noisy_signals_c, 25)
            aic_err = 1 - (aic_counter == nfreq).sum() / num_test
            mdl_err = 1 - (mdl_counter == nfreq).sum() / num_test
            sorte_err = 1 - (sorte_counter == nfreq).sum() / num_test
            aic_acc.append(100 * aic_err)
            mdl_acc.append(100 * mdl_err)
            sorte_acc.append(100 * sorte_err)

            music_ps_mdl = fr.music(noisy_signals_c, xgrid, mdl_counter, 25)
            music_ps_aic = fr.music(noisy_signals_c, xgrid, aic_counter, 25)

            f_music_aic = fr.find_freq(music_ps_aic, aic_counter, xgrid, 50)
            f_music_mdl = fr.find_freq(music_ps_mdl, aic_counter, xgrid, 50)

            chamfer_music_aic = loss.chamfer(f_music_aic, f)
            chamfer_music_mdl = loss.chamfer(f_music_mdl, f)

            music_aic_chamfer.append(chamfer_music_aic / num_test)
            music_mdl_chamfer.append(chamfer_music_mdl / num_test)

    if os.path.isfile(os.path.join(args.cblasso_dir, 'fnr')):
        with open(os.path.join(args.cblasso_dir, 'fnr')) as f:
            cblasso_fnr = json.load(f)
        cblasso_fnr = [cblasso_fnr[str(x)] for x in dB]
    else:
        cblasso_fnr = None

    fig, ax = plt.subplots()
    ax.grid(linestyle='--', linewidth=0.5)
    ax.plot(dB, music_fnr_arr, label='MUSIC', marker='^', linestyle='--', c=palette[0])
    ax.plot(dB, periodogram_fnr_arr, label='Periodogram', marker='p', linestyle='--', c=palette[8])
    if cblasso_fnr is not None:
        ax.plot(dB, cblasso_fnr, label='CBLasso', marker='o', linestyle='--', c=palette[2])
    if args.psnet_path is not None:
        ax.plot(dB, psnet_fnr_arr, label='PSnet', marker='h', linestyle=':', c=palette[5])
    ax.plot(dB, model_fnr_arr, label='DeepFreq', marker='d', c=palette[3])
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('FNR (\%)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.savefig(os.path.join(args.output_dir, 'fnr.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    if args.counter_path is not None:

        fig, ax = plt.subplots()
        ax.grid(linestyle='--', linewidth=0.5)
        ax.plot(dB, aic_acc, label='AIC', marker='^', linestyle='--', c=palette[0])
        ax.plot(dB, mdl_acc, label='MDL', marker='v', linestyle='--', c=palette[1])
        ax.plot(dB, sorte_acc, label='SORTE', marker='o', linestyle='--', c=palette[2])
        if args.psnet_path is not None and args.psnet_counter_path is not None:
            ax.plot(dB, psnet_counter_acc, label='PSnet + Counter', marker='h', linestyle=':', c=palette[5])
        ax.plot(dB, counter_acc, label='DeepFreq', marker='d', c=palette[3])
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error (\%)')
        ax.set_ylim(bottom=0.)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        plt.savefig(os.path.join(args.output_dir, 'counter.pdf'), bbox_inches='tight', pad_inches=0.0)
        plt.close()

        if os.path.isfile(os.path.join(args.cblasso_dir, 'chamfer')):
            with open(os.path.join(args.cblasso_dir, 'chamfer')) as f:
                cblasso_chamfer = json.load(f)
            cblasso_chamfer = [cblasso_chamfer[str(x)] for x in dB]
        else:
            cblasso_chamfer = None
        fig, ax = plt.subplots()
        ax.grid(linestyle='--', linewidth=0.5)
        ax.semilogy(dB, music_aic_chamfer, label='AIC + MUSIC', marker='^', linestyle='--', c=palette[0])
        ax.semilogy(dB, music_mdl_chamfer, label='MDL + MUSIC', marker='v', linestyle='--', c=palette[1])
        if cblasso_chamfer is not None:
            ax.semilogy(dB, cblasso_chamfer, label='CBLasso', marker='o', linestyle='--', c=palette[2])
        if args.psnet_path is not None and args.psnet_counter_path is not None:
            ax.plot(dB, psnet_chamfer, label='PSnet + Counter', marker='h', linestyle=':', c=palette[5])
        ax.semilogy(dB, model_chamfer, label='DeepFreq', marker='d', c=palette[3])
        ax.set(yscale='log')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Chamfer error')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        plt.savefig(os.path.join(args.output_dir, 'endtoend.pdf'), bbox_inches='tight', pad_inches=0.0)
        plt.close()
