import os
import sys
import time
import argparse
import logging

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from data import dataset
import modules
import util
from data.noise import noise_torch
from data import fr
from data.loss import fnr

logger = logging.getLogger(__name__)


def train_frequency_representation(args, fr_module, fr_optimizer, fr_criterion, fr_scheduler, train_loader, val_loader,
                                   xgrid, epoch, tb_writer):
    """
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    loss_train_fr = 0
    for batch_idx, (clean_signal, target_fr, freq) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_fr = clean_signal.cuda(), target_fr.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        fr_optimizer.zero_grad()
        output_fr = fr_module(noisy_signal)
        loss_fr = fr_criterion(output_fr, target_fr)
        loss_fr.backward()
        fr_optimizer.step()
        loss_train_fr += loss_fr.data.item()

    fr_module.eval()
    loss_val_fr, fnr_val = 0, 0
    for batch_idx, (noisy_signal, _, target_fr, freq) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_fr = noisy_signal.cuda(), target_fr.cuda()
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
        loss_fr = fr_criterion(output_fr, target_fr)
        loss_val_fr += loss_fr.data.item()
        nfreq = (freq >= -0.5).sum(dim=1)
        f_hat = fr.find_freq(output_fr.cpu().detach().numpy(), nfreq, xgrid)
        fnr_val += fnr(f_hat, freq.cpu().numpy(), args.signal_dim)

    loss_train_fr /= args.n_training
    loss_val_fr /= args.n_validation
    fnr_val *= 100 / args.n_validation

    tb_writer.add_scalar('fr_l2_training', loss_train_fr, epoch)
    tb_writer.add_scalar('fr_l2_validation', loss_val_fr, epoch)
    tb_writer.add_scalar('fr_FNR', fnr_val, epoch)

    fr_scheduler.step(loss_val_fr)
    logger.info("Epochs: %d / %d, Time: %.1f, FR training L2 loss %.2f, FR validation L2 loss %.2f, FNR %.2f %%",
                epoch, args.n_epochs_fr + args.n_epochs_fc, time.time() - epoch_start_time, loss_train_fr, loss_val_fr,
                fnr_val)


def train_frequency_counting(args, fr_module, fc_module, fc_optimizer, fc_criterion, fc_scheduler, train_loader,
                             val_loader, epoch, tb_writer):
    """
    Train the frequency-counting module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.eval()
    fc_module.train()
    loss_train_fc, acc_train_fc = 0, 0
    for batch_idx, (clean_signal, target_fr, freq) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_fr, freq = clean_signal.cuda(), target_fr.cuda(), freq.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        nfreq = (freq >= -0.5).sum(dim=1)
        if args.use_cuda:
            nfreq = nfreq.cuda()
        if args.fc_module_type == 'classification':
            nfreq = nfreq - 1
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
            output_fr = output_fr.detach()
        output_fc = fc_module(output_fr)
        if args.fc_module_type == 'regression':
            output_fc = output_fc.view(output_fc.size(0))

            nfreq = nfreq.float()
        loss_fc = fc_criterion(output_fc, nfreq)
        if args.fc_module_type == 'classification':
            estimate = output_fc.max(1)[1]
        else:
            estimate = torch.round(output_fc)
        acc_train_fc += estimate.eq(nfreq).sum().cpu().item()
        loss_train_fc += loss_fc.data.item()

        fc_optimizer.zero_grad()
        loss_fc.backward()
        fc_optimizer.step()

    loss_train_fc /= args.n_training
    acc_train_fc *= 100 / args.n_training

    fc_module.eval()

    loss_val_fc = 0
    acc_val_fc = 0
    for batch_idx, (noisy_signal, _, target_fr, freq) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_fr = noisy_signal.cuda(), target_fr.cuda()
        nfreq = (freq >= -0.5).sum(dim=1)
        if args.use_cuda:
            nfreq = nfreq.cuda()
        if args.fc_module_type == 'classification':
            nfreq = nfreq - 1
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
            output_fc = fc_module(output_fr)
        if args.fc_module_type == 'regression':
            output_fc = output_fc.view(output_fc.size(0))
            nfreq = nfreq.float()
        loss_fc = fc_criterion(output_fc, nfreq)
        if args.fc_module_type == 'regression':
            estimate = torch.round(output_fc)
        elif args.fc_module_type == 'classification':
            estimate = torch.argmax(output_fc, dim=1)

        acc_val_fc += estimate.eq(nfreq).sum().item()
        loss_val_fc += loss_fc.data.item()

    loss_val_fc /= args.n_validation
    acc_val_fc *= 100 / args.n_validation

    fc_scheduler.step(acc_val_fc)

    tb_writer.add_scalar('fc_loss_training', loss_train_fc, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('fc_loss_validation', loss_val_fc, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('fc_accuracy_training', acc_train_fc, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('fc_accuracy_validation', acc_val_fc, epoch - args.n_epochs_fr)

    logger.info("Epochs: %d / %d, Time: %.1f, Training fc loss: %.2f, Vadidation fc loss: %.2f, "
                "Training accuracy: %.2f %%, Validation accuracy: %.2f %%",
                epoch, args.n_epochs_fr + args.n_epochs_fc, time.time() - epoch_start_time, loss_train_fc,
                loss_val_fc, acc_train_fc, acc_val_fc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/experiment_name', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=256, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=50, help='dimensionof the input signal')
    parser.add_argument('--fr_size', type=int, default=1000, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=10,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    parser.add_argument('--min_sep', type=float, default=1.,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='normal_floor', help='spike amplitude distribution')
    parser.add_argument('--floor_amplitude', type=float, default=0.1, help='minimum amplitude of spikes')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=1., help='snr parameter')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=20, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=64, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=25, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=125, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=8,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')
    # frequency-counting (fc) module parameters
    parser.add_argument('--fc_module_type', type=str, default='regression', help='[regression | classification]')
    parser.add_argument('--fc_n_layers', type=int, default=20, help='number of layers in the fc module')
    parser.add_argument('--fc_n_filters', type=int, default=32, help='number of filters per layer in the fc module')
    parser.add_argument('--fc_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fc module')
    parser.add_argument('--fc_downsampling', type=int, default=5, help='stride of the first convolutional layer')
    parser.add_argument('--fc_kernel_in', type=int, default=25, help='kernel size of the first convolutional layer')
    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=4000,
                        help='slope of the triangle kernel normalized by signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.3,
                        help='std of the gaussian kernel normalized by signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=50000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=1000, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.0003,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--lr_fc', type=float, default=0.0003,
                        help='initial learning rate for adam optimizer used for the frequency-counting module')
    parser.add_argument('--n_epochs_fr', type=int, default=200, help='number of epochs used to train the fr module')
    parser.add_argument('--n_epochs_fc', type=int, default=100, help='number of epochs used to train the fc module')
    parser.add_argument('--save_epoch_freq', type=int, default=50,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    train_loader = dataset.make_train_data(args)
    val_loader = dataset.make_eval_data(args)

    fr_module = modules.set_fr_module(args)
    fc_module = modules.set_fc_module(args)
    fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'fr')
    fc_optimizer, fc_scheduler = util.set_optim(args, fc_module, 'fc')
    start_epoch = 1

    logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
                util.model_parameters(fr_module) / 1e6))
    logger.info('[Network] Number of parameters in the frequency-counting module : %.3f M' % (
                util.model_parameters(fc_module) / 1e6))

    fr_criterion = torch.nn.MSELoss(reduction='sum')

    if args.fc_module_type == 'classification':
        fc_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    else:
        fc_criterion = torch.nn.MSELoss(reduction='sum')

    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)

    for epoch in range(start_epoch, args.n_epochs_fc + args.n_epochs_fr + 1):

        if epoch < args.n_epochs_fr:
            train_frequency_representation(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer, fr_criterion=fr_criterion,
                                           fr_scheduler=fr_scheduler, train_loader=train_loader, val_loader=val_loader,
                                           xgrid=xgrid, epoch=epoch, tb_writer=tb_writer)
        else:
            train_frequency_counting(args=args, fr_module=fr_module, fc_module=fc_module,
                                     fc_optimizer=fc_optimizer, fc_criterion=fc_criterion,
                                     fc_scheduler=fc_scheduler, train_loader=train_loader,
                                     val_loader=val_loader, epoch=epoch, tb_writer=tb_writer)

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_fr + args.n_epochs_fc:
            util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)
            util.save(fc_module, fc_optimizer, fc_scheduler, args, epoch, 'fc')
