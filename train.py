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


def train_eval_fr(args, fr_module, fr_optimizer, fr_criterion, fr_scheduler, train_loader, val_loader, xgrid,
                  epoch, tb_writer):
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
                epoch, args.n_epochs_fr + args.n_epochs_c, time.time() - epoch_start_time, loss_train_fr, loss_val_fr,
                fnr_val)


def train_eval_counter(args, fr_module, counter_module, counter_optimizer, counter_criterion, counter_scheduler,
                       train_loader, val_loader, epoch, tb_writer):
    epoch_start_time = time.time()
    fr_module.eval()
    counter_module.train()
    loss_train_counter, acc_train_counter = 0, 0
    for batch_idx, (clean_signal, target_fr, freq) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_fr, freq = clean_signal.cuda(), target_fr.cuda(), freq.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        nfreq = (freq >= -0.5).sum(dim=1)
        if args.use_cuda:
            nfreq = nfreq.cuda()
        if args.c_module_type == 'classification':
            nfreq = nfreq - 1
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
            output_fr = output_fr.detach()
        output_counter = counter_module(output_fr)
        if args.c_module_type == 'regression':
            output_counter = output_counter.view(output_counter.size(0))

            nfreq = nfreq.float()
        loss_counter = counter_criterion(output_counter, nfreq)
        if args.c_module_type == 'classification':
            estimate = output_counter.max(1)[1]
        else:
            estimate = torch.round(output_counter)
        acc_train_counter += estimate.eq(nfreq).sum().cpu().item()
        loss_train_counter += loss_counter.data.item()

        counter_optimizer.zero_grad()
        loss_counter.backward()
        counter_optimizer.step()

    loss_train_counter /= args.n_training
    acc_train_counter *= 100 / args.n_training

    counter_module.eval()

    loss_val_counter = 0
    acc_val_counter = 0
    for batch_idx, (noisy_signal, _, target_fr, freq) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_fr = noisy_signal.cuda(), target_fr.cuda()
        nfreq = (freq >= -0.5).sum(dim=1)
        if args.use_cuda:
            nfreq = nfreq.cuda()
        if args.c_module_type == 'classification':
            nfreq = nfreq - 1
        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
            output_counter = counter_module(output_fr)
        if args.c_module_type == 'regression':
            output_counter = output_counter.view(output_counter.size(0))
            nfreq = nfreq.float()
        loss_counter = counter_criterion(output_counter, nfreq)
        output_counter = output_counter
        if args.c_module_type == 'regression':
            estimate = torch.round(output_counter)
        elif args.c_module_type == 'classification':
            estimate = torch.argmax(output_counter, dim=1)

        acc_val_counter += estimate.eq(nfreq).sum().item()
        loss_val_counter += loss_counter.data.item()

    loss_val_counter /= args.n_validation
    acc_val_counter *= 100 / args.n_validation

    counter_scheduler.step(acc_val_counter)

    tb_writer.add_scalar('counter_loss_training', loss_train_counter, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('counter_loss_validation', loss_val_counter, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('counter_accuracy_training', acc_train_counter, epoch - args.n_epochs_fr)
    tb_writer.add_scalar('counter_accuracy_validation', acc_val_counter, epoch - args.n_epochs_fr)

    logger.info("Epochs: %d / %d, Time: %.1f, Training counter loss: %.2f, Vadidation counter loss: %.2f, "
                "Training accuracy: %.2f %%, Validation accuracy: %.2f %%",
                epoch, args.n_epochs_fr + args.n_epochs_c, time.time() - epoch_start_time, loss_train_counter,
                loss_val_counter,
                acc_train_counter, acc_val_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/experiment_name', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--signal_dim', type=int, default=50, help='input dimension')
    parser.add_argument('--fr_size', type=int, default=1000, help='size of the frequency representation')
    parser.add_argument('--min_sep', type=float, default=1.,
                        help='minimum separation between spikes, normalized by 1/signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='normal_floor', help='amplitude distribution of spikes')
    parser.add_argument('--floor_amplitude', type=float, default=0.1, help='minimum amplitude of spikes')
    # frequency representation module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: fr or psnet')
    parser.add_argument('--fr_n_layers', type=int, default=20, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=64, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=25, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=125, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=8,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')
    # counter module parameters
    parser.add_argument('--c_module_type', type=str, default='regression', help='[regression or classification]')
    parser.add_argument('--c_n_layers', type=int, default=20, help='number of layers in the counter module')
    parser.add_argument('--c_n_filters', type=int, default=32, help='number of filters per layer in the counter module')
    parser.add_argument('--c_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the counter module')
    parser.add_argument('--c_downsampling', type=int, default=5, help='stride of the first convolutional layer')
    parser.add_argument('--c_kernel_in', type=int, default=25, help='kernel size of the first convolutional layer')
    # kernel parameters used to generate the target frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the artificial frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=4000,
                        help='slope of the triangle kernel normalized by 1/signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.3,
                        help='std of the gaussian kernel normalized by 1/signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=50000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=1000, help='# of validation data')
    parser.add_argument('--n_figures', type=int, default=20, help='# of validation signals to print')
    parser.add_argument('--lr_fr', type=float, default=0.0003,
                        help='initial learning rate for adam optimizer used for the frequency representation module')
    parser.add_argument('--lr_counter', type=float, default=0.0003,
                        help='initial learning rate for adam optimizer used for the counter module')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=1., help='snr of input signals')
    parser.add_argument('--n_epochs_fr', type=int, default=200, help='number of epochs used to train the fr module')
    parser.add_argument('--n_epochs_c', type=int, default=100, help='number of epochs used to train the counter module')
    parser.add_argument('--max_n_freq', type=int, default=10, help='maximum number of frequencies')
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
    counter_module = modules.set_counter_module(args)
    fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'fr')
    counter_optimizer, counter_scheduler = util.set_optim(args, counter_module, 'counter')
    start_epoch = 1

    logger.info('[Network] Number of parameters in FR module : %.3f M' % (util.model_parameters(fr_module) / 1e6))
    logger.info(
        '[Network] Number of parameters in counter module : %.3f M' % (util.model_parameters(counter_module) / 1e6))

    fr_criterion = torch.nn.MSELoss(reduction='sum')

    if args.c_module_type == 'classification':
        counter_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    else:
        counter_criterion = torch.nn.MSELoss(reduction='sum')

    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)

    for epoch in range(start_epoch, args.n_epochs_c + args.n_epochs_fr + 1):

        if epoch < args.n_epochs_fr:
            train_eval_fr(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer, fr_criterion=fr_criterion,
                          fr_scheduler=fr_scheduler, train_loader=train_loader, val_loader=val_loader,
                          xgrid=xgrid, epoch=epoch, tb_writer=tb_writer)
        else:
            train_eval_counter(args=args, fr_module=fr_module, counter_module=counter_module,
                               counter_optimizer=counter_optimizer, counter_criterion=counter_criterion,
                               counter_scheduler=counter_scheduler, train_loader=train_loader,
                               val_loader=val_loader, epoch=epoch, tb_writer=tb_writer)

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs:
            util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)
            util.save(counter_module, counter_optimizer, counter_scheduler, args, epoch, 'counter')
