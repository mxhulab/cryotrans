import argparse
import mrcfile
import sys
import torch
import numpy as np
from pathlib import Path
from skimage.transform import downscale_local_mean
from time import time
from typing import Optional
from numpy.typing import NDArray
from .ode_net import ODENet
from .particles import particles
from .gridding import Gridding
from .wasserstein2_loss import Wasserstein2Loss
from .tetramesh import tetramesh, MeshLoss

def parse_args():
    parser = argparse.ArgumentParser(description = 'CryoTRANS: Predicting high-resolution maps of rare conformations using neural ODEs in cryo-EM.')

    basic_group = parser.add_argument_group('Basic arguments.')
    basic_group.add_argument('-i0', '--initial-map',       type = str,                     help = 'Path of initial map.')
    basic_group.add_argument('-t0', '--initial-threshold', type = float,                   help = 'Threshold for the initial map.')
    basic_group.add_argument('-i1', '--target-map',        type = str,                     help = 'Path of target map.')
    basic_group.add_argument('-t1', '--target-threshold',  type = float,                   help = 'Threshold for the target map.')
    basic_group.add_argument('-d',  '--directory',         type = str,                     help = 'Working directory.')
    basic_group.add_argument('-g',  '--gpu',               type = int,   default = 0,      help = 'Which gpu to use, 0 by default.')
    basic_group.add_argument('-w',  '--weight',            type = str,   required = False, help = 'Path of network weight file as initial model.')
    basic_group.add_argument('-b',  '--binning',           type = int,   default = 1,      help = 'Binning level, 1 by default.')
    basic_group.add_argument('-n',  '--n_steps',           type = int,                     help = 'Number of training steps.')
    basic_group.add_argument('-p',  '--period',            type = int,                     help = 'For periodic report.')

    advanced_group = parser.add_argument_group('Advanced arguments.')
    advanced_group.add_argument('--depth',   type = int,   default = 3,    help = 'Depth of velocity net (MLP).')
    advanced_group.add_argument('--width',   type = int,   default = 100,  help = 'Width of velocity net (MLP).')
    advanced_group.add_argument('--w2_eps',  type = float, default = 1e-4, help = 'Entropic regularisation parameter for W2 loss.')
    advanced_group.add_argument('--w2_iter', type = int,   default = 5,    help = 'Number of Sinkhorn iteration for computing W2 loss.')
    advanced_group.add_argument('--l2',      action = 'store_true',        help = 'Use L2 refine instead of W2.')
    advanced_group.add_argument('--lr',      type = float, default = 1e-3, help = 'Learning rate. Suggest 1e-4 when doing L2 refine.')
    advanced_group.add_argument('--mu_mesh', type = float,                 help = 'Regularization parameter for tetrahedral mesh loss.')

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def train(
    a0 : NDArray[np.float32],
    a1 : NDArray[np.float32],
    odenet : ODENet,
    n_steps : int,
    period : int,
    directory : Path,
    w2_eps : float = 1e-4,
    w2_iter : int = 5,
    l2 : bool = False,
    lr : float = 1e-3,
    mu_mesh : Optional[float] = None,
):
    rho, xs = particles(a0)
    if mu_mesh is not None:
        mesh = tetramesh(xs, a0.shape[0])
        mesh = torch.tensor(mesh, dtype = torch.float32, device = 'cuda')
        mesh = MeshLoss(mesh)
    rho = torch.tensor(rho, dtype = torch.float32, device = 'cuda')
    xs = torch.tensor(xs, dtype = torch.float32, device = 'cuda')
    a1 = torch.tensor(a1, dtype = torch.float32, device = 'cuda')

    grd = Gridding(a0.shape)
    if not l2:
        w2 = Wasserstein2Loss(eps = w2_eps, maxiter = w2_iter)
    optimizer = torch.optim.Adam(odenet.parameters(), lr = lr)

    losses = []
    loss_file = directory.joinpath('loss.log')
    loss_file.write_text('')

    time0 = time()
    print('|-----------------------------------------|')
    print('|   iter   |       loss       |   time    |')
    print('|----------|------------------|-----------|')

    i_period = 0
    for i_step in range(n_steps + 1):
        optimizer.zero_grad()
        b1 = grd(rho, odenet(xs))
        loss = torch.sum(torch.square(b1 - a1)) if l2 else w2(b1, a1)
        if mu_mesh is not None:
            loss += mu_mesh * mesh(odenet)

        losses.append(loss.item())
        if i_step % period == 0:
            print(f'|{i_step:^10d}|{loss.item():^18.9e}|{time() - time0:^11.3e}|')
            with loss_file.open('+a') as loss_out:
                np.savetxt(loss_out, losses)
                losses.clear()
            torch.save(odenet.state_dict(), directory.joinpath(f'net_{i_period}.pt'))
            i_period += 1

        if i_step != n_steps:
            loss.backward()
            optimizer.step()

    print('|-----------------------------------------|')
    print('Save network ...', end = ' ')
    torch.save(odenet.state_dict(), directory.joinpath('net.pt'))
    print('done.')

def main():
    args = parse_args()

    a0 : NDArray[np.float32] = mrcfile.read(args.initial_map)
    a1 : NDArray[np.float32] = mrcfile.read(args.target_map)
    n = a0.shape[0]
    assert a0.dtype == a1.dtype == np.float32
    assert a0.shape == a1.shape == (n, n, n)

    t0 = args.initial_threshold
    t1 = args.target_threshold
    a0 = np.where(a0 < t0, 0, a0)
    a1 = np.where(a1 < t1, 0, a1)
    a0 /= a0.max()
    a1 *= a0.sum() / a1.sum()

    directory = Path(args.directory).absolute()
    directory.mkdir(parents = True, exist_ok = True)
    if not directory.is_dir():
        raise RuntimeError(f'Invalid working directory: {args.directory}.')

    torch.cuda.set_device(torch.device('cuda', args.gpu))
    odenet = ODENet(depth = args.depth, width = args.width)
    if args.weight is not None:
        try:
            print(f'Try loading input weight file {args.weight} ...', end = ' ')
            odenet.load_state_dict(torch.load(args.weight))
            print('succeeded!')
        except:
            print('failed!')
            print('Random initialization applied.')
    odenet.to('cuda')

    binning = args.binning
    if binning <= 0:
        raise ValueError(f'Binning level should be a positive integer.')
    if n % binning != 0:
        raise ValueError(f'Binning level {binning} dost not divide boxsize.')
    if binning > 1:
        print('Bin maps ...', end = ' ')
        a0 = downscale_local_mean(a0, (binning, binning, binning))
        a1 = downscale_local_mean(a1, (binning, binning, binning))
        print('done.')

    train(
        a0, a1, odenet, args.n_steps, args.period, directory,
        w2_eps = args.w2_eps,
        w2_iter = args.w2_iter,
        l2 = args.l2,
        lr = args.lr,
        mu_mesh = args.mu_mesh
    )
