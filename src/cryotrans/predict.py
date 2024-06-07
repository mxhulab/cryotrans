import argparse
import mrcfile
import sys
import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from .ode_net import ODENet
from .particles import particles
from .gridding import Gridding

def parse_args():
    parser = argparse.ArgumentParser(description = 'CryoTRANS: Quality Preserved Trajectory for Boosting Resolutions of Rare Conformations in cryo-EM.')

    basic_group = parser.add_argument_group('Basic arguments.')
    basic_group.add_argument('-i', '--initial-map',       type = str,                      help = 'Path of initial map.')
    basic_group.add_argument('-t', '--initial-threshold', type = float, default = 0.,      help = 'Threshold for the initial map.')
    basic_group.add_argument('-d', '--directory',         type = str,                      help = 'Working directory.')
    basic_group.add_argument('-p', '--prefix',            type = str,   default = 'frame', help = 'Prefix for output movie.')
    basic_group.add_argument('-g', '--gpu',               type = int,   default = 0,       help = 'Which gpu to use, 0 by default.')
    basic_group.add_argument('-w', '--weight',            type = str,                      help = 'Path of network weight file as initial model.')

    advanced_group = parser.add_argument_group('Advanced arguments.')
    advanced_group.add_argument('--depth',   type = int,   default = 3,    help = 'Depth of velocity net (MLP).')
    advanced_group.add_argument('--width',   type = int,   default = 100,  help = 'Width of velocity net (MLP).')

    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()

def predict(
    a0 : NDArray[np.float32],
    voxel_size : float,
    odenet : ODENet,
    directory : Path,
    prefix : str
):
    print('------------------------Predicting-------------------------')

    rho, xs = particles(a0)
    rho = torch.tensor(rho, dtype = torch.float32, device = 'cuda')
    xs  = torch.tensor(xs , dtype = torch.float32, device = 'cuda')
    grd = Gridding(a0.shape)

    with torch.no_grad():
        for i, xi in enumerate(odenet.trajectory(xs)):
            print(f'Processing frame {i:02d} ...', end = ' ')
            vol = grd(rho, xi).to('cpu').numpy()
            with mrcfile.new(directory.joinpath(f'{prefix}_{i:02d}.mrc'), data = vol, overwrite = True) as mrc:
                mrc.voxel_size = voxel_size
            print('done.')

def main():
    args = parse_args()

    with mrcfile.open(args.initial_map, permissive = True) as mrc:
        a0 : NDArray[np.float32] = mrc.data
        voxel_size = mrc.voxel_size
        print(f'Voxelsize read from input map: {voxel_size} Angstrom.')
    n = a0.shape[0]
    assert a0.dtype == np.float32
    assert a0.shape == (n, n, n)
    a0 = np.where(a0 < args.initial_threshold, 0, a0)

    directory = Path(args.directory).absolute()
    directory.mkdir(parents = True, exist_ok = True)
    if not directory.is_dir():
        raise RuntimeError(f'Invalid working directory: {args.directory}.')

    torch.cuda.set_device(torch.device('cuda', args.gpu))
    odenet = ODENet(depth = args.depth, width = args.width)
    print(f'Try loading input weight file {args.weight} ...', end = ' ')
    odenet.load_state_dict(torch.load(args.weight))
    print('succeeded!')
    odenet.to('cuda')

    predict(a0, voxel_size, odenet, directory, args.prefix)
