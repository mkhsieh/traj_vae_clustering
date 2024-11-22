import numpy as np
import matplotlib.pyplot as plt
import pyemma, itertools
import mdtraj as md
import argparse
import os, time, argparse

def pdb_dcd_input(args):
    '''
    cluster sample analysis for the residual pair distance matrix and contact map.
    corresponding input paramenters:
        python cvae_example_tmp.py
            --pdb cluster/cluster_1.pdb
            --traj cluster/cluster_1.dcd
            --sel1 resid 0 to 42 and name CA
            --sel2 resid 43 to 85 and name CA
            --DistanceMap --ContactMap 
            --outfolder cluster/cluste_1
    '''

    # define residues pairwise maxtrix
    print(f'read pdb file {args.pdb} for topology setup')
    file = md.formats.PDBTrajectoryFile(args.pdb, mode='r')
    top=file.topology
    ## alpha carbon is utilized for protein residue
    indices1 = top.select(' '.join(args.sel1))
    indices2 = top.select(' '.join(args.sel2))
    args.n1 = len(indices1)
    args.n2 = len(indices2)
    print(f'the frist selection: {" ".join(args.sel1)}, {args.n1} atoms are selected')
    print(f'the second selection: {" ".join(args.sel2)}, {args.n2} atoms are selected')
    pairs = list(itertools.product(indices1,indices2))
    feat = pyemma.coordinates.featurizer(top)
    feat.add_distances(pairs, periodic=True)

    # import data from traj files
    print(f'read trajectory files: {args.traj}')
    pairdist_traj = args.traj
    #args.namebase, _ = os.path.splitext(filename)
    pairdist = pyemma.coordinates.load(pairdist_traj, feat)

    return pairdist

def dis_matrix(args):
    pairdist = pdb_dcd_input(args)
    pairdist = np.array(pairdist)
    pairdist = pairdist*10
    ## for single traj file
    ## data shape [ n_frame_per_file, n_sel1 x n_sel2 ]
    #  for mutiple traj files
    ## data shape [ n_traj_file, n_frame_per_file, n_sel1 x n_sel2 ]
    #print(pairdist.shape)
    #print(pairdist.ndim)

    if len(args.traj) == 1 : pairdist = pairdist.reshape(pairdist.shape[0],pairdist.shape[-1])
    else : pairdist = pairdist.reshape(pairdist.shape[0]*pairdist.shape[1],pairdist.shape[-1])
    #if pairdist.ndim == 3: pairdist = pairdist.reshape(pairdist.shape[0]*pairdist.shape[1],pairdist.shape[-1])
    #if pairdist.ndim == 2: pairdist = pairdist.reshape(pairdist.shape[0],pairdist.shape[-1])
    pairdist = pairdist.reshape(len(pairdist),args.n1,args.n2)

    dis = np.mean(pairdist,axis=0)
    #se = np.std(pairdist, axis=0, ddof=1) / np.sqrt(np.size(pairdist, axis=1))
    frq = np.where(pairdist < args.cutoff, 1, 0)
    #print("frq shape:", frq.shape)
    freq = np.mean(frq,axis=0)
    with open(args.outfolder+'/contact_dis.npy','wb') as f:
        np.save(f,dis)

    with open(args.outfolder+'/contact_freq.npy','wb') as f:
        np.save(f,freq)

    if args.DistanceMap:
        generate_distanacemap(args,dis)

    if args.ContactMap:
        generate_contactmap(args,freq)

    return None

def generate_distanacemap(args,dis):
    xini=1
    yini=1
    plt.figure(figsize=(48,48))
    im = plt.imshow(dis,cmap='Blues', interpolation='nearest',origin='upper')
    ax = plt.gca();
    '''
    # Major ticks
    ax.set_xticks(np.arange(0, n2, 1))
    ax.set_yticks(np.arange(0, n1, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(xini, n2+xini, 1))
    ax.set_yticklabels(np.arange(yini, n1+yini, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, n2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n1, 1), minor=True)

    '''
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.rcParams["font.weight"]= 'bold'
    plt.rcParams["axes.labelweight"]='bold'
    plt.rcParams["figure.autolayout"]=True
    plt.grid(c='black')
    cbar =plt.colorbar(im)
    cbar.ax.tick_params(labelsize=25)
    plt.savefig(args.outfolder+'/DistanceMap.png')
    plt.close()

def generate_contactmap(args,freq):
    xini=1
    yini=1
    plt.figure(figsize=(48,48))
    im = plt.imshow(freq,cmap='Greens', interpolation='nearest',origin='upper')
    ax = plt.gca();
    '''
    # Major ticks
    ax.set_xticks(np.arange(0, n2, 1))
    ax.set_yticks(np.arange(0, n1, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(xini, n2+xini, 1))
    ax.set_yticklabels(np.arange(yini, n1+yini, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, n2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n1, 1), minor=True)

    '''
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.rcParams["font.weight"]= 'bold'
    plt.rcParams["axes.labelweight"]='bold'
    plt.rcParams["figure.autolayout"]=True
    plt.grid(c='black')
    cbar =plt.colorbar(im)
    cbar.ax.tick_params(labelsize=25)
    plt.savefig(args.outfolder+'/ContactMap.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
            description='calculation the contacts from DCD trajectory files.'
    )
    parser.add_argument('--pdb', dest='pdb', type=str, default=[False],
                        help='the corresponding PDB file of the DCD trajectory. '
    )
    parser.add_argument('--traj', dest='traj', type=str, default=[False], nargs='+',
                        help='Name of the trajectory file'
    )
    parser.add_argument('--sel1', dest='sel1', type=str, default=['all'], nargs='+',
                        help='A atom selection (mdtraj selection style) string which determines the first group of selected '
                             'residues. '
    )
    parser.add_argument('--sel2', dest='sel2', type=str, default=['all'], nargs='+',
                        help='A atom selection (mdtraj selection style) string which determines the second group of selected '
                             'residues.'
    )
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=12, nargs=1,
                        help='Non-bonded interaction distance cutoff (Angstroms) for pairfilter which generates '
                             'frequency matrix of contacts.'
    )
    parser.add_argument(
            '--DistanceMap', dest='DistanceMap', action='store_true',
            help='generation of residue pairwise (mean) distance map. ',
            default=False,
    )
    parser.add_argument(
            '--ContactMap', dest='ContactMap', action='store_true',
            help='generation of residue contact frequency map with . ',
            default=False,
    )
    parser.add_argument('--outfolder', dest='outfolder', type=str, default='outfolder',
                        help='Folder path for storing calculation results. '
    )

    args = parser.parse_args()
    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    dis_matrix(args)

if __name__ == '__main__':
    main()
