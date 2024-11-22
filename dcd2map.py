import pyemma
import itertools
import numpy as np
import mdtraj as md
import os

def pdb_dcd_input(args):
    '''
    dcd2npy prepares the residual pair distance matrix and rmsd of protein structures generated from selected trajectory files.
    corresponding input paramenters:
        python cvae_example_tmp.py
            --pdb protein_dcd/a_prot.pdb
            --traj protein_dcd/a_prot_dyn
            --sel1 resid 0 to 42 and name CA
            --sel2 resid 43 to 85 and name CA
            --first_traj 1
            --end_traj 60
    output:
        dataset.npy: residual pair distance matrix as the data pre-processing input for the training
        rmsd.npy: rmsd for individual protein configurations referring to the first frame of the input trajectory
    '''

    # define residues pairwise maxtrix
    print(f'read pdb file {args.pdb} for topology setup')
    file = md.formats.PDBTrajectoryFile(args.pdb, mode='r')
    top=file.topology
    ## alpha carbon is utilized for protein residue
    indices1 = top.select(' '.join(args.sel1))
    indices2 = top.select(' '.join(args.sel2))
    args.in_size[0] = len(indices1)
    args.in_size[1] = len(indices2)
    print(f'the frist selection: {" ".join(args.sel1)}, {args.in_size[0]} atoms are selected')
    print(f'the second selection: {" ".join(args.sel2)}, {args.in_size[1]} atoms are selected')
    pairs = list(itertools.product(indices1,indices2))
    feat = pyemma.coordinates.featurizer(top)
    feat.add_distances(pairs, periodic=True)

    # import data from traj files
    print(f'read trajectory files from: {args.traj_prefix}')
    pairdist_traj = [(args.traj_prefix+str(i)+'.dcd') for i in range(args.first_traj,args.end_traj+1)]
    pairdist = pyemma.coordinates.load(pairdist_traj, feat)

    return pairdist, pairdist_traj

def dcd2npy(args):
    pairdist_data, data_traj = pdb_dcd_input(args)
    n_traj_file = args.end_traj - args.first_traj + 1
    all_data = np.concatenate(([pairdist_data[i] for i in range(len(data_traj))]))

    pdb_name = args.pdb
    t = md.load_dcd(data_traj[0],top=pdb_name)
    for i in range(1,len(data_traj)):
        a = md.load_dcd(data_traj[i],top=pdb_name)
        t = t.join(a)

    # reduce the volume of data if needed
    all_data = all_data[::10]
    print("all_data_length: ", len(all_data)) 
    t=t[::10]
    print("label length: ",len(t))
    args.n_frame_per_traj = int(len(all_data)/n_traj_file)
    print('n_frame_per_traj = ', args.n_frame_per_traj)
    # rmsd calc.
    rmsds = md.rmsd(t,t[0],0)
    rmsds = rmsds*10
    rmsds = np.array(rmsds)

    # save rmsd and distance maxtrix
    with open('rmsds.npy','wb') as f:
        np.save(f,rmsds)
    with open('dataset.npy','wb') as f:
        np.save(f,all_data)


def gen_sample_dcd(args, labels):
    with open('label.npy','rb') as f:
        labels = np.load(f)

    args.unique = sorted(np.unique(labels))
    if len(args.unique) != args.num_classes:
        print(f'{len(args.unique)} {args.unique} identified unique cluster(s) is different from the input number of cluster {args.num_classes}.') 
    data_sort = []

    for i,n in enumerate(args.unique):
        #data_sort.append(np.argwhere(labels == n)[:1000])
        data_sort.append(np.argwhere(labels == n))

    for i in range(len(data_sort)):
        print(f'Cluster id: {i:>3d},{len(data_sort[i]):>6d} data points')

    # assume the number of frame is equal among all trajectory files
    all_samples = []
    for row in range(len(data_sort)):
        temp = []
        for j in range(len(data_sort[row])):
            if data_sort[row][j] > 1114545:
                a,b = ((data_sort[row][j]-4546)//args.n_frame_per_traj+1,(data_sort[row][j]-4546)%args.n_frame_per_traj)
                ab = np.array((a,b))
                temp.append(ab)
            else:
                a,b = (data_sort[row][j]//args.n_frame_per_traj,data_sort[row][j]%args.n_frame_per_traj)
                ab = np.array((a,b))
                temp.append(ab)
        temp=np.array(temp).squeeze()
        # numbers of data may different within individual clusters
        #print(row,temp.shape)
        all_samples.append(temp)


    pdb_name = args.pdb
    data_traj = [(args.traj+str(i)+'.dcd') for i in range(args.first_traj,args.end_traj+1)]
    top = md.load_pdb(pdb_name)

    trj_source = pyemma.coordinates.source(data_traj, top=top.topology)

    os.makedirs('cluster',exist_ok=True)
    pyemma.coordinates.save_trajs(trj_source, all_samples, outfiles=['./cluster/raw_samples_{}.dcd'.format(n) for n in range(len(args.unique))])

    gen_cluster_dcd(args)


def gen_cluster_dcd(args):
    for i in range(len(args.unique)):
        t = './cluster/raw_samples_'+str(i)+'.dcd'
        top = md.load_pdb(args.pdb)
        trj = md.load_dcd(t,top=top.topology)
        # align config
        trj = md.Trajectory.superpose(trj,trj[0])
        ca = [a.index for a in top.topology.atoms if a.name=='CA']
        avg = trj.xyz.mean(axis=0)
        top.xyz = avg
        rmsd = md.rmsd(trj,top,0,atom_indices=ca)
        indices = rmsd.argsort()[:50]
        name='./cluster/cluster_'+str(i)+'.dcd'
        namepdb = './cluster/cluster_'+str(i)+'.pdb'
        trj[indices].save_dcd(name)
        trj[indices[0]].save_pdb(namepdb)


