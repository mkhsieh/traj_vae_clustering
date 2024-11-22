import numpy as np
import sys, os, glob, re
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from loader import NPY2Dataset, npy_loader
from utility import conv2d_output_shape, convtransp2d_output_shape, EarlyStopping
import argparse
import models as models
from sklearn.cluster import KMeans
from dcd2map import dcd2npy, gen_sample_dcd

'''
This is Variational AutoEncoder (VAE) based protein strucutre analysis for extracting enssential protein 
configurations from a molecular dynamics (MD) simulations trajectory. A series of residue-residue pair 
distance as an input is trained by using a VAE algorithm to generate a model, then the latern vector is 
further classified by using a clustering algorithm. The centers of each cluster that nearest to the average 
of protein coordiate (measureed by the residual RMSD) is collected as new trajector files (e.g. cluster_0.dcd) 
that could be used for further analysis of individual clusters.
'''
def main(args):
    '''
    dcd2npy prepares the residual pair distance matrix and rmsd of protein structures generated from selected trajectory files.
    '''
    dcd2npy(args)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # loading data for training, validation and all samples 
    train_dataloader, val_dataloader, sample_dataloader = dataloading(args)

    model = models.ConvVAE(args).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize EarlyStopping
    # Define the folder path
    folder_path = "checkpoints"

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=5, verbose=True, path='./checkpoints/vae_checkpoint.pth')

    epochs = args.epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, optimizer, device)
        val_loss = validate(val_dataloader, model, device)
        print(f"Epoch {t + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check early stopping
        # save the checkpoint
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break
    print("Done!")

    # Load the best model for evaluation or further use
    model.load_state_dict(torch.load('vae_checkpoint.pth'))

    # post-clustering 
    labels = clustering(sample_dataloader, model, device, args.num_classes)
    
    # generate cluster dcd and pdb files 
    gen_sample_dcd(args,labels)

def dataloading(args):
    # Assuming you have data and labels (e.g., as numpy arrays)
    data = npy_loader('./dataset.npy').float() # Your data (e.g., images)
    labels = npy_loader('./rmsds.npy')  # Corresponding labels
    #print(data.max())
    data =torch.div(data, data.max())
    labels=labels.long()
    train_data = data[:int(len(data) * args.train_ratio)] 
    val_data = data[int(len(data) * args.train_ratio):] 
    train_labels = labels[:int(len(data) * args.train_ratio)] 
    val_labels = labels[int(len(data) * args.train_ratio):] 
    print("read from npy_loader",train_data.type())
    print("read from npy_loader",train_data.size())
    print("read from npy_loader",val_data.type())
    print("read from npy_loader",val_data.size())
    train_dataset = NPY2Dataset(train_data, train_labels, args.image_channels, args.in_size[0],args.in_size[1])
    val_dataset = NPY2Dataset(val_data, val_labels, args.image_channels, args.in_size[0],args.in_size[1])
    sample_dataset = NPY2Dataset(data, labels, args.image_channels, args.in_size[0],args.in_size[1])
    print(len(train_dataset))
    print(len(val_dataset))
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)
    sample_dataloader = DataLoader(sample_dataset,batch_size=args.batch_size,shuffle=False)
    for X, y in train_dataloader:
        print(f"Train Shape of X [N, C, H, W]: {X.shape}")
        print(f"Train Shape of y: {y.shape} {y.dtype}")
        break
    for X, y in val_dataloader:
        print(f"Val Shape of X [N, C, H, W]: {X.shape}")
        print(f"Val Shape of y: {y.shape} {y.dtype}")
        break
    return train_dataloader, val_dataloader, sample_dataloader

def train(dataloader, model, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    overall_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        # Compute prediction error
        pred, mean, log_var = model(X)
        #loss = loss_fn(X, pred, mean, log_var)
        loss = model.loss_function(X, pred, mean, log_var)
        
        # loss per data point
        curr_loss = loss.item()
        overall_loss += curr_loss
        # Backpropagation
        loss.backward()
        optimizer.step()

        current = (batch + 1) * len(X)
        print(f"batch_loss: {curr_loss/len(X):>7f} [{current:>5d}/{size:>5d}]")
    overall_loss = overall_loss/size
    print(f"overall_loss: {overall_loss:>7f} [{size:>5d}]")
    return overall_loss

def validate(dataloader, model, device):
    size = len(dataloader.dataset)
    #num_batches = len(dataloader)
    model.eval()
    overall_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, mean, log_var = model(X)
            loss = model.loss_function(X, pred, mean, log_var)
            #loss = loss_fn(X, pred, mean, log_var)
            curr_loss = loss.item()
            overall_loss += curr_loss
            #print(f"batch_loss: {curr_loss/len(X):>7f} [/{size:>5d}]")
    overall_loss /= size
    print(f"Test Avg loss: {overall_loss:>8f} [{size:>5d}]\n")
    return overall_loss

def clustering(dataloader, model, device, n_cluster):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            mu, _ = model.encode(x)
            latent_vectors.append(mu.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)

    # Perform clustering the latent_vectors 
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(latent_vectors)
    with open('label.npy','wb') as f:
        np.save(f, labels)
    #dcd_sample(labels)
    # Visualize the latent space and clusters
    plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.title("Clustering in Latent Space")
    # Save the figure as a PNG file
    plt.savefig("latent_space_clusters.png", format="png", dpi=300, bbox_inches="tight")
    #plt.show()
    return labels

def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)

    return clusters

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_size', nargs='+', type=int, default=[43,43], help='input map size, i.e.,Number of atoms in trajectory')
    parser.add_argument('--image_channels', type=int, default=1, help='channel of input map')
    parser.add_argument('--batch_size', type=int, default=3000, help='Batch size')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--latent_dim', type=int, default=6, help='latent dimension')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of clusters')
    parser.add_argument('--dense_n',type=int, default=128, help='Number of neurons for dense layer')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[8,16,32,64], help='number of convolutional filters in each layer')
    parser.add_argument('--kernel_size', nargs='+', type=int, default=[4,4,4,4], help='Convolutional kernel sizes')
    parser.add_argument('--stride', nargs='+', type=int, default=[2,2,2,2])
    parser.add_argument('--padding', nargs='+', type=int, default=[1,1,1,1])
    parser.add_argument('--output_padding', nargs='+', type=int, default=[0,0,0,0])
    parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of validation data')
    parser.add_argument('--pdb', dest='pdb', type=str, default=[False], help='the corresponding PDB file of the DCD trajectory. ')
    parser.add_argument('--traj_prefix', dest='traj_prefix', type=str, default=[False], help='Name of the trajectory file')
    parser.add_argument('--first_traj', dest='first_traj', type=int, default=1, help='the first dcd index') 
    parser.add_argument('--end_traj', dest='end_traj', type=int, default=1, help='the last dcd index') 
    parser.add_argument('--sel1', dest='sel1', type=str, default=['all'], nargs='+',
                        help='A atom selection (mdtraj selection style) string which determines the first group of selected '
                             'residues.')
    parser.add_argument('--sel2', dest='sel2', type=str, default=['all'], nargs='+',
                        help='A atom selection (mdtraj selection style) string which determines the second group of selected '
                             'residues.')
    args = parser.parse_args()

    main(args)

