# traj_vae_clustering
Min-Kang Hsieh

This is Variational AutoEncoder (VAE) based protein strucutre analysis for extracting enssential protein configurations from a molecular dynamics (MD) simulations trajectory.
A series of residue-residue pair distance as an input is trained by using a VAE algorithm to generate a model, then the latern vector is further classified by using a clustering algorithm.
The centers of each cluster that nearest to the average of protein coordiate (measureed by the residual RMSD) is collected as new trajector files (e.g. cluster_0.dcd) that could be used for further analysis of individual clusters. 

The defualt module utitize a Convetional VAE algorithm with 4 layers of neural network; the latent vector dimension is set to 6 and the k-mean clustering is used to classify the latent vector from CVAE training outcome; the defualt cluster size is set to 10.  

**Usage:**\
usage: traj_vae_clustering.py [-h] [--image_channels IMAGE_CHANNELS] [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS] [--latent_dim LATENT_DIM] [--num_classes NUM_CLASSES] [--dense_n DENSE_N]
                              [--hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]] [--kernel_size KERNEL_SIZE [KERNEL_SIZE ...]] [--stride STRIDE [STRIDE ...]] [--padding PADDING [PADDING ...]]
                              [--output_padding OUTPUT_PADDING [OUTPUT_PADDING ...]] [--train_ratio TRAIN_RATIO] [--pdb PDB] [--traj_prefix TRAJ_PREFIX] [--first_traj FIRST_TRAJ] [--end_traj END_TRAJ]
                              [--sel1 SEL1 [SEL1 ...]] [--sel2 SEL2 [SEL2 ...]]

**options:**\
  -h, --help            show this help message and exit \
  --image_channels IMAGE_CHANNELS \
                        channel of input map \
  --batch_size BATCH_SIZE \
                        Batch size \
  --lr LR               Learning rate \
  --epochs EPOCHS       Number of epochs for training \
  --latent_dim LATENT_DIM \
                        latent dimension \
  --num_classes NUM_CLASSES \
                        Number of clusters \
  --dense_n DENSE_N     Number of neurons for dense layer \
  --hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...] \
                        number of convolutional filters in each layer \
  --kernel_size KERNEL_SIZE [KERNEL_SIZE ...] \
                        Convolutional kernel sizes \
  --stride STRIDE [STRIDE ...] \
  --padding PADDING [PADDING ...] \
  --output_padding OUTPUT_PADDING [OUTPUT_PADDING ...] \
  --train_ratio TRAIN_RATIO \
                        ratio of validation data \
  --pdb PDB             the corresponding PDB file of the DCD trajectory. \
  --traj_prefix TRAJ_PREFIX \
                        Name of the trajectory file \
  --first_traj FIRST_TRAJ \
                        the first dcd index \
  --end_traj END_TRAJ   the last dcd index \
  --sel1 SEL1 [SEL1 ...] \
                        A atom selection (mdtraj selection style) string which determines the first group of selected residues. \
  --sel2 SEL2 [SEL2 ...] \
                        A atom selection (mdtraj selection style) string which determines the second group of selected residues.\

example:\
python traj_vae_clustering.py --pdb protein_dcd/a_prot.pdb \
                            --traj protein_dcd/a_prot_dyn \
                            --sel1 resid 0 to 42 and name CA \
                            --sel2 resid 43 to 85 and name CA \
                            --first_traj 1 \
                            --end_traj 60 \
                            > vae_clustering.log
