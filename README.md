# traj_vae_clustering
Min-Kang Hsieh

This is Variational AutoEncoder (VAE) based protein strucutre analysis for extracting enssential protein configurations from a molecular dynamics (MD) simulations trajectory.
A series of residue-residue pair distance as an input is trained by using a VAE algorithm to generate a model, then the latern vector is further classified by using a clustering algorithm.
The centers of each cluster that nearest to the average of protein coordiate (measureed by the residual RMSD) is collected as new trajector files (e.g. cluster_0.dcd) that could be used for further analysis of individual clusters. 
