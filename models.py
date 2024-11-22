import numpy as np
import sys, os, glob, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from loader import NPY2Dataset, npy_loader
from utility import conv2d_output_shape, convtransp2d_output_shape
# Define model

# define a Conv VAE
#kernel_size = 4 # (4, 4) kernel
#init_channels = 8 # initial number of filters
#image_channels = 1 # MNIST images are grayscale
#latent_dim = 16 # latent dimension for sampling

class ConvVAE(nn.Module):
    # to do
    # output clauster prediction...
    # regeneration sample from model...
    # dropout ...

    def __init__(self,args): 
        self.hidden_dims=args.hidden_dims #    hidden_dims=[8,16,32,64],
        self.kernel_size=args.kernel_size #    kernel_size=[4,4,4,4],
        self.stride=args.stride #    stride=[2,2,2,2],
        self.padding=args.padding #    stride=[2,2,2,2],
        self.output_padding=args.output_padding #    stride=[2,2,2,2],
        self.latent_dim=args.latent_dim #    latent_dim=16,
        self.num_classes=args.num_classes #    num_classes=10,
        self.in_size=args.in_size #    in_size=[inputH,inputW],
        self.image_channels=args.image_channels #    image_channels =1,
        self.dense_n=args.dense_n
        super(ConvVAE, self).__init__()

        # encoder
        modules = []
        en_dim = (self.in_size[0],self.in_size[1])
        in_channels = self.image_channels

        for i in range(len(self.hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=self.hidden_dims[i],
                              kernel_size= self.kernel_size[i],
                              stride= self.stride[i],
                              padding  = self.padding[i]),
                    nn.BatchNorm2d(self.hidden_dims[i]),
                    nn.LeakyReLU()
                ),
            )
            curr_en_dim = conv2d_output_shape(en_dim, kernel_size=self.kernel_size[i], stride=self.stride[i], pad=self.padding[i], dilation=1)
            #print("encoder -> input_channel: ",in_channels, "out_channel: ",self.hidden_dims[i],"input size: ", en_dim,"output size: ",curr_en_dim)
            en_dim=curr_en_dim 
            in_channels=self.hidden_dims[i]

        self.encoder = nn.Sequential(*modules)

        # fully connected layers for learning representations
        # there is a suggestion for dense_n = en_dim[0]*en_dim[1]* hidden_dims[-1] 
        self.fc1 = nn.Linear(self.hidden_dims[-1]*en_dim[0]*en_dim[1], self.dense_n)
        self.fc_mu = nn.Linear(self.dense_n, self.latent_dim)
        self.fc_log_var = nn.Linear(self.dense_n, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.hidden_dims[-1]*en_dim[0]*en_dim[1])

        # decoder
        modules = []

        rev_hidden_dims = self.hidden_dims.copy()
        rev_hidden_dims.reverse()
        rev_kernel_size = self.kernel_size.copy()
        rev_kernel_size.reverse()
        rev_stride = self.stride.copy()
        rev_stride.reverse()
        rev_padding = self.padding.copy()
        rev_padding.reverse()
        self.in_size_decoder = (en_dim[0],en_dim[1])
        de_dim = self.in_size_decoder
        #print("self.in_size_decoder: ",de_dim)
        for i in range(len(rev_hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(rev_hidden_dims[i],
                                       rev_hidden_dims[i + 1], 
                                       kernel_size=rev_kernel_size[i],
                                       stride = rev_stride[i],
                                       padding= rev_padding[i],
                                       output_padding=self.output_padding[i]),
                    nn.BatchNorm2d(rev_hidden_dims[i + 1]),
                    nn.LeakyReLU()
                ),
            )
            curr_de_dim = convtransp2d_output_shape(de_dim, kernel_size=rev_kernel_size[i], stride=rev_stride[i], pad=rev_padding[i], dilation=1, out_pad=self.output_padding[i])
            #print("decoder -> input_channel: ",rev_hidden_dims[i], "out_channel: ",rev_hidden_dims[i + 1],"input size: ", de_dim,"output size: ",curr_de_dim)
            de_dim=curr_de_dim 

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                    nn.ConvTranspose2d(rev_hidden_dims[-1],
                                       rev_hidden_dims[-1],
                                       kernel_size=rev_kernel_size[-1],
                                       stride = rev_stride[-1],
                                       padding= rev_padding[-1],
                                       output_padding=self.output_padding[-1]),
                    nn.BatchNorm2d(rev_hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(rev_hidden_dims[-1], out_channels= 1,
                              kernel_size= rev_kernel_size[-1], padding= rev_padding[-1]),
                    nn.Tanh())

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        in_size = x.size()
        x = self.encoder(x)
        #batch, ch, h, w = x.shape
        #print("x.shape: ",batch, ch, h, w)
        x = torch.flatten(x, start_dim=1)
        hidden = self.fc1(x)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        return [mu, log_var]

    def decode(self, z):
        z = z.view(-1, self.hidden_dims[-1], self.in_size_decoder[0],self.in_size_decoder [1])
        result = self.decoder(z)
        reconstruction = self.final_layer(result)
        ### the size of output of decoder is different from the size of input
        reconstruction = F.adaptive_avg_pool2d(reconstruction,(self.in_size[0],self.in_size[1]) )

        return reconstruction 

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size

        return mu + (eps * std)

    def forward(self, x):
        ## no embeded_class
        ## if there is no class label for input data
        ## need to decide by model
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        recon = self.decode(z) 

        return  [recon, mu, log_var]

    '''
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]
    '''

    def loss_function(self, x, x_hat, mean, log_var):
        #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss + KLD

        #return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'KLD':kld_loss}
        return loss

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def sample_gumbel(self, shape):
        """ sample from Gumbel(0,1)

        Args:
            shape: (array) containing dimensions of the specified sample
        """
        U = torch.rand_like(shape)

        return -torch.log(-torch.log(U+1e-6)+1e-6)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """ sample from the gumbel-softmax distribution

        args:
            logits(array): [batch_size, n_class] unnormalized log-probs
            temperature(float): non-negative scalar
            hard: if True take argmax, but differentiate with respect to soft sample y

        returns:
            y: [batch_size, n_class] sample from the gumbel-softmax distribution
            If true the returned sample will be one-hot, otherwise it will be a probability
            distribution that sums to 1 across classes
        """
        gumbel_softmax_sample = logits + self.sample_gumbel(logits)
        y = nn.Softmax(gumbel_softmax_sample/self.temperature)
        if hard:

            #k = tf.shape(logits)[-1]
            #y_hard = tf.cast(tf.equal(y, tf.reduce_sum(y, 1, keep_dims=True)), y.dtype)
            #y = tf.stop_gradient(y_hard-y)+y
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.equal(y, tf.reduce_sum(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard-y)+y
        return y


