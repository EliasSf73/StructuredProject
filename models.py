import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        #convolutional layers of encoder
        #initial size=32*32*
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1) # new size: 16*16*16
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1) #new size2: 8*8*32
        # self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1) #new size3: 4*4*32
        # self.conv4=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        #Flatten  layer
        self.flatten=nn.Flatten()  #new size4: 1024*1
        #Linear Layers for mean and log variance: map high-dimensional input to a lower-dimensional latent space then capture the mean and log variance of of latent distribution  #check if the input
        self.fc_mu=nn.Linear(in_features=8*8*16, out_features=latent_dim)#mean vector
        self.logvar=nn.Linear(in_features=8*8*16, out_features=latent_dim)#log variance
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        # x=F.relu(self.conv3(x))
        # x=F.relu(self.conv4(x))
        x=self.flatten(x)
        mu=self.fc_mu(x)
        logvar=self.logvar(x)
        return mu,logvar
class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder,self).__init__()
        #input layer (from the latent space)
        self.fc=nn.Linear(in_features=latent_dim,out_features=8*8*16)
        #transpose convolutions to get back to image dimensions
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=3,stride=2,padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=3,stride=2,padding=1, output_padding=1)
        # self.conv_trans3=nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=3,stride=2,padding=1, output_padding=1)
        # self.conv_trans4=nn.ConvTranspose2d(in_channels=8,out_channels=3,kernel_size=3,stride=2,padding=1, output_padding=1)

    def forward(self,z):
        # Expand z to the initial size of the tensor before convolutions
        z=self.fc(z)
        z=z.view(-1,16,8,8) # Reshape z to (batch_size, channels, height, width)
        #Apply transposed convolution with ReLu activation,except for output layer
        z=F.relu(self.conv_trans1(z))
        z=F.relu(self.conv_trans2(z))
        # z=F.relu(self.conv_trans3(z))
        # z=self.conv_trans4(z)
        z=torch.sigmoid(z) # Apply sigmoid activation to the last layer
        return z
class VAE(nn.Module):
    def __init__(self,latent_dim):
        super(VAE,self).__init__()
        self.encoder=Encoder(latent_dim)
        self.decoder=Decoder(latent_dim)
    def reparametrize(self,mu,logvar):
        """
        Applies the reparameterization trick: z = mu + sigma * epsilon,
        where epsilon is sampled from a standard normal distribution.
        """
        std = torch.exp(0.5 * logvar)  # sigma = exp(0.5 * log_var), for numerical stability

       
        epsilon = torch.randn_like(std)  # Sample epsilon from a standard normal distribution
        z = mu + std * epsilon  # Reparameterize to get z
        return z
    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        """
        mu, log_var = self.encoder(x)  # Encode input to get mu and log_var
        z = self.reparametrize(mu, log_var)  # Reparameterize to get latent vector z
        return self.decoder(z), mu, log_var
