import torch
import torch.nn as n
import torch.nn.functional as f

class variationalAutoEncoder(n.Module):
    def __init__(self): 
        super().__init__()

        # Encoder  Network
        self.relu=n.ReLU()
        self.encoder= n.Sequential(
            n.Conv2d(1,6,3,1),
            self.relu,
            n.Conv2d(6,16,3,1),
            self.relu,
            n.Conv2d(16,84,5,1),
            self.relu,
            n.Flatten(),
            n.Linear(20*20*84, 120),
            self.relu,
            n.Linear(120,64)
        )
        # self.conv1=n.Conv2d(1,6,3,1)
        # self.conv2=n.Conv2d(6,16,3,1)
        # self.conv3=n.Conv2d(16,84,5,1)
            #linear
        # self.fc1 = n.Linear(20*20*84, 120)
        # self.fc2=n.Linear(120,64)
        
        self.mean_gaussian = n.Linear(64, 32)  # Mean of the Gaussian distribution
        self.logvar_gaussian = n.Linear(64, 32)  # Log variance of the Gaussian distribution

        # Decoder Network
        self.decoder=n.Sequential(
            n.Linear(32, 64),
            self.relu,
            n.Linear(64, 120),
            self.relu,
            n.Linear(120, 20*20*84),
        )

        self.reverse_conv = n.Sequential(
            n.ConvTranspose2d(84, 16, 5, 1),
            self.relu,
            n.ConvTranspose2d(16, 6, kernel_size=3, stride=1),
            #output_padding=1 for image size to be    same as input 
            self.relu,
            n.ConvTranspose2d(6,1,3,1) 
        )
                     # self.z_layer1=n.Linear(32, 64)
        # self.z_layer2=n.Linear(64,120)
        # self.z_layer3=n.Linear(120, 20*20*84)
            #conv layer
        # self.z_layer3.view(1,-1,20,20)   # Reshaping to fit Conv layers
        # self.deconv1 = n.ConvTranspose2d(84, 16, 5, 1)
        # self.deconv2 = n.ConvTranspose2d(16, 6, 3, 1, output_padding=1)
        # self.out_layer = n.Conv2d(6,1,3,1)

        # activation function
        

    def encode(self,x):
        #q_phi(z/x)
        embedding=self.encoder(x)
        mean=self.mean_gaussian(embedding)
        logvar=self.logvar_gaussian(embedding)
        return mean,logvar

    def decode(self,z):
        #p_theta(x/z)

        x_reconst=self.decoder(z)
        x_reconst = x_reconst.view(x_reconst.shape[0],-1, 20, 20)

        x_reconst = self.reverse_conv(x_reconst)

        return torch.sigmoid(x_reconst)
    def forward(self,x):
        #Forward pass to calculate loss
        mean,logvar=self.encode(x)
        epsilon=torch.randn_like(logvar)
        z_reparameterized=mean+epsilon*logvar
        x_reconst=self.decode(z_reparameterized)
        return x_reconst,mean,logvar
        




if __name__=='__main__':
    x=torch.randn(1, 1,28, 28) #28 * 28
    vae=variationalAutoEncoder()
    x_recon,mean,logvar=vae(x)
    print("x_rec:",x_recon.shape)
    print('mu:',mean.shape)
    print('sigma:',logvar.shape)
