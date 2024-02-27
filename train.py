import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from models import VAE
import numpy as np

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# the hyperparameters
batch_size = 64
epochs = 20
latent_dim = 64
learning_rate = 1e-2
lambda_= 1
#CIFAR10 Data Loading
transform=transforms.Compose([transforms.ToTensor(),])
train_dataset=datasets.CIFAR10('./data',train=True,download=True,transform=transform)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# Initialize VAE model and optimizer
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#List to track Loss
recon_losses = []
kl_losses=[]
def train(epoch):
    model.train()
    train_loss=0
    for batch_idx,(data,_) in enumerate(train_loader):
        data=data.to(device)
        # Clear previous gradients
        optimizer.zero_grad()
        # Forward pass
        reconstructo,mu,log_var=model(data)
        recon_loss=F.mse_loss(reconstructo,data,reduction='sum')
        kl_loss=-0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
        loss=recon_loss+(kl_loss * lambda_/batch_size)
        # Backpropagate the loss
        loss.backward()
        # Accumulate the loss
        train_loss+=loss.item()
        # Update model parameters
        optimizer.step()
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    # Calculate average losses and store for visualization
    avg_recon_loss = train_loss / len(train_loader.dataset)
    recon_losses.append(avg_recon_loss)
    kl_losses.append(kl_loss.item() / len(train_loader.dataset))
    print(f'====> Epoch: {epoch} Average loss: {avg_recon_loss:.4f}')

# Plotting the losses after training
def visualize_training_loss():
    plt.figure(figsize=(10,5))
    plt.title("Training Losses")
    plt.plot(recon_losses, label="Reconstruction Loss")
    # plt.plot(kl_losses, label="KL Divergence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def visualize_reconstructions():
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(train_loader))
        data = data.to(device)
        reconstruction, _, _ = model(data)
        data = data.cpu()
        reconstruction = reconstruction.cpu()

        # Display original and reconstructed images
        num_images = 8
        fig, axs = plt.subplots(2, num_images, figsize=(20, 4))
        for i in range(num_images):
            axs[0, i].imshow(np.transpose(data[i].numpy(), (1, 2, 0)))
            axs[0, i].set_title('Original')
            axs[1, i].imshow(np.transpose(reconstruction[i].numpy(), (1, 2, 0)))
            axs[1, i].set_title('Reconstruction')
            axs[0, i].axis('off')
            axs[1, i].axis('off')
        plt.show()

for epoch in range(1, epochs + 1):
    train(epoch)

torch.save(model.state_dict(), 'vae_model.pth')


visualize_training_loss()
visualize_reconstructions()