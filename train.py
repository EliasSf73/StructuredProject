import torch
import torchvision.datasets as datasets 
from tqdm import tqdm
import os
from torch import nn,optim
from model import variationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import sys
# Device configuration
Device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM=784 # Dimension of input image (for MNIST: 28*28 pixels)
H_DIM=200
Z_DIm=20
NUM_EPOCHS=10
BATCH_SIZE =32
LEARNING_RATE=3e-4
Lambda=0.01

#DATASET LOADING
dataset=datasets.MNIST("./",download=True,train=True,transform=transforms.ToTensor())
train_loader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)
model=variationalAutoEncoder().to(Device)
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_fn=nn.MSELoss()

#training
for epoch in tqdm(range(NUM_EPOCHS)):
    for batch in train_loader:
        images, labels = batch
        images=images.to(Device)

        result = model.forward(images)
        reconst_images= result[0]
        mu=result[1]
        sigma=result[2]
        KL_loss=(sigma**2 +mu**2-torch.log(sigma)-0.5).sum()
        loss=loss_fn(reconst_images,images)-Lambda*KL_loss
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tqdm.set_description(f'loss: {loss}')
    tqdm.update(1)
    




        


        

        
        



