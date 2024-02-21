from model import variationalAutoEncoder
import torch
import torch.nn as n
import torch.nn.functional as f
# Use trained model
mu=[]
sigma=[]
for i in range(32):
    gaussian_mu=torch.randn()
    mu.append(gaussian_mu)
    gaussian_sigma=torch.randn()
    sigma.append(0.1*gaussian_sigma)
mu=torch.ToTensor(mu)
sigma=torch.Tensor(sigma)
model=variationalAutoEncoder()
new_image=model.decode(mu+sigma).cpu()
print("Image shape: ", new_image.shape)    