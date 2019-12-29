import torch
import torchvision
from torchvision.datasets import FashionMNIST, MNIST
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

multiple_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,),(.5,)), torch.flatten])

traindata = MNIST(root='./data', transform=multiple_transforms, train=True, download=True)
batches = 6000

#traindata = [(data,label) for (data,label) in traindata if label==0]
#traindata = traindata[:5000]
#batches = 500

trainloader = torch.utils.data.DataLoader(traindata, batch_size=batches, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        ip_emb = 784
        emb1 = 256
        emb2 = 128
        out_emb = 1
        
        self.layer1 = nn.Sequential(
        nn.Linear(ip_emb, emb1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
            
        self.layer2 = nn.Sequential(
        nn.Linear(emb1, emb2),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3))
        
        self.layer_out = nn.Sequential(
        nn.Linear(emb2, out_emb),
        nn.Sigmoid())
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer_out(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        ip_emb = 128
        emb1 = 256
        emb2 = 512
        emb3 = 1024
        out_emb = 784
        
        self.layer1 = nn.Sequential(
        nn.Linear(ip_emb, emb1),
        nn.LeakyReLU(0.2))
        
        self.layer2 = nn.Sequential(
        nn.Linear(emb1, emb2),
        nn.LeakyReLU(0.2))
        
        self.layer3 = nn.Sequential(
        nn.Linear(emb2, emb3),
        nn.LeakyReLU(0.2))
        
        self.layer_out = nn.Sequential(
        nn.Linear(emb3, out_emb),
        nn.Tanh())
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer_out(x)
        return x

discriminator = nn.DataParallel(Discriminator())
generator = nn.DataParallel(Generator())

criterion = nn.BCELoss()
discriminator_optim = optim.Adam(discriminator.parameters(), lr= 0.0002)
generator_optim = optim.Adam(generator.parameters(), lr=0.0002)

def noise(x,y):
    return torch.randn(x,y)

derrors = []
gerrors = []
dxcumul = []
gxcumul = []

for epoch in range(2000):
    dx = 0
    gx = 0
    derr = 0
    gerr = 0
    for pos_samples in trainloader:
        # Training Discriminator network
        discriminator_optim.zero_grad()
        pos_predicted = discriminator(pos_samples[0])
        pos_error = criterion(pos_predicted, torch.ones(batches,1))
        
        neg_samples = generator(noise(batches, 128))
        neg_predicted = discriminator(neg_samples)
        neg_error = criterion(neg_predicted, torch.zeros(batches,1))
        
        discriminator_error = pos_error + neg_error
        discriminator_error.backward()
        discriminator_optim.step()
        
        # Training generator network
        generator_optim.zero_grad()
        gen_neg_samples = generator(noise(batches, 128))
        gen_neg_predicted = discriminator(gen_neg_samples)
        generator_error = criterion(gen_neg_predicted, torch.ones(batches, 1))
        generator_error.backward()
        generator_optim.step()
        
        derr += discriminator_error
        gerr += generator_error
        dx += pos_predicted.data.mean()
        gx += neg_predicted.data.mean()
        
    print(f'Epoch:{epoch}.. D x : {dx/10:.4f}.. G x: {gx/10:.4f}.. D err : {derr/10:.4f}.. G err: {gerr/10:.4f}')
    torch.save(discriminator, 'discriminator_model.pt')
    torch.save(generator, 'generator_model.pt')
    derrors.append(dx/10)
    gerrors.append(gx/10)
    plt.imshow(generator(noise(1, 128)).detach().view(28,28).numpy(), cmap=cm.gray)
    plt.show()

# Plotting the errors
plt.plot(range(2000),[x.item() for x in derrors], color='r')
plt.plot(range(2000),[y.item() for y in gerrors], color='g')
plt.show()

# Images created by Generator network
for i in range(10):
    plt.imshow(generator(noise(1, 128)).cpu().detach().view(28,28).numpy(), cmap=cm.gray)
    plt.show()