import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import model

os.makedirs('checkpoints', exist_ok=True)

# Set hyperparameters
batch_size = 64
num_epochs = 100
nz = 100
ngf = 64
ndf = 64
nc = 3  # Number of channels in CIFAR-10 images


# Create the generator and discriminator models
generator = model.Generator(nz, ngf, nc)
discriminator = model.Discriminator(nc, ndf)

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load CIFAR-10 data using PyTorch DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Training loop
for epoch in range(num_epochs):
    data_iterator = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Epoch {epoch + 1}', dynamic_ncols=True)
    for i, data in data_iterator:
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label_real = torch.full((batch_size,), 1.0, device=device)
        label_fake = torch.full((batch_size,), 0.0, device=device)

        optimizer_d.zero_grad()  # Clear gradients for the discriminator

        output = discriminator(real_data).view(-1)
        loss_d_real = criterion(output, label_real)
        loss_d_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = generator(noise)

        output = discriminator(fake_data.detach()).view(-1)  # Detach the fake_data
        loss_d_fake = criterion(output, label_fake)
        loss_d_fake.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()  # Clear gradients for the generator

        label_real.fill_(1.0)  # Real label for generator loss
        output = discriminator(fake_data).view(-1)
        loss_g = criterion(output, label_real)
        loss_g.backward()
        optimizer_g.step()
        
        # print(f'Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(dataloader)}]')
        
        
    if (epoch + 1) % 10 == 0:
        generator_path = os.path.join('checkpoints', f'generator_epoch_{epoch + 1}.pth')
        discriminator_path = os.path.join('checkpoints', f'discriminator_epoch_{epoch + 1}.pth')
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)