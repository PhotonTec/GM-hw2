import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import model_withlabel

os.makedirs('checkpoints_withlabel', exist_ok=True)

# Set hyperparameters
batch_size = 64
num_epochs = 100
nz = 100
ngf = 64
ndf = 64
nc = 3  # Number of channels in CIFAR-10 images
num_classes = 10  # Number of classes in CIFAR-10 dataset

# Create the generator and discriminator models
generator = model_withlabel.ConditionalGenerator(nz, ngf, nc, num_classes)
discriminator = model_withlabel.ConditionalDiscriminator(nc, ndf, num_classes)

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
        real_data, real_labels = data
        real_data = real_data.to(device)
        real_labels = real_labels.to(device)

        batch_size = real_data.size(0)
        label_real = torch.full((batch_size,), 1.0, device=device)
        label_fake = torch.full((batch_size,), 0.0, device=device)

        optimizer_d.zero_grad()  # Clear gradients for the discriminator

        # In the discriminator forward pass, pass the real_labels as an argument
        output = discriminator(real_data, real_labels).view(-1)
        loss_d_real = criterion(output, label_real)
        loss_d_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_labels = torch.randint(0, num_classes, size=(batch_size,), device=device)  # Generate fake labels
        fake_data = generator(noise, fake_labels)

        # In the discriminator forward pass, pass the fake_labels as an argument
        output = discriminator(fake_data.detach(), fake_labels).view(-1)
        loss_d_fake = criterion(output, label_fake)
        loss_d_fake.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()  # Clear gradients for the generator

        label_real.fill_(1.0)  # Real label for generator loss
        output = discriminator(fake_data, fake_labels).view(-1)
        loss_g = criterion(output, label_real)
        loss_g.backward()
        optimizer_g.step()

        # Print current loss
        data_iterator.set_postfix(loss_g=f'{loss_g.item():.4f}', loss_d_real=f'{loss_d_real.item():.4f}', loss_d_fake=f'{loss_d_fake.item():.4f}')

    if (epoch + 1) % 10 == 0:
        generator_path = os.path.join('checkpoints_withlabel', f'generator_epoch_{epoch + 1}.pth')
        discriminator_path = os.path.join('checkpoints_withlabel', f'discriminator_epoch_{epoch + 1}.pth')
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
