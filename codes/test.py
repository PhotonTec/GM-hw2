import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import model_withlabel
import os
import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
nz = 100  # Size of generator input noise vector
ngf = 64  # Number of generator feature maps
nc = 3    # Number of channels in generated images
batch_size = 64
num_classes = 10

# Create the generator model
generator = model_withlabel.ConditionalGenerator(nz, ngf, nc, num_classes)
generator.to(device)

# Load the trained generator's weights
generator.load_state_dict(torch.load('checkpoints_withlabel/generator_epoch_100.pth'))
generator.eval()

# Load the CIFAR-10 data for evaluation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Generate images of a specific category
desired_category = 1  # Change this to your desired category
desired_category_labels = torch.full((batch_size,), desired_category, dtype=torch.long).to(device)
mean = 0
std = 0.2  # Smaller standard deviation for higher quality noise
noise = torch.normal(mean=mean, std=std, size=(batch_size, nz, 1, 1)).to(device)
# Concatenate the desired category labels to the noise vector
# noise_with_labels = torch.cat((noise, desired_category_labels.view(batch_size, 1, 1, 1).float()), dim=1)
fake_images = generator(noise, desired_category_labels)

# Save generated images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)
vutils.save_image(fake_images, os.path.join(output_dir, f'generated_category_{desired_category}.png'), normalize=True, nrow=8)

# Calculate FID score
def calculate_fid(real_images, generated_images, batch_size, device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)

    # Compute real image statistics
    real_images = real_images.to(device)  # Move real_images to the same device
    real_features = inception_model(real_images)[0].view(real_images.shape[0], -1)
    mu_real, sigma_real = torch.mean(real_features, dim=0).to(device), torch_cov(real_features, rowvar=False).to(device)

    # Compute generated image statistics
    generated_images = generated_images.to(device)  # Move generated_images to the same device
    generated_features = inception_model(generated_images)[0].view(generated_images.shape[0], -1)
    mu_generated, sigma_generated = torch.mean(generated_features, dim=0).to(device), torch_cov(generated_features, rowvar=False).to(device)

    # Compute the FID score
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)

    return fid_score


# Function to compute covariance matrix
def torch_cov(m, rowvar=False):
    m = m - m.mean(0, keepdim=True)
    if not rowvar:
        m = m.t()
    return m.mm(m.t()) / (m.size(0) - 1)

# Function to compute the Fréchet distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmat1 = sigma1.cpu().numpy()
    covmat2 = sigma2.cpu().numpy()
    assert covmat1.shape == covmat2.shape
    covmean, _ = sqrtm(covmat1.dot(covmat2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (mu1 - mu2).dot(mu1 - mu2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# Calculate the FID score
fid_score = calculate_fid(dataloader, fake_images, batch_size, device)
print("FID Score:", fid_score)
