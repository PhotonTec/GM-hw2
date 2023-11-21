import torch
import torch.nn as nn

# Specify the device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Conditional Generator model
class ConditionalGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.num_classes = num_classes

        # Generator architecture with conditional input
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + self.num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        # Move labels and input to the same device
        labels = labels.to(device)
        input = input.to(device)

        # Concatenate labels to the noise vector
        reshaped_labels = torch.zeros(input.size(0), self.num_classes).to(device)
        reshaped_labels.scatter_(1, labels.unsqueeze(1), 1)
        reshaped_labels = reshaped_labels.unsqueeze(-1).unsqueeze(-1)
        reshaped_labels = reshaped_labels.expand(-1, -1, input.size(2), input.size(3))
        input = torch.cat((input, reshaped_labels), dim=1)
        return self.main(input)

# Conditional Discriminator model
class ConditionalDiscriminator(nn.Module):
    def __init__(self, nc, ndf, num_classes):
        super(ConditionalDiscriminator, self).__init()
        self.num_classes = num_classes

        # Discriminator architecture with conditional input
        self.main = nn.Sequential(
            nn.Conv2d(nc + self.num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        # Move labels and input to the same device
        labels = labels.to(device)
        input = input.to(device)

        # Get the embedding for the labels
        embedded_labels = torch.zeros(input.size(0), self.num_classes).to(device)
        embedded_labels.scatter_(1, labels.unsqueeze(1), 1)
        # Expand labels to have the same spatial dimensions as input
        embedded_labels = embedded_labels.unsqueeze(-1).unsqueeze(-1)  # Shape (batch_size, num_classes, 1, 1)
        embedded_labels = embedded_labels.expand(-1, -1, input.size(2), input.size(3))  # Shape (batch_size, num_classes, H, W)
        input = torch.cat((input, embedded_labels), dim=1)
        return self.main(input)
