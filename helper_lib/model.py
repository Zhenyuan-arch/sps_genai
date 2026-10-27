import torch
import torch.nn as nn
import torch.nn.functional as F


# Fully Connected Neural Network
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()

        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 32 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Enhanced CNN
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(
            128 * 2 * 2, 128
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )

        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )

        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )

        x = self.pool(
            F.relu(self.bn4(self.conv4(x)))
        )

        x = x.view(-1, 128 * 2 * 2)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x



class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(
            128 * 4 * 4, latent_dim
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4, latent_dim
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(
            2, 128 * 4 * 4
        )

        self.convtrans1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.convtrans2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.convtrans3 = nn.ConvTranspose2d(
            32,
            1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x):
        x = self.fc(x)

        x = x.view(
            -1, 128, 4, 4
        )

        x = F.relu(
            self.convtrans1(x)
        )

        x = F.relu(
            self.convtrans2(x)
        )

        x = torch.sigmoid(
            self.convtrans3(x)
        )

        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=1.0):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(
        self,
        mu,
        logvar
    ):
        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):

        mu, logvar = self.encoder(x)

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

# GAN Generator
class GANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(GANGenerator, self).__init__()

        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        return self.generator(x)


# GAN Discriminator
class GANDiscriminator(nn.Module):
    def __init__(self):
        super(GANDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


# GAN Model
class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super(GAN, self).__init__()

        self.generator = GANGenerator(latent_dim)
        self.discriminator = GANDiscriminator()

    def forward(self, z):
        return self.generator(z)


# GAN Generator
class GANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        return self.generator(x)


# GAN Discriminator
class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


# GAN Model
class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.generator = GANGenerator(latent_dim)
        self.discriminator = GANDiscriminator()

    def forward(self, z):
        return self.generator(z)




def get_model(model_name):
    if model_name == "FCNN":
        model = FCNN()

    elif model_name == "CNN":
        model = SimpleCNN()

    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()

    elif model_name == "GAN":
        model = GAN()

    elif model_name == "VAE":
        encoder = VariationalEncoder()
        decoder = Decoder()
        model = VAE(
            encoder,
            decoder
        )

    else:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    return model