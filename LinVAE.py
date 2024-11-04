import torch
from torch import nn, Tensor
from dataclasses import dataclass, field
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

@dataclass
class VAEBaseConfig:
    input_dim: int = 512
    latent_dim: int = 6
    layer_dims: list = field(default_factory=lambda: [256, 128])

    input_channels: int = 1


@dataclass
class EncoderConfig(VAEBaseConfig):
    pass


@dataclass
class DecoderConfig(VAEBaseConfig):
    pass


@dataclass
class VAEConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super(Encoder, self).__init__()
        self.config = config

        layers = []
        in_dim = config.input_dim
        print(f"Encoder Input")
        for out_dim in config.layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            print(f"Layer: {in_dim} -> {out_dim}")
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(config.layer_dims[-1], config.latent_dim)
        self.fc_log_var = nn.Linear(config.layer_dims[-1], config.latent_dim)

    def forward(self, x: Tensor):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.net(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return [mu, log_var]


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super(Decoder, self).__init__()
        self.config = config

        layers = []
        in_dim = config.latent_dim
        print(f"Decoder Input")
        for out_dim in config.layer_dims[::-1] + [config.input_dim]:  # Reverse layer dimensions
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            print(f"Layer: {in_dim} -> {out_dim}")
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(
            config.layer_dims[0],
            config.input_dim,
        )

    def forward(self, x: Tensor):
        x = self.net(x)
        x = self.output_layer(x)
        return x


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.config = config

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        y = self.decoder(z)
        return [x, y, z, mu, log_var]

    def calculate_mse_loss(
        self, x: Tensor, y: Tensor, z: Tensor, mu: Tensor, log_var: Tensor
    ) -> Tensor:
        recon_loss = nn.functional.mse_loss(
            y, x.view(x.size(0), -1)
        )  # Flatten the input
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_divergence

    def sample(self, num_samples: int, device) -> Tensor:
        z = torch.randn(num_samples, self.config.encoder.latent_dim, device=device)
        generated_samples = self.decoder(z)
        return generated_samples, z


def train_vae(vae: VAE, train_loader, val_loader, num_epochs: int, device: str):
    optimizer = optim.AdamW(vae.parameters(), lr=3e-3)
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0.0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            x, y, z, mu, log_var = vae(x_batch)

            loss = vae.calculate_mse_loss(x_batch, y, z, mu, log_var)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)

        vae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)

                x, y, z, mu, log_var = vae(x_batch)
                loss = vae.calculate_mse_loss(x_batch, y, z, mu, log_var)
                val_loss += loss.item() * x_batch.size(0)

            val_loss /= len(val_loader.dataset)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/val", val_loss, epoch)

    writer.close()
