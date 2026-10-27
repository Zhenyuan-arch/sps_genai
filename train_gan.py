
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.data_loader import get_data_loader


def main():
    # Choose device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load MNIST dataset
    train_loader = get_data_loader(
        data_dir="./data",
        batch_size=64,
        train=True,
        dataset_name="MNIST"
    )

    # Create GAN
    model = get_model("GAN")

    # Loss function
    criterion = nn.BCELoss()

    # Separate optimizers
    optimizer_g = optim.Adam(
        model.generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )

    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )

    # Train GAN
    model = train_gan(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=(optimizer_g, optimizer_d),
        device=device,
        epochs=10
    )

    # Save trained model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    torch.save(
        model.generator.state_dict(),
        output_dir / "gan_generator.pth"
    )

    torch.save(
        model.discriminator.state_dict(),
        output_dir / "gan_discriminator.pth"
    )

    print("GAN models saved successfully!")


if __name__ == "__main__":
    main()
