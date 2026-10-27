import torch
import matplotlib.pyplot as plt
import math


def generate_samples(model, device="cpu", num_samples=10):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, 2).to(device)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()

    cols = 5
    rows = math.ceil(num_samples / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 2 * rows)
    )

    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(
            samples[i].squeeze(),
            cmap="gray"
        )
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Generate MNIST samples from a trained GAN
def generate_gan_samples(model, device="cpu", num_samples=16,
                         save_path="outputs/gan_samples.png"):
    import torch
    import matplotlib.pyplot as plt
    from pathlib import Path

    generator = model.generator.to(device)
    generator.eval()

    noise = torch.randn(num_samples, 100, device=device)

    with torch.no_grad():
        images = generator(noise).cpu()

    # Convert from [-1, 1] to [0, 1]
    images = (images + 1) / 2

    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < num_samples:
            axes[i].imshow(images[i, 0], cmap="gray",
                           vmin=0, vmax=1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

    return images
