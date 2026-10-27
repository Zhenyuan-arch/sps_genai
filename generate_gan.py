
import torch

from helper_lib.model import get_model
from helper_lib.generator import generate_gan_samples


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = get_model("GAN")

    state_dict = torch.load(
        "outputs/gan_generator.pth",
        map_location=device,
        weights_only=True
    )

    model.generator.load_state_dict(state_dict)

    generate_gan_samples(
        model,
        device=device,
        num_samples=16,
        save_path="outputs/gan_samples.png"
    )

    print("GAN sample images saved successfully!")


if __name__ == "__main__":
    main()
