import torch


def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}"
        )

    return model

def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            recon, mu, logvar = model(inputs)

            loss = criterion(
                recon,
                inputs,
                mu,
                logvar
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}"
        )

    print("Finished Training")

    return model

def vae_loss_function(
    recon_x,
    x,
    mu,
    logvar
):
    beta = 500

    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x,
        x,
        reduction="sum"
    )

    KLD = -0.5 * torch.sum(
        1 + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return beta * BCE + KLD



# Train GAN model
def train_gan(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model = model.to(device)

    generator = model.generator
    discriminator = model.discriminator

    # GAN needs separate optimizers for G and D
    if isinstance(optimizer, (tuple, list)):
        optimizer_g, optimizer_d = optimizer
    else:
        optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(
                batch_size, 1, device=device
            )
            fake_labels = torch.zeros(
                batch_size, 1, device=device
            )

            # Train Discriminator
            optimizer_d.zero_grad()

            real_output = discriminator(real_images)
            real_loss = criterion(
                real_output, real_labels
            )

            noise = torch.randn(
                batch_size, 100, device=device
            )
            fake_images = generator(noise)

            fake_output = discriminator(
                fake_images.detach()
            )
            fake_loss = criterion(
                fake_output, fake_labels
            )

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            noise = torch.randn(
                batch_size, 100, device=device
            )
            generated_images = generator(noise)

            output = discriminator(generated_images)
            g_loss = criterion(output, real_labels)

            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(data_loader)
        avg_g_loss = total_g_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"D Loss: {avg_d_loss:.4f}, "
            f"G Loss: {avg_g_loss:.4f}"
        )

    print("Finished GAN Training")

    return model
