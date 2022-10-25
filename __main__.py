""" Training of GAN using WGAN-GP loss"""
import os.path
import sys
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import ImageDataset
from statistics import TrainingStatistics
from utils import compute_gradient_penalty, save_images
from models import Discriminator, Generator


def save_sample(file):
    # a batch of images here
    noise_batch = torch.randn(8, opt.latent_dim)
    fake_batch = generator(noise_batch)
    save_images(fake_batch.data, file)


def train_on_batch(dataloader, generator, discriminator, device):
    stats = TrainingStatistics()
    pbar = tqdm.tqdm(unit="batch", file=sys.stdout, total=len(dataloader))

    for batch_idx, imgs in enumerate(dataloader):
        # Configure input
        real_imgs = imgs.type(Tensor)
        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

        # Alternate between optimizing the generator and discriminator on a per-batch basis
        if batch_idx % 2 == 0:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Generate a batch of images
            fake_imgs = generator(z).detach()

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)

            # Measure discriminator's ability to classify real from generated samples
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            d_loss = 10 * gradient_penalty \
                     - torch.mean(real_validity) \
                     + torch.mean(fake_validity) \
                     + 0.001 * torch.mean(real_validity ** 2)

            d_loss.backward()
            optimizer_D.step()

            stats.on_training_step({'d_loss': d_loss.item()})

        else:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            stats.on_training_step({'g_loss': g_loss.item()})

        pbar.set_postfix(stats.get_progbar_postfix())
        pbar.update(1)

    pbar.close()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="", help="Path to the dataset")
    parser.add_argument("--log-folder", type=str, default="out", help="Path where to save log files")
    parser.add_argument("--target-size", type=int, default=128, help="Size to resize input images")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch-size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam: learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3)

    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.log_folder):
        os.mkdir(opt.log_folder)
        os.mkdir(opt.log_folder + '/checkpoints')
        os.mkdir(opt.log_folder + '/samples')

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, opt.channels).to(device)
    discriminator = Discriminator(opt.channels).to(device)

    summary(generator, (opt.latent_dim,))
    summary(discriminator, (3, 128, 128))

    # generator.load_state_dict(torch.load('out/checkpoints/generator_9.pth'))
    # discriminator.load_state_dict(torch.load('out/checkpoints/discriminator_9.pth'))

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0, 0.99))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0, 0.99))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        ImageDataset(opt.dataset_path, opt.target_size),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    for epoch in range(opt.epochs):
        print("[Epoch: %d/%d]" % (epoch + 1, opt.epochs))
        train_on_batch(dataloader, generator, discriminator, device)

        save_sample('%s/samples/size-%d-epoch-%d.png' % (opt.log_folder, opt.target_size, epoch + 1))

        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(opt.log_folder, "checkpoints/generator_%d.pth" % epoch))
        torch.save(discriminator.state_dict(), os.path.join(opt.log_folder, "checkpoints/discriminator_%d.pth" % epoch))

