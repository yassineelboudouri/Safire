""" Training of ProGAN using WGAN-GP loss"""
import sys
import tqdm
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import ImageDataset
from statistics import TrainingStatistics
from utils import compute_gradient_penalty
from models import Discriminator, Generator


def train_on_batch(dataloader, generator, discriminator, alpha):
    stats = TrainingStatistics()
    pbar = tqdm.tqdm(unit="batch", file=sys.stdout, total=len(dataloader))

    for batch_idx, imgs in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # Alternate between optimizing the generator and discriminator on a per-minibatch basis

        if batch_idx % 2 == 0:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1).to(device)

            # Generate a batch of images
            fake_imgs = generator(z, alpha, i).detach()

            real_validity = discriminator(real_imgs, alpha, i)
            fake_validity = discriminator(fake_imgs, alpha, i)

            # Measure discriminator's ability to classify real from generated samples
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, i)
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
            fake_imgs = generator(z, alpha, i)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs, alpha, i)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            stats.on_training_step({'g_loss': g_loss.item()})

        pbar.set_postfix(stats.get_progbar_postfix())
        pbar.update(1)

    pbar.close()


def convert_img(img_tensor, nrow):
    from PIL import Image
    import torchvision.utils as vutils

    img_tensor = img_tensor.cpu()
    grid = vutils.make_grid(img_tensor, nrow=nrow, padding=2)
    ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr)
    return im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="", help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs of training")
    parser.add_argument("--batch-size", type=int, default=4, help="size of the batches")
    parser.add_argument("--start-size", type=int, default=4, help="Starting image size")
    parser.add_argument("--scale", type=int, default=9, help="Number of times the image size is scaled by 2")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent-dim", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--in-channels", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3, help="Number of image's channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, opt.in_channels, opt.channels).to(device)
    discriminator = Discriminator(opt.in_channels, opt.channels).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for i in range(opt.scale):
        alpha = 1e-5
        target_size = 4 * 2 ** i

        # Configure data loader
        dataloader = torch.utils.data.DataLoader(
            ImageDataset('HRArt', target_size),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

        for epoch in range(opt.epochs):
            print("[Epoch: %d/%d, Image size: %d]" % (epoch + 1, opt.epochs, target_size))
            train_on_batch(dataloader, generator, discriminator, alpha)

            alpha = min(((epoch + 1) / opt.epochs) * 2, 1)

            # a batch of images here
            noise_batch = torch.FloatTensor(8, opt.latent_dim, 1, 1).normal_(0, 1)
            noise_batch = Variable(noise_batch)
            fake_batch = generator(noise_batch, alpha, i)
            im = convert_img(fake_batch.data, 4)
            im.save('log/samples/size-%d-epoch-%d.png' % (target_size, epoch + 1))
