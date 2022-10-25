import torch
import torch.autograd as autograd
from torch.autograd import Variable

import visdom
import torchvision.utils as vutils
import torchvision.transforms as Transforms

import numpy as np


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def resize_tensor(data):
    out_data_size = (data.size()[0], data.size()[1], data.size()[2], data.size()[3])

    outdata = torch.empty(out_data_size)
    data = torch.clamp(data, min=-1, max=1)

    transform = Transforms.Compose([Transforms.Normalize((-1., -1., -1.), (2, 2, 2)),
                                    Transforms.ToPILImage(),
                                    Transforms.Resize((data.size()[2], data.size()[3])),
                                    Transforms.ToTensor()])

    for img in range(out_data_size[0]):
        outdata[img] = transform(data[img])

    return outdata


def publish_tensors(data, caption="", window_token=None, env="main", nrow=16):
    vis = visdom.Visdom()
    outdata = resize_tensor(data)
    return vis.images(outdata, opts=dict(caption=caption), win=window_token, env=env, nrow=nrow)


def save_images(data, path):
    outdata = resize_tensor(data)
    vutils.save_image(outdata, path)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
