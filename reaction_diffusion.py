# %%
import numpy as np
import imageio
from utils import *
import matplotlib.pyplot as plt

from debug import debug

import os

import tqdm
import torch
from torchvision.utils import save_image

import importlib
import utils
importlib.reload(utils)

torch.autograd.set_grad_enabled(False)
# cmap = cmaps['gray_spectral']
cmap = cmaps['gue']


def I(X, Y):
    A = torch.ones_like(X) + torch.rand_like(X) * 0.1
    d = (X - 0.5)**2 + (Y - 0.5)**2
    B = (torch.clamp(0.001 - d, min=0) > 0).float() * \
        0.4 + torch.rand_like(X) * 0.1
    return A, B


# def initial_step(u, C):
#     return u - laplace(u) * C**2 / 2


rA = 0.16
rB = .08
rF = .055
rK = .062

T = 25100
dt = 1
n_grid = 512

reduce_res = 1
skip_frames = 30

n_steps = int(T / dt)
x = torch.linspace(0, 1, n_grid)
dx = x[1] - x[0]
mesh = torch.meshgrid(x, x)


def step(A, B):
    R = A * B ** 2
    A += (rA * laplace(A) - R + rF * (1 - A)) * dt
    B += (rB * laplace(B) + R - (rK + rF) * B) * dt

# %%


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mesh = mesh[0].to(device), mesh[1].to(device)

os.makedirs('figures/reaction_diffusion', exist_ok=True)

A, B = I(*mesh)

for t in tqdm.trange(n_steps, desc='simulating', unit_scale=dt):
    step(A, B)
    A.clamp_(min=0, max=1)
    B.clamp_(min=0, max=1)

    if t % skip_frames == 0:
        # image = torch.stack([1.3 - 1.3 * A.clamp(min=0, max=1), 0.87 * torch.ones_like(B), 1 - B.clamp(min=0, max=1)])
        image = (1 - B / 0.35).clamp(min=0, max=1)
        if reduce_res != 1:
            image = F.avg_pool2d(image.unsqueeze_(0), reduce_res).squeeze_()
        save_image(image, f'figures/reaction_diffusion/t={t}.png')

    if t % 10 == 0:
        assert A.isfinite().all() and B.isfinite().all(), 'NANs detected'

# %%


dev = False
dev = True

if dev:
    jump = 20 * skip_frames
    debug.disable = False
else:
    jump = 0
    debug.disable = True


def postproc(x):
    return x


file_out = 'reaction_diffusion.gif'

cmap.to(device)
fx_to_device(device)

with imageio.get_writer(file_out, mode='I', fps=1 / dt / skip_frames * 300) as writer:
    for t in tqdm.trange(int(n_steps / skip_frames), desc='writing file'):
        image = imageio.imread(
            f'figures/reaction_diffusion/t={t * skip_frames + jump}.png')

        image = torch.Tensor(image[:, :, 0] / 255)
        if dev:
            image = torch.Tensor(image[200:-200, 200:-200])
        # imshow(image)

        # image = 1 - image * 2
        H, W = image.shape
        # image = 1 - (1 - image) ** 8
        image = smoothstep(image)
        # image = blur((image > 0.95).astype('float')) ** 5
        # image = thin(image)

        # image = image * 2 - 1

        colored = cmap(image)
        colored = apply_lighting(image, colored, amb_color=[
                                 1, 0, 0], amb_amount=0.07)
        colored = postproc(colored)

        # image = 1 - (1 - image) ** 4
        # imshow(image)
        # image = image ** 4

        # image = blur(image)
        # image = apply_spec(image, spec)
        # imshow(colored)

        colored = colored.permute(1, 2, 0).numpy()
        colored = (colored * 255).astype('uint8')
        writer.append_data(colored)
        if dev:
            break
