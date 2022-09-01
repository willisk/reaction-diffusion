# %%
# ====================================
# Simulation
# ====================================
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
    # A = torch.ones_like(X) + torch.rand_like(X) * 0.1
    # d = (X - 0.5)**2 + (Y - 0.5)**2
    # B = (torch.clamp(0.001 - d, min=0) > 0).float() * \
    #     0.4 + torch.rand_like(X) * 0.1
    A = torch.rand_like(X)
    B = torch.rand_like(X)
    C = torch.rand_like(X)
    return A, B, C


# def initial_step(u, C):
#     return u - laplace(u) * C**2 / 2


T = 1000
dt = 1
n_grid = 512

reduce_res = 1
skip_frames = 1

n_steps = int(T / dt)
x = torch.linspace(0, 1, n_grid)
dx = x[1] - x[0]
mesh = torch.meshgrid(x, x)

mode = 'replicate'


def avg5(u):
    u_pad = F.pad(u.unsqueeze(0).unsqueeze_(
        0), (1, 1, 1, 1), mode=mode).squeeze_()
    u = (u +
         u_pad[2:, 1:-1] +
         u_pad[:-2, 1:-1] +
         u_pad[1:-1, 2:] +
         u_pad[1:-1, :-2]
         )
    return u / 5


def avg9(u):
    u_pad = F.pad(u.unsqueeze(0).unsqueeze_(
        0), (1, 1, 1, 1), mode=mode).squeeze_()
    u = (u +
         u_pad[2:, 2:] +
         u_pad[2:, :-2] +
         u_pad[:-2, 2:] +
         u_pad[:-2, :-2] +
         u_pad[2:, 1:-1] +
         u_pad[:-2, 1:-1] +
         u_pad[1:-1, 2:] +
         u_pad[1:-1, :-2]
         )
    # u[:, 0] = u[0, :] = u[:, -1] = u[-1, :] = 0
    return u / 9


def laplace5(u):
    u_pad = F.pad(u.unsqueeze(0).unsqueeze_(
        0), (1, 1, 1, 1), mode=mode).squeeze_()
    u = - 4 * u + u_pad[2:, 1:-1] + u_pad[1:-1, 2:] + \
        u_pad[:-2, 1:-1] + u_pad[1:-1, :-2]
    # u[:, 0] = u[0, :] = u[:, -1] = u[-1, :] = 0
    return u / 5


def laplace9(u):
    u_pad = F.pad(u.unsqueeze(0).unsqueeze_(
        0), (1, 1, 1, 1), mode=mode).squeeze_()
    u = (- 8 * u +
         u_pad[2:, 2:] +
         u_pad[2:, :-2] +
         u_pad[:-2, 2:] +
         u_pad[:-2, :-2] +
         u_pad[2:, 1:-1] +
         u_pad[:-2, 1:-1] +
         u_pad[1:-1, 2:] +
         u_pad[1:-1, :-2]
         )
    # u[:, 0] = u[0, :] = u[:, -1] = u[-1, :] = 0
    return u / 9


def step(A, B, C):
    avg_A = avg5(A)
    avg_B = avg5(B)
    avg_C = avg5(C)
    A += (laplace5(A) + avg_A * (avg_B - avg_C)) * dt
    B += (laplace5(B) + avg_B * (avg_C - avg_A)) * dt
    C += (laplace5(C) + avg_C * (avg_A - avg_B)) * dt
    # A = avg_A * (1 + (avg_B - avg_C))
    # B = avg_B * (1 + (avg_C - avg_A))
    # C = avg_C * (1 + (avg_A - avg_B))
    A.clamp_(min=0, max=1)
    B.clamp_(min=0, max=1)
    C.clamp_(min=0, max=1)
    # A[:, 0] = A[0, :] = A[:, -1] = A[-1, :] = 0
    # B[:, 0] = B[0, :] = B[:, -1] = B[-1, :] = 0
    C[:, 0] = C[0, :] = C[:, -1] = C[-1, :] = 1
    return A, B, C

# %%


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mesh = mesh[0].to(device), mesh[1].to(device)

os.makedirs('figures/3c_reaction_diffusion', exist_ok=True)

A, B, C = I(*mesh)

for t in tqdm.trange(n_steps, desc='simulating', unit_scale=dt):
    A, B, C = step(A, B, C)

    if t % skip_frames == 0:
        # image = torch.stack([1.3 - 1.3 * A.clamp(min=0, max=1), 0.87 * torch.ones_like(B), 1 - B.clamp(min=0, max=1)])
        # image = (1 - B / 0.35).clamp(min=0, max=1)
        image = torch.stack([A, B, C])
        # if reduce_res != 1:
        #     image = F.avg_pool2d(image.unsqueeze_(0), reduce_res).squeeze_()
        save_image(image, f'figures/3c_reaction_diffusion/t={t}.png')

    if t % 10 == 0:
        assert A.isfinite().all() and B.isfinite().all(), 'NANs detected'

# %%

# ====================================
# Visualization
# ====================================

# dev = False
dev = True

if dev:
    # jump = 10 * skip_frames
    jump = 0
    debug.disable = False
    file_out = '3c_reaction_diffusion.gif'
else:
    jump = 0
    debug.disable = True
    file_out = '3c_reaction_diffusion.mp4'


def postproc(x):
    return x


device = 'cpu'

cmap.to(device)
fx_to_device(device)
colormap = torch.stack([cyan, magenta, yellow])

with imageio.get_writer(file_out, fps=15) as writer:
    for t in tqdm.trange(int(n_steps / skip_frames), desc=f'writing file {file_out}'):
        image = imageio.imread(
            f'figures/3c_reaction_diffusion/t={t * skip_frames + jump}.png')
        # writer.append_data(255 - image)
        writer.append_data(image)
        continue

        # note: applying color maps did not look better on this experiment

        image = torch.Tensor(image.transpose(2, 0, 1) / 255).to(device)
        if dev:
            image = image[:, 200:-200, 200:-200]

        # colored = torch.einsum('kc,chw->khw', [colormap, image])
        colored = image
        # imshow(image)

        # image = 1 - image * 2
        # H, W = image.shape
        # image = 1 - (1 - image) ** 8
        # image = smoothstep(image)
        # image = blur((image > 0.95).astype('float')) ** 5
        # image = thin(image)

        # image = image * 2 - 1

        # colored = image
        # colored = apply_lighting(image, colored, amb_color=[
        #                          1, 0, 0], amb_amount=0.07)
        colored = postproc(colored)

        # image = 1 - (1 - image) ** 4
        # imshow(image)
        # image = image ** 4

        # image = blur(image)
        # image = apply_spec(image, spec)
        # imshow(colored)

        colored = colored.permute(1, 2, 0).cpu().numpy()
        colored = (colored * 255).astype('uint8')
        writer.append_data(colored)
        # if dev and t > n_steps / 4:
        # break

# %%
