# %%
import matplotlib.pyplot as plt

from debug import debug
import os

import tqdm
import torch
from torchvision.utils import save_image

import importlib
import utils
importlib.reload(utils)
from utils import *

torch.autograd.set_grad_enabled(False)
os.makedirs('figures/wave_equation', exist_ok=True)

cmap = cmaps['wave']


def I(X, Y):
    # d = (X - 0.5)**2 + (Y - 0.5)**2
    # return (torch.clamp_(0.01 - d, min=0) > 0).float() * 0.5
    d = (X - Y).abs() * (X + Y - 1).abs()
    # d = (X + Y - 1).abs()
    return (torch.clamp_(0.01 - d, min=0) > 0).float() * 0.5


def initial_step(u, C):
    return u - laplace(u) * C**2 / 2


T = 10
n_grid = 1024
skip_frames = 5
reduce_res = 1
dt = 0.001

# T = 1
# n_grid = 512
# reduce_res = 1
# skip_frames = 10
# dt = 0.002

# T = 0.2
n_steps = int(T / dt)
x = torch.linspace(0, 1, n_grid)
dx = x[1] - x[0]
mesh = torch.meshgrid(x, x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mesh = mesh[0].to(device), mesh[1].to(device)

c = 0.5
C = (c * dt / dx)

U_prev = I(*mesh)
U = initial_step(U_prev, C)


def step(u, u_prev):
    return -u_prev + 2 * u + laplace(u) * C**2

# %%


for t in tqdm.trange(n_steps, desc='simulating', unit_scale=dt):
    U, U_prev = step(U, U_prev), U
    if t % skip_frames == 0:
        image = ((U + 1) / 2).clamp_(min=0, max=1)
        if reduce_res != 1:
            image = F.avg_pool2d(image.unsqueeze_(0), reduce_res).squeeze_()
        save_image(image.cpu(), f'figures/wave_equation/t={t}.png')

    if t % 10 == 0:
        assert U.isfinite().all(), 'NANs detected'

# %%

import tqdm
import imageio

import importlib
import utils
importlib.reload(utils)
from utils import *


def imshow(x):
    if not dev:
        return
    x = x.squeeze()
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    debug(x)
    plt.imshow(x)
    plt.colorbar()
    plt.show()


def apply_lighting(x, color, amb_color=None, amb_amount=0, symmetric=False):

    # imshow(x)

    amb = spec = highlightNW(x, symmetric=True) * 3 / 2.9
    amb = smoothstep(2 * amb - 1, slope=1)
    amb = gaussian_blur(amb, repeat=4) * 5
    # amb = (amb > 0) * amb.abs() ** 1.5 + amb * (amb < 0)

    # # amb = amb.abs() ** 1.5 * amb.sign()
    # # amb = (amb > 0) * smoothstep(amb.abs() ** 1, slope=1) ** 1.5 + amb * (amb < 0)
    # print('AMB')
    # imshow(amb)

    spec = (2 * spec - 1) * 5
    spec = smoothstep(spec * (spec > 0), slope=1, alpha=30) + spec * (spec < 0)
    spec = gaussian_blur(spec, repeat=2) * 0.6
    # print('SPEC')
    # imshow(spec)

    outline = edge_det(x)
    outline = gaussian_blur(outline, repeat=2) * 15
    # print('OUTLINE')
    # imshow(outline)

    if amb_color is None:
        amb_color = [1, 1, 1]
    amb_color = torch.as_tensor(amb_color, device=x.device).reshape(3, 1, 1)

    # print('NO LIGHTING')
    # imshow(color)
    color = (color
             + amb_amount * amb_color * (outline + 1) / 2
             #  + amb_amount * amb_color * (amb + 1) / 2
             )
    color = color.clamp(0, 1)
    color = apply_spec(color, amb, repeat=1)
    color = apply_spec(color, outline, repeat=2)
    color = gaussian_blur(color, repeat=1)
    color = apply_spec(color, spec, repeat=3)
    # color = gaussian_blur(color, repeat=1)

    # imshow(color)

    return color


def apply_spec(rgb, specular, th=0, repeat=1):
    specular = specular.unsqueeze(0)
    spec_up = (specular > th) * specular.abs()
    spec_down = (-specular > th) * specular.abs()
    for _ in range(repeat):
        rgb = rgb ** (1 / (1 + spec_up))
        rgb = rgb ** (1 + spec_down)
    return rgb


dev = False
# dev = True

render_gif = False
render_gif = True

file_out = 'wave_equation.mp4'
if dev:
    file_out = 'wave_equation.gif'


if dev and not render_gif:
    jump_frames = 20
    debug.disable = False
else:
    jump_frames = 0
    debug.disable = True


light_green = 0.5 * green + 0.2 * white
cmap = Cmap(purple, yellow, red)
cmap = Cmap(0.6 * cyan, 0.4 * gray, 0.95 * mix([white, orange, yellow, purple], [1, 8, 5, 20]))
cmap = Cmap(mix([cyan, green], [3, 1]), 0.4 * gray, mix([orange, yellow], [1, 3]))
cmaps['spectral_gray'] = Cmap(purple, red, orange, yellow, 0.8 * gray, green, cyan, blue, purple,
                              mix=[1, 0.8, 0.5, 0.5, 1, 0.4, 0.5, 0.7, 1])
cmap = cmaps['spectral_gray']

cmap.to(device)
fx_to_device(device)

with imageio.get_writer(file_out, mode='I', fps=0.4 / dt / skip_frames) as writer:
    for t in tqdm.trange(int(n_steps / skip_frames), desc='writing file'):
        image = imageio.imread(f'figures/wave_equation/t={(t + jump_frames) * skip_frames}.png')
        image = torch.as_tensor(image[:, :, 0] / 255, device=device, dtype=torch.float)
        # if dev:
        #     image = torch.Tensor(image[200:-200, 200:-200])

        H, W = image.shape
        image = 2 * image - 1
        # imshow(image)
        # image = blur(image / 0.5, repeat=2)
        image = smoothstep(blur(image * 5, repeat=2), slope=1)
        # imshow(image)
        # image = thin(image)

        image = (image + 1) / 2
        image = image.clip_(min=0, max=1)
        colored = cmap(image)
        colored = apply_lighting(image, colored, amb_amount=0.1)
        # imshow(colored)

        colored = colored.permute(1, 2, 0).cpu().numpy()
        colored = (colored * 255).astype('uint8')
        writer.append_data(colored)
        if dev and not render_gif:
            break

if not dev:
    from pygifsicle import optimize
    if file_out.split('.')[-1] == 'gif':
        optimize(file_out)
