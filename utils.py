import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib
import debug
importlib.reload(debug)
from debug import debug

torch.autograd.set_grad_enabled(False)

red = torch.Tensor([1, 0, 0])
green = torch.Tensor([0, 1, 0])
blue = torch.Tensor([0, 0, 1])

black = torch.Tensor([0, 0, 0])
gray = torch.Tensor([0.5, 0.5, 0.5])
white = torch.Tensor([1, 1, 1])

cyan = torch.Tensor([0, 1, 1])
yellow = torch.Tensor([1, 1, 0])
magenta = torch.Tensor([1, 0, 1])

purple = magenta * 0.5
orange = 0.5 * yellow + 0.5 * red


def calc_strengths(x, anchors=None):
    N = len(anchors)
    ancs_pad = F.pad(anchors, pad=(1, 1), value=1)
    ancs_pad[0] = 0
    s = (anchors.reshape(N, 1) - x).abs_().clamp_(min=0, max=1)
    mask = (ancs_pad[:-2].reshape(N, 1) <= x.reshape(1, -1)) * (x.reshape(1, -1) <= ancs_pad[2:].reshape(N, 1))
    mask_int = (anchors[:-1].reshape(1, -1) <= x.reshape(-1, 1)) * (x.reshape(-1, 1) <= anchors[1:].reshape(1, -1))
    scales = mask_int.float() @ (anchors[1:] - anchors[:-1])
    s = (1 - s / scales.unsqueeze(0)).clamp(min=0, max=1) * mask
    return s


def mix(colors, weights):
    sum_w = sum(weights)
    return sum([c * w / sum_w for c, w in zip(colors, weights)])


class Cmap(nn.Module):
    def __init__(self, *args, colors=None, anchors=None, mix=None, exponent=1):
        super().__init__()
        if colors is None:
            colors = args
        cvals = torch.stack(colors)

        mix = mix or [1] * len(cvals)
        cvals_pad = F.pad(cvals.T.reshape(1, 3, -1), (1, 1), mode='reflect').squeeze().T
        for i in range(len(cvals)):
            cvals[i] = cvals_pad[i + 1] * mix[i] + (1 - mix[i]) / 2 * cvals_pad[i] + (1 - mix[i]) / 2 * cvals_pad[i + 2]

        self.register_buffer('cvals', cvals)
        self.exponent = exponent

        N = len(self.cvals)
        if anchors is None:
            anchors = torch.Tensor([n / (N - 1) for n in range(N)])
            self.equidistant_anchors = True
        else:
            self.equidistant_anchors = False
        self.register_buffer('anchors', torch.as_tensor(anchors).reshape(N, 1))

    def __call__(self, x):
        assert 0 <= x.min() and x.max() <= 1, 'invalid range'

        shape = x.shape
        x = x.flatten()

        if self.equidistant_anchors:
            s = (1 - (self.anchors - x).abs_().mul_(len(self.cvals) - 1).clamp_(min=0, max=1))
        else:
            s = calc_strengths(x, self.anchors)

        # for i in range(len(s)):
        #     plt.plot(x, s[i], color=self.cvals[i].numpy())
        # plt.gca().set_xlim(0, 1)
        # plt.gca().set_aspect('equal')
        # plt.xlim(0, 1)
        # plt.show()

        if self.exponent != 1:
            s = s ** self.exponent
        s /= s.sum(dim=0, keepdim=True)

        rgb = torch.tensordot(self.cvals, s, dims=([0], [0])).reshape((3,) + tuple(shape))
        return rgb.clamp_(min=0, max=1)


cmaps = {}
cmaps['bgr'] = Cmap(blue, green, red)
# cmaps['bgr'] = Cmap(blue, green, red, yellow, anchors=[0, 0.3, 0.5, 1])
cmaps['jet'] = Cmap(blue, cyan, yellow, red)
cmaps['viridis'] = Cmap(0.2 * blue + 0.8 * purple, cyan * 0.6, 0.7 * yellow + 0.3 * orange)
cmaps['viridis'] = Cmap(0.2 * blue + 0.8 * purple, 0.7 * yellow + 0.3 * orange)
cmaps['plasma'] = Cmap(0.8 * blue, magenta * 0.5 + 0.2 * red, 0.6 * yellow + 0.3 * orange + 0.1 * green)
cmaps['coolwarm'] = Cmap(0.7 * blue, cyan, 0.8 * white, orange, 0.7 * red, mix=[1, 0.4, 0.9, 0.6, 1])
cmaps['spectral'] = Cmap(purple, magenta, red, yellow, green, cyan, blue,
                         purple, mix=[1, 0.4, 0.7, 0.75, 0.4, 0.7, 0.7, 1])
cmaps['spectral_gray'] = Cmap(purple, red, orange, yellow, gray, green, cyan, blue, purple,
                              mix=[1, 0.8, 0.5, 0.5, 1, 0.4, 0.5, 0.7, 1])
cmaps['gray_spectral'] = Cmap(gray, green, cyan, blue, purple, red, orange, yellow,
                              mix=[1, 0.5, 0.8, 0.6, 1, 0.4, 0.5, 0.7])
cmaps['mystic'] = Cmap(black, yellow, cyan, purple, red, orange, yellow,
                       mix=[1, 0.7, 1, 1, 1, 1, 1])
cmaps['gue'] = Cmap(black, cyan * 0.8 + 0.1 * blue)
cmaps['wave'] = Cmap(purple, yellow, red)


def prepare_tensor(x):
    # x = torch.as_tensor(x, dtype=torch.float).squeeze()
    x = x.squeeze()
    W, H = x.shape[-2:]
    x = x.reshape((1, -1, H, W))
    assert x.shape[1] < W, "likely using numpy format (H, W, C)"
    return x


register_fx = []


def fx_to_device(device):
    for fx in register_fx:
        fx.to(device)


class Conv2d(nn.Module):
    def __init__(self, kernel_weight):
        super().__init__()
        kernel_size = len(kernel_weight)
        padding = (kernel_size - 1) // 2
        self.pad = (padding,) * 4
        self.register_buffer('weight', torch.Tensor(kernel_weight).float().reshape((1, 1, kernel_size, kernel_size)))
        register_fx.append(self)

    def forward(self, x, clamp=None, exponent=1, invert=False, symmetric=False, repeat=1):
        shape = x.shape
        out = prepare_tensor(x)
        n_ch = out.shape[1]
        if invert:
            out = 1 - out
        if symmetric:
            out = out * 2 - 1
        for _ in range(repeat):
            out_padded = F.pad(out, pad=self.pad, mode='replicate')
            out = torch.cat([F.conv2d(out_padded[:, c, :, :].unsqueeze_(0), weight=self.weight)
                            for c in range(n_ch)], dim=1)
            if clamp is not None:
                out = out.clamp_(min=clamp[0], max=clamp[1])
            if exponent != 1:
                out = out.abs_() ** exponent * out.sign()
        if symmetric:
            out = (out + 1) / 2
        elif invert:
            out = 1 - out
        return out.reshape(shape)


highlightN = Conv2d(torch.Tensor([
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0],
]) / 2)
highlightW = Conv2d(torch.Tensor([
    [0, 0, 0],
    [1, 0, -1],
    [0, 0, 0],
]) / 2)

highlightNW = Conv2d(torch.Tensor([
    [1, 1, 0],
    [1, 0, -1],
    [0, -1, -1],
]) / 6)

highlightNW = Conv2d(torch.Tensor([
    [1, 2, 4, 6, 4],
    [2, 3, 6, 0, -6],
    [4, 6, 0, -6, -4],
    [6, 0, -6, -3, -2],
    [-4, -6, -4, -2, -1],
]) / 68)

sharp = Conv2d([
    [0, 1, 0],
    [1, -3, 1],
    [0, 1, 0],
])

laplace2d = Conv2d([
    [.05, .2, .05],
    [.2, -1, .2],
    [.05, .2, .05],
])

edge_det = Conv2d([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
])

gaussian_blur = Conv2d(torch.Tensor([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 32, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
]) / 256)

blur = Conv2d(torch.Tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]) / 16)

thin = Conv2d(torch.Tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
]) / 16)


def laplace(u):
    u_pad = F.pad(u.unsqueeze(0).unsqueeze_(0), (1, 1, 1, 1), mode='replicate').squeeze_()
    u = - 4 * u + u_pad[2:, 1:-1] + u_pad[1:-1, 2:] + u_pad[:-2, 1:-1] + u_pad[1:-1, :-2]
    u[:, 0] = u[0, :] = u[:, -1] = u[-1, :] = 0
    return u


dev = False


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


# @debug


# @debug
def apply_lighting(x, color, amb_color=None, amb_amount=0):

    # imshow(color)

    amb = spec = highlightNW(x, symmetric=True) * 3 / 2.9
    amb = smoothstep(2 * amb - 1, slope=2)
    amb = gaussian_blur(amb, repeat=4)
    # amb = amb.abs() ** 1.5 * amb.sign()
    # amb = (amb > 0) * smoothstep(amb.abs() ** 1, slope=1) ** 1.5 + amb * (amb < 0)
    amb = (amb > 0) * amb.abs() ** 1.5 + amb * (amb < 0)
    # imshow(amb)

    spec = (2 * spec - 1)
    spec = (spec > 0) * spec
    spec = spec ** 1.5
    # spec = smoothstep(spec, slope=1.5)
    # spec = 0.3 * spec + gaussian_blur(spec, repeat=1) * 0.7
    spec = gaussian_blur(spec, repeat=2)
    # spec = 0.3 * spec + gaussian_blur(spec, repeat=2) * 0.7
    # spec = gaussian_blur(spec, repeat=1)
    spec = translate(spec, (1, 1))
    spec = spec * 0.4
    # spec = spec * 0
    spec_mask = smoothstep(1 - x, slope=1.5)
    spec = spec_mask * spec
    imshow(spec)

    outline = edge_det(x)
    outline = gaussian_blur(outline, repeat=4) * 3
    # imshow(outline)

    if amb_color is None:
        amb_color = [1, 1, 1]
    amb_color = torch.Tensor(amb_color, device=x.device).reshape(3, 1, 1)

    color = (color
             + amb_amount * amb_color * (outline + 1) / 2
             + amb_amount * amb_color * (amb + 1) / 2)
    color = color.clamp(0, 1)
    color = apply_spec(color, outline)
    color = apply_spec(color, amb)
    color = apply_spec(color, spec)

    imshow(color)

    return color


def apply_spec(rgb, specular, th=0):
    specular = specular.unsqueeze(0)
    spec_up = (specular > th) * specular.abs()
    spec_down = (-specular > th) * specular.abs()
    for _ in range(3):
        rgb = rgb ** (1 / (1 + spec_up))
    rgb = rgb ** (1 + spec_down)
    return rgb


def translate(x, shifts):
    rolled = x.roll(shifts=shifts, dims=(0, 1))
    for s in range(shifts[0]):
        rolled[s, :] = x[s, :]
    for s in range(shifts[1]):
        rolled[:, s] = x[:, s]
    return rolled


def smoothclamp(x, a=0, b=1):
    return a * (x <= a) + (b - a) * (3 * x**2 - 2 * x**3) * (x > a) * (x < b) + b * (x >= b)


def smoothstep(x, slope=1, alpha=1):
    return smoothclamp((x - 0.5) * slope + 0.5) ** alpha * (x > 0) - smoothclamp((-x - 0.5) * slope + 0.5) ** alpha * (x < 0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 10
    cmap = cmaps['mystic']
    # plt.imshow(cmap(torch.linspace(0, 1, steps=N).reshape(1, N)).expand(3, N, N).permute(1, 2, 0))
    # plt.show()
    # img = (smoothstep(torch.linspace(0, 1, steps=N), slope=2).reshape(1, N)).expand(N, N)
    # plt.imshow(img)
    # plt.show()
    # img = blur(img, repeat=1)
    # plt.imshow(img)
    # plt.show()
    img = cmap(torch.linspace(0, 1, steps=N).reshape(1, N)).expand(3, N, N).permute(1, 2, 0)
    plt.imshow(img)
    plt.show()
    img = blur(img.permute(2, 0, 1), repeat=10).permute(1, 2, 0)
    plt.imshow(img)
    plt.show()

    for name, cmap in cmaps.items():
        rgb = cmap(torch.linspace(0, 1, steps=N).reshape(1, N)).expand(3, N, N)
        plt.title(name)
        plt.imshow(rgb.permute(1, 2, 0))
        plt.show()
