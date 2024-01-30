import numpy as np


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(
        loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i], dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(
                warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(
                0, x.shape[1]-1., num=warped.size), warped).T
    return ret


def weak_aug(x):
    return scaling(jitter(x))


def strong_aug(x):
    return window_warp(x.transpose(1, 0, 2)).transpose(1, 0, 2)
