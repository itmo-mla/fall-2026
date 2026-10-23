from typing import Optional
import os

import numpy as np
from matplotlib import pyplot as plt

from .model import Model
from .dataset import DataSample

_SAVE_FIG_DPI = 200


def _get_savefig_path(filename: str, folder: str = 'img', ext: str = 'png') -> str:
    if not folder:
        folder = '.'
    if not ext:
        ext = 'png'
    if ext.startswith('.'):
        ext = ext.lstrip('.')
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not filename.endswith(f'.{ext}'):
        filename = f'{filename}.{ext}'

    return os.path.join(folder, filename)


def plot_margins(model: Model, sample: DataSample, filename: Optional[str] = None):
    margins = model.margins if model.margins is not None else model.calc_margins_sample(sample)
    sorted_margins: list[np.float64] = sorted(list(margins))
    plot_x = np.array([i for i in range(len(sorted_margins))])
    unsure_range: tuple[float, float] = (-0.3, 0.5)

    used_labels: set[str] = set()
    for x, m in zip(plot_x, sorted_margins):
        if m <= unsure_range[0]:
            color, label = 'r', 'noise'
        elif m >= unsure_range[1]:
            color, label = 'g', 'sure'
        else:
            color, label = 'y', 'unsure'
        if label not in used_labels:
            used_labels.add(label)
        else:
            label = None
        plt.plot([x, x], [m, 0], color, linewidth=0.5, label=label)
    plt.plot([plot_x[0], plot_x[-1]], [0, 0], 'gray')
    
    if 'unsure' in used_labels:
        plt.plot(plot_x, [unsure_range[0]] * len(plot_x), color='gray', linestyle='--')
        plt.plot(
            plot_x,
            [unsure_range[1]] * len(plot_x),
            color='gray',
            linestyle='--',
            label=r'$unsure \in ({}, {})$'.format(*(round(x, 2) for x in unsure_range)),
        )

    plt.plot(plot_x, sorted_margins)

    x_delta = plot_x[-1] - plot_x[0]
    m_delta = sorted_margins[-1] - sorted_margins[0]
    plt.axis([ # type: ignore
        plot_x[0] - round(x_delta * 0.125),
        plot_x[-1] + round(x_delta * 0.025),
        sorted_margins[0] - round(m_delta * 0.05),
        sorted_margins[-1] + round(m_delta * 0.05)
    ])

    plt.title(f'{model.name} Sorted Margins ({sample.type})')
    plt.xlabel('i')
    plt.ylabel('margin')
    plt.legend()

    if filename:
        plt.savefig(_get_savefig_path(filename), dpi=_SAVE_FIG_DPI)
    else:
        plt.show()
    plt.close()


def plot_loss_history(model: Model, save_image: bool = True):
    if model.loss_history is None:
        raise Exception('Model must have loss history to plot it')
    train_history, test_history = model.loss_history.train, model.loss_history.test

    if len(train_history) * len(test_history) == 0:
        raise Exception('Both train_history and test_history must be not empty')
    
    plt.plot(range(len(train_history)), train_history, label='train')
    plt.plot(range(len(test_history)), test_history, label='test')
    
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title(f'{(model.name + ' ') if model.name else ''}Loss')
    
    plt.legend()
    if save_image:
        plt.savefig(_get_savefig_path(f'{model.name}_loss'), dpi=_SAVE_FIG_DPI)
    else:
        plt.show()
    plt.close()
