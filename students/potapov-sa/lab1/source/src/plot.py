import numpy as np
from matplotlib import pyplot as plt

from .linear_classificator import LinearClassificator
from .dataset import DataSample


def plot_margins(model: LinearClassificator, sample: DataSample):
    sorted_margins: list[np.float64] = sorted(list(model.calc_margins(sample.X_1.to_numpy(), sample.Y.to_numpy())))
    plot_x = np.array([i for i in range(0, len(sorted_margins))])

    for x, m in zip(plot_x, sorted_margins):
        if m <= -0.3:
            color = 'r'
        elif m >= 0.5:
            color = 'g'
        else:
            color = 'y'
        plt.plot([x, x], [m, 0], color, linewidth=0.5)
    plt.plot([plot_x[0], plot_x[-1]], [0, 0], 'gray')

    plt.plot(plot_x, sorted_margins)

    x_delta = plot_x[-1] - plot_x[0]
    m_delta = sorted_margins[-1] - sorted_margins[0]
    plt.axis([
        plot_x[0] - round(x_delta * 0.125),
        plot_x[-1] + round(x_delta * 0.025),
        sorted_margins[0] - round(m_delta * 0.05),
        sorted_margins[-1] + round(m_delta * 0.05)
    ]) # type: ignore

    # plt.xticks(np.linspace(plot_x[0], plot_x[-1] + 1, x_delta // 20))
    # plt.yticks(np.linspace(sorted_margins[0], sorted_margins[-1], x_delta // 20))
    # plt.minorticks_on()

    plt.show()
