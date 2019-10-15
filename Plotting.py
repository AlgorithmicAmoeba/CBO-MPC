import matplotlib.pyplot as plt
import numpy


def plot_all(df, show=True):
    ts = df.ts
    us_names = df.columns[df.columns.str.contains('u')].values.tolist()
    ys_names = df.columns[df.columns.str.contains('y')].values.tolist()
    rs_names = df.columns[df.columns.str.contains('r')].values.tolist()

    us = df[us_names].values
    ys = df[ys_names].values
    rs = df[rs_names].values

    plt.subplot(2, 1, 1)
    plt.plot(ts, ys, '-')
    plt.plot(ts, rs, '--')
    plt.xlim(numpy.min(ts), numpy.max(ts))
    plt.ylim([numpy.min(ys) - numpy.std(ys), numpy.max(ys) + numpy.std(ys)])
    plt.legend([rf"${name}$" for name in ys_names + rs_names])

    plt.subplot(2, 1, 2)
    plt.plot(ts, us, '-')
    plt.xlim(numpy.min(ts), numpy.max(ts))
    plt.ylim([numpy.min(us) - numpy.std(us), numpy.max(us) + numpy.std(us)])
    plt.legend([rf"${name}$" for name in us_names])

    if show:
        plt.show()
