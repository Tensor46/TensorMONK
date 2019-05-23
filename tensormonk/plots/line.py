""" TensorMONK :: plots """


def line_plot(data,
              xaxis: str = None,
              yaxis: str = None,
              labels: str = None,
              x_log: bool = False,
              y_log: bool = False,
              x_limit: tuple = None,
              y_limit: tuple = None,
              save_png: str = None,
              show: bool = True):
    r"""Seaborn line plots using pandas DataFrameself.

    Args:
        data (DataFrame, required): A DataFrame
        xaxis (str, required): column name in the DataFrame
            example: data["iterations"] = [0, 1, 2, 3]
        yaxis (str, required): column name in the DataFrame
            example: data["loss"] = [1.0, 0.96, 0.97, 0.86]
        labels (str, required): column name in the DataFrame
            example: data["labels"] = ["center", "center", "center", "center"]
        x_log (bool, optional): converts x-axis to log scale, default=False
        y_log (bool, optional): converts y-axis to log scale, default=False
        x_limit (tuple, optional): limits x-axis
        y_limit (tuple, optional): limits y-axis
        save_png (str, optional): save the plot to png file, default=None
        show (bool, optional): returns the plot when True, default=True

    Return:
        matplotlib's pyplot
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError("line_plots: data must be DataFrame: "
                        "{}".format(type(data).__name__))
    if not isinstance(xaxis, str):
        raise TypeError("line_plots: xaxis must be string: "
                        "{}".format(type(xaxis).__name__))
    if not isinstance(yaxis, str):
        raise TypeError("line_plots: yaxis must be string: "
                        "{}".format(type(yaxis).__name__))
    if not isinstance(labels, str):
        raise TypeError("line_plots: labels must be string: "
                        "{}".format(type(labels).__name__))

    sns.set_style("whitegrid")
    f, ax = plt.subplots()
    ax = sns.lineplot(x=xaxis, y=yaxis, data=data,
                      palette="muted",
                      hue=labels, style=labels)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if x_limit is not None:
        ax.set_xlim(*x_limit)
    if y_limit is not None:
        ax.set_ylim(*y_limit)

    f.tight_layout()
    if isinstance(save_png, str) and save_png.endswith(".png"):
        f.savefig(save_png, format="png", dpi=464)
    if show:
        return plt.show()
    else:
        return None
