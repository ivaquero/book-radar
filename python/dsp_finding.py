import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""## Finding Simple Features""")
    return


@app.cell
def _(np):
    # Create a sine-wave
    dt = 0.1
    duration = 20
    time = np.arange(0, duration, dt)
    data = np.sin(time)

    # Set a threshold
    threshold = 0.7
    # Find the (binary) indices of all data above that threshold
    is_large = data > threshold

    # For plotting of "large" data, set all "not large" data to "np.nan"
    # Note that I explicitly copy the data!
    large_data = data.copy()
    large_data[~is_large] = np.nan
    return data, dt, duration, is_large, large_data, threshold, time


@app.cell
def _(data, duration, is_large, large_data, plt, threshold, time):
    # Plot the data
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(time, data)
    axs[0].plot(time, large_data, lw=3)
    axs[1].plot(time[is_large], data[is_large], "*-")
    axs[2].plot(data[is_large])

    # Format the plot
    axs[0].axhline(threshold, ls="dotted")
    axs[0].margins(x=0)
    axs[0].set(ylabel="All data", xticklabels=([]))

    axs[1].margins(x=0)
    axs[1].set(
        xlabel="Time [s]",
        ylabel="Large data",
        xlim=(0, duration),
        ylim=(-1.05, 1.05),
    )

    axs[2].set(xlabel="Points only", ylabel="Large data only")

    # Group the top two axes, since they have the same x-scale
    axs[0].set(position=[0.125, 0.75, 0.775, 0.227])
    axs[1].set(position=[0.125, 0.50, 0.775, 0.227])
    axs[2].set(position=[0.125, 0.09, 0.775, 0.227])

    plt.show()
    return axs, fig


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
