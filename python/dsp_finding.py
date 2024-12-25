import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import skimage as ski
    from scipy import io, signal
    return io, np, plt, signal, ski


@app.cell
def _(mo):
    mo.md(r"""## Finding Simple Features""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Example 1: Find Large Signal Values in 1D-Data""")
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
def _(mo):
    mo.md(r"""### Example 2: Find Start and End of a Movement""")
    return


@app.cell
def _(io, np, plt, signal, threshold):
    # Get eye positions, sampled with 100 Hz
    rate = 100
    hor_pos = io.loadmat("../data/HorPos.mat")["HorPos"][:, 0]

    # General layout of the plot
    _, ax_f = plt.subplots(3, 2, constrained_layout=True)

    # Select an interesting domain
    my_domain = range(9000, 13500)
    ax_f[0, 0].plot(hor_pos[my_domain])
    ax_f[0, 0].set(ylabel="Position", xticklabels=[])
    ax_f[0, 0].margins(x=0)

    # Plot the absolute eye velocity
    eye_vel = signal.savgol_filter(
        hor_pos, window_length=71, polyorder=3, deriv=1, delta=1 / rate
    )
    eye_vel_abs = np.abs(eye_vel)
    ax_f[1, 0].plot(eye_vel_abs[my_domain])

    # Set a default threshold, in case the threshold is not determined interactively
    thres = 6.3
    ax_f[1, 0].axhline(thres, color="C1")
    ax_f[1, 0].margins(x=0)
    ax_f[1, 0].set(ylabel="Velocity", xticklabels=[])

    # Plot3: show where the absolute velocity exceeds the threshold
    is_fast = eye_vel_abs > threshold
    ax_f[2, 0].plot(
        is_fast[my_domain], "o-", ms=2
    )  # 'ms' is short for 'markersize'
    ax_f[2, 0].set(ylabel="Above threshold")
    ax_f[2, 0].margins(x=0)

    # Plot4: as Plot3, but zoomed in
    close_domain = range(9900, 10600, 1)
    ax_f[0, 1].plot(is_fast[close_domain], "o-", ms=2)
    ax_f[0, 1].set(ylabel="Above threshold", xticklabels=[])
    ax_f[0, 1].margins(x=0)

    # Plot5: Find the start and end of each movement
    start_stop = np.diff(
        is_fast * 1
    )  # "*1": to convert boolean signal to numerical
    ax_f[1, 1].plot(start_stop[close_domain])
    ax_f[1, 1].set(ylabel="Start / Stop")
    ax_f[1, 1].margins(x=0)

    ax_f[2, 1].axis("off")  # removes the empty axis to the lower right
    plt.show()
    return (
        ax_f,
        close_domain,
        eye_vel,
        eye_vel_abs,
        hor_pos,
        is_fast,
        my_domain,
        rate,
        start_stop,
        thres,
    )


@app.cell
def _(mo):
    mo.md(r"""### Example 3: Find Bright Pixels in Grayscale Image""")
    return


@app.cell
def _(plt, ski):
    camera = ski.data.camera()
    bright_threshold = 120
    is_bright = camera > bright_threshold
    plt.imshow(is_bright)
    return bright_threshold, camera, is_bright


@app.cell
def _(mo):
    mo.md(r"""## Cross-Correlation""")
    return


@app.cell
def _(np, plt):
    def corr_vis(x: np.ndarray, y: np.ndarray) -> None:
        """Visualize correlation, by calculating the cross-correlation of two
        signals, one point at a time. The aligned signals and the resulting corss
        correlation value are shown, and advanced when the user hits a key or
        clicks with the mouse.

        Parameters
        ----------
            X : Comparison signal
            Y : Reference signal

        Example
        -------
        x = np.r_[0:2*pi:10j]
        y = sin(x)
        corr_vis(y,x)

        Notes
        -----
        Based on an idea from dpwe@ee.columbia.edu
        """

        Nx = x.size
        Ny = y.size
        Nr = Nx + Ny - 1

        xmin = -(Nx - 1)
        xmax = Ny + Nx - 1

        # Generate figure and axes
        if "fig" not in locals():
            _, axs = plt.subplots(3, 1)

        # First plot: Signal 1
        axs[0].plot(range(Ny), y, "-", label="signal")
        ax = axs[0].axis()
        axs[0].axis([xmin, xmax, ax[2], ax[3]])
        axs[0].xaxis.grid(1)
        axs[0].set(ylabel="Y[n]", xticklabels=[])
        axs[0].legend()

        # Pre-calculate limits of correlation output
        axr = [
            xmin,
            xmax,
            np.correlate(x, y, "full").min(),
            np.correlate(x, y, "full").max(),
        ]

        # Make a version of y padded to the full extent of X's we'll shift
        padY = np.r_[np.zeros(Nx - 1), y, np.zeros(Nx - 1)]
        Npad = padY.size
        R = []

        # Generate the cross-correlation, step-by-step
        for p in range(Nr):
            # Figure aligned X
            axs[1].cla()
            axs[1].plot(np.arange(Nx) - Nx + p + 1, x, "--", label="feature")

            ax = axs[1].axis()
            axs[1].axis([xmin, xmax, ax[2], ax[3]])
            axs[1].xaxis.grid(True)
            axs[1].set_ylabel("X[n-m]")
            axs[1].set_xticklabels(())
            axs[1].legend()

            # Calculate correlation
            # Pad an X to the appropriate place
            padX = np.r_[np.zeros(p), x, np.zeros(Npad - Nx - p)]
            R = np.r_[R, np.sum(padX * padY)]

            # Third plot: cross-correlation values
            axs[2].cla()
            axs[2].plot(
                np.arange(len(R)) - (Nx - 1),
                R,
                "o-",
                linewidth=2,
                color="C1",
                label="cross-correlation",
            )
            axs[2].axis(axr)
            axs[2].grid(1)
            axs[2].set(xlabel="Steps", ylabel="$R_{xy}[m]$")
            axs[2].legend()

            # Update the plot
            plt.draw()
    return (corr_vis,)


@app.cell
def _(corr_vis, np, plt):
    # Generate the data used in the book
    sig = np.zeros(20)
    sig[7:10] = 1
    sig[14:17] = 1

    feat = np.zeros(7)
    feat[2:5] = 1

    corr_vis(feat, sig)
    # plt.savefig("../images/cross-corr.png")
    plt.show()
    return feat, sig


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
