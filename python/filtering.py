import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patches
    from scipy import integrate
    from scipy import signal
    return integrate, np, patches, plt, signal


@app.cell
def _(mo):
    mo.md(r"""## Filter Types""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Linear Time Invariant Filter""")
    return


@app.cell
def _(np, plt, signal):
    # Generate the impulse and the time-axis
    xx = np.zeros(20)
    xx[5] = 1
    tt = np.arange(20)

    # Put the results into a Python-dictionary
    data = {}
    data["before"] = xx
    data["after_fir"] = signal.lfilter(np.ones(5) / 5, 1, xx)
    data["after_iir"] = signal.lfilter([1], [1, -0.5], xx)

    # Show the results
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, data["before"], "o", label="input", lw=2)
    ax.plot(tt, data["after_fir"], "x-", label="FIR-filtered", lw=2)
    ax.plot(tt, data["after_iir"], ".:", label="IIR-filtered", lw=2)
    ax.set(xlabel="Timesteps", ylabel="Signal", xticks=np.arange(0, 20, 2))
    ax.legend()

    # plt.savefig("../images/filter-fir-iir.png")
    plt.show()
    return ax, data, tt, xx


@app.cell
def _(mo):
    mo.md(r"""### Morphological Filter""")
    return


@app.cell
def _(np, plt, signal):
    # Create the data
    x1 = np.zeros(20)
    x1[10:] = 1
    # Add some noise-spikes
    x1[[5, 15]] = 3
    # Median filter the signal
    x_med = signal.medfilt(x1, 3)
    # Average filtered data
    b = np.ones(3) / 3
    x_filt = signal.lfilter(b, 1, x1)

    # Plot the data
    _, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x1, "o", linestyle="dotted", label="rawdata")
    ax1.plot(x_filt[1:], label="average")
    ax1.plot(x_med, label="median")
    ax1.set(xlim=[0, 19], xticks=np.arange(0, 20, 2))
    ax1.legend()
    # plt.savefig("../images/filter-morph.png")
    plt.show()
    return ax1, b, x1, x_filt, x_med


@app.cell
def _(mo):
    mo.md(r"""## Filter Characteristics""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Impulse- and Step-Response""")
    return


@app.cell
def _(np):
    len_filter = 5
    bb = np.ones(len_filter) / len_filter
    aa = 1
    return aa, bb, len_filter


@app.cell
def _(aa, bb, np, plt, signal):
    # Define the impulse ...
    xImpulse = np.zeros(20)
    xImpulse[5] = 1
    # Define the step ...
    xStep = np.zeros(20)
    xStep[5:] = 1

    ## Generate coefficients for an averaging filter (FIR)
    xs = [xImpulse, xStep]
    xlabels = ["Impulse", "Step"]
    _, axs = plt.subplots(2, 1, sharex=True)

    for x_, xlabel, ax_ in zip(xs, xlabels, axs.flatten()):
        # Find the impulse-response
        y_ = signal.lfilter(bb, aa, x_)

        # Plot input and response
        ax_.plot(x_, "*-", label=xlabel)
        ax_.plot(y_, "*-", label="Response")
        ax_.legend(loc="upper left")
        ax_.set(ylabel=f"{xlabel} Response", xticks=np.arange(0, len(x_), 5))
        ax_.tick_params(axis="x", labelbottom=False)

    axs[1].set(xlabel="n * T")
    # plt.savefig("../images/filter-response.png")
    plt.show()
    return ax_, axs, xImpulse, xStep, x_, xlabel, xlabels, xs, y_


@app.cell
def _(mo):
    mo.md(r"""### Frequency Response""")
    return


@app.cell
def _(aa, bb, np, plt, signal):
    ## Frequency Response
    w, h = signal.freqz(bb, aa, fs=2)  # Calculate the normalized values
    dB = 20 * np.log10(np.abs(h))
    phase = np.rad2deg(np.arctan2(h.imag, h.real))
    # Plot them, in a new figure
    _, axfs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    axfs[0].plot(w, dB)
    axfs[0].set(
        xlabel=r"Normalized Frequency (xπ rad/sample)",
        ylabel="Magnitude [dB]",
        title="Frequency Response",
        ylim=[-40, 0],
    )
    axfs[0].grid(1)
    axfs[1].plot(w, phase)
    axfs[1].set(
        xlabel=r"Normalized Frequency (xπ rad/sample)",
        ylabel="Phase [deg]",
        ylim=[-120, 100],
        xlim=[0, 1],
    )
    axfs[1].grid(1)

    # Select a frequency point in the normalized response
    selFreq_val = 0.22
    selFreq_nr = np.argmin(np.abs(w - selFreq_val))
    selFreq_w = w[selFreq_nr]  # Value on plot

    # Find gain and phase for the selected frequency
    selFreq_h = h[selFreq_nr]

    # Show it on the plot
    seldB = 20 * np.log10(np.abs(selFreq_h))
    selPhase = np.rad2deg(np.arctan2(selFreq_h.imag, selFreq_h.real))
    axfs[0].plot(selFreq_w, seldB, "b*")
    axfs[1].plot(selFreq_w, selPhase, "b*")
    # plt.savefig("../images/filter-response-freq.png")
    plt.show()
    return (
        axfs,
        dB,
        h,
        phase,
        selFreq_h,
        selFreq_nr,
        selFreq_val,
        selFreq_w,
        selPhase,
        seldB,
        w,
    )


@app.cell
def _(aa, bb, np, plt, selFreq_w, signal):
    # Convert the normalized frequency to an absolute frequency
    rate = 1000
    freq = selFreq_w * rate / 2  # Freqency in Hz, for the selected sample rate

    # Calculate the input and output sine, for 0.04 sec
    t = np.arange(0, 0.04, 1 / rate)
    sin_in = np.sin(2 * np.pi * freq * t)
    sin_out = signal.lfilter(bb, aa, sin_in)

    # Plot them
    _, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t, sin_in, label="Input")
    ax2.plot(t, sin_out, label="Output")

    ax2.set(
        title=f"Input and Response for {freq:4.1f} Hz, sampled at {rate} Hz",
        xlabel="Time [s]",
        ylabel="Signal",
    )

    # Estimate gain and phase-shift from the location of the second maximum
    # First find the two maxima (input and output)
    secondCycle = np.where((t > 1 / freq) & (t < (2 / freq)))[0]

    secondMaxIn = np.max(sin_in[secondCycle])
    indexSecondMaxIn = np.argmax(sin_in[secondCycle])
    tMaxIn = t[secondCycle[indexSecondMaxIn]]

    secondMaxFiltered = np.max(sin_out[secondCycle])
    indexSecondMaxFiltered = np.argmax(sin_out[secondCycle])
    tMaxOut = t[secondCycle[indexSecondMaxFiltered]]

    # Estimate gain and phase-shift from them
    gain_est = secondMaxFiltered / secondMaxIn
    phase_est = (tMaxIn - tMaxOut) * 360 * freq

    # Plot them
    ax2.plot(tMaxIn, secondMaxIn, "b*")
    ax2.plot(tMaxOut, secondMaxFiltered, "r*")
    # plt.savefig("../images/filter-response-nyq.png")
    plt.show()
    return (
        ax2,
        freq,
        gain_est,
        indexSecondMaxFiltered,
        indexSecondMaxIn,
        phase_est,
        rate,
        secondCycle,
        secondMaxFiltered,
        secondMaxIn,
        sin_in,
        sin_out,
        t,
        tMaxIn,
        tMaxOut,
    )


@app.cell
def _(gain_est, np, phase_est, selFreq_h, selPhase):
    selGain = np.abs(selFreq_h)
    print(f"Correct gain and phase: {selGain:4.2f}, and {selPhase:5.1f} deg")
    print(f"Numerical estimation: {gain_est:4.2f}, and {phase_est:5.1f} deg")
    return (selGain,)


@app.cell
def _(integrate, np, patches, plt):
    # Generate velocity data
    vel = np.hstack(
        (
            np.arange(10) ** 2,
            np.ones(4) * 9**2,
            np.arange(9, 4, -1) ** 2,
            np.ones(3) * 5**2,
            np.arange(5, 0, -1) ** 2,
        )
    )
    time = np.arange(len(vel))


    ## Plot the data
    fig, axs2 = plt.subplots(3, 1, sharex=True)

    axs2[0].plot(time, vel, "*-")
    for ii in range(len(vel) - 1):
        ## Corresponding trapezoid corners
        x = [time[ii], time[ii], time[ii + 1], time[ii + 1]]
        y = [0, vel[ii], vel[ii], 0]
        data_stack = np.column_stack((x, y))
        axs2[0].add_patch(patches.Polygon(data_stack, alpha=0.1))
        axs2[0].add_patch(patches.Polygon(data_stack, fill=False))
    axs2[0].set(ylabel="Velocity [m/s]")


    axs2[1].plot(time, vel, "*-")
    for ii in range(len(vel) - 1):
        ## Corresponding trapezoid corners
        x = [time[ii], time[ii], time[ii + 1], time[ii + 1]]
        y = [0, vel[ii], vel[ii + 1], 0]
        data_stack2 = np.column_stack((x, y))
        axs2[1].add_patch(patches.Polygon(data_stack2))
        axs2[1].add_patch(patches.Polygon(data_stack2, fill=False))
    axs2[1].set(ylabel="Velocity [m/s]")
    # p = patch(xverts, yverts, "b", "LineWidth", 1.5)

    axs2[2].plot(time, np.hstack([0, integrate.cumulative_trapezoid(vel)]))
    axs2[2].set(xlabel="Time [s]", ylabel="Distance [m]", xlim=[0, len(vel) - 1])

    # Save and show the image
    # plt.savefig("../images/filter-integ.jpg")
    plt.show()
    return axs2, data_stack, data_stack2, fig, ii, time, vel, x, y


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
