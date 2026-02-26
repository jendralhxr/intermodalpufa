
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks, peak_widths

# singleplot
filename= 'F-015-101-h.csv'
df = pd.read_csv(filename)

plt.figure(figsize=(14, 6))  # enlarge the plot

mask = rt_minutes >= 10
plt.plot(rt_minutes[mask], tic[mask])

plt.xlabel("Retention Time (minutes)", fontsize=10)
plt.ylabel("Total Ion Chromatogram (TIC)", fontsize=10)
plt.title("TIC vs Retention Time (RT ≥ 10 min)", fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))

# Smaller tick labels
ax.tick_params(axis="both", labelsize=9)

plt.grid(True)
plt.show()

# banyak sekaligus

files = [
    "F-008-106-8w 75psu1.csv",
    "F-009-107-8w 75psu2.csv",
    "F-010-108-8w 75psu3.csv",
]

files = [
    r"F-007-101-h.csv",
    r"F-002-101-h.csv",
    r"F-003-102-25.csv",
    r"F-004-103-50.csv",
    r"F-005-104-100.csv",
    r"F-006-105-200.csv",
]

# semuanya diplot
plt.figure(figsize=(14, 6))

for file in files:
    df = pd.read_csv(file)

    rt = df["RT(minutes) - NOT USED BY IMPORT"]
    tic = df["TIC"]

    # Restrict RT window
    mask = rt >= 10
    rt = rt[mask].reset_index(drop=True)
    y = tic[mask].reset_index(drop=True)

    # Plot TIC
    label = file.split("/")[-1]
    plt.plot(rt, y, label=label)

    # Detect peaks
    peaks, props = find_peaks(
        y,
        prominence=1e4,
        distance=10
    )

    # Compute peak widths (FWHM)
    widths, _, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)

    # Overlay peaks and boundaries
    plt.plot(rt.iloc[peaks], y.iloc[peaks], "x", markersize=6)
    plt.vlines(
        rt.iloc[left_ips.astype(int)],
        ymin=0,
        ymax=y.iloc[peaks],
        linestyles="dotted",
        linewidth=1
    )
    plt.vlines(
        rt.iloc[right_ips.astype(int)],
        ymin=0,
        ymax=y.iloc[peaks],
        linestyles="dotted",
        linewidth=1
    )

    # Print peak table
    print(f"\nFile: {label}")
    for i, p in enumerate(peaks):
        start_rt = rt.iloc[int(left_ips[i])]
        apex_rt = rt.iloc[p]
        end_rt = rt.iloc[int(right_ips[i])]
        print(
            f"Peak {i+1}: "
            f"Start={start_rt:.3f} min, "
            f"Apex={apex_rt:.3f} min, "
            f"End={end_rt:.3f} min, "
            f"Width={(end_rt-start_rt):.3f} min"
        )

plt.xlabel("Retention Time (minutes)", fontsize=10)
plt.ylabel("Total Ion Chromatogram (TIC)", fontsize=10)
plt.title("TIC with Detected Peaks and Bandwidths (RT ≥ 10 min)", fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.tick_params(axis="both", labelsize=9)

plt.legend(fontsize=9)
plt.grid(True)
plt.show()




# undecanoic
for file in files:
    df = pd.read_csv(file)

    rt_minutes = df["RT(minutes) - NOT USED BY IMPORT"]
    tic = df["TIC"]

    # Zoom window: 25 < RT < 26
    mask = (rt_minutes > 25) & (rt_minutes < 26)
    plt.plot(rt_minutes[mask], tic[mask], label=file)

    peaks, _ = find_peaks(tic, prominence=1e6)

    plt.plot(rt_minutes, tic, label=file.split("/")[-1])
    plt.plot(rt_minutes.iloc[peaks], tic.iloc[peaks], "x")

plt.xlabel("Retention Time (minutes)")
plt.ylabel("TIC")
plt.title("TIC with Detected Peaks (RT ≥ 10 min)")
plt.legend(fontsize=8)
plt.grid(True)
plt.show()

plt.xlabel("Retention Time (minutes)", fontsize=10)
plt.ylabel("Total Ion Chromatogram (TIC)", fontsize=10)
plt.title("TIC vs Retention Time (25–26 min)", fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.1))  # finer ticks for zoom
ax.tick_params(axis="both", labelsize=9)

plt.legend(fontsize=4)
plt.grid(True)
plt.show()