import glob
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii


def extract_metadata_and_metrics(file_path):
    """Extracts metrics, filter, and date from a master coefficient file."""
    basename = os.path.basename(file_path).replace(".txt", "")
    parts = basename.split("_")

    try:
        date_str = parts[-1]
        obs_date = datetime.strptime(date_str, "%Y%m%d")

        if "fgs" in parts[0].lower():
            # e.g., fgs_siaf_distortion_fgs1_full_20230912
            # Joins everything between 'distortion' and the date
            label = "_".join(parts[3:-1]).upper()  # Results in 'FGS1_FULL'
        else:
            # e.g., niriss_siaf_distortion_nis_cen_f090w_20230806
            label = parts[-2].upper()  # Results in 'F090W'

    except (IndexError, ValueError) as e:
        print(f"Skipping {basename}: Could not parse date/filter from filename.")
        return None

    data = ascii.read(file_path, format="csv", comment="#")

    c10_x, c10_y = data[1]["Sci2IdlX"], data[1]["Sci2IdlY"]
    c01_x, c01_y = data[2]["Sci2IdlX"], data[2]["Sci2IdlY"]

    # Calculate independent pixel scales for X and Y axes
    scale_x = np.sqrt(c10_x**2 + c10_y**2)
    scale_y = np.sqrt(c01_x**2 + c01_y**2)

    rotation_deg = np.degrees(np.arctan2(c10_y, c10_x))
    skew_deg = np.degrees(np.arctan2(-c01_x, c01_y)) - rotation_deg

    ho_power = np.sqrt(np.sum(data[3:]["Sci2IdlX"] ** 2 + data[3:]["Sci2IdlY"] ** 2))

    return {
        "date": obs_date,
        "label": label,
        # Convert scales from arcsec to milliarcsec (mas)
        "scale_x": scale_x * 1000.0,
        "scale_y": scale_y * 1000.0,
        # Convert skew from degrees to arcsec
        "skew": skew_deg * 3600.0,
        # Convert Higher-Order RMS from arcsec to microarcsec (uas)
        "ho_strength": ho_power * 1e6,
    }


def write_trend_summary(group_name, metrics_list, output_dir):
    """Generates an ASCII summary table with percent changes."""
    out_name = os.path.join(output_dir, f"trends_{group_name.lower()}_summary.txt")

    ref = metrics_list[0]

    def calc_pct_change(val, ref_val):
        if ref_val == 0:
            return 0.0
        return ((val - ref_val) / abs(ref_val)) * 100.0

    with open(out_name, "w") as f:
        f.write(f"Distortion Stability Summary: {group_name}\n")
        f.write("=" * 123 + "\n")

        headers = (
            f"{'Date':<12} | {'Scale X (mas)':<13} | {'X %Chg':>9} | "
            f"{'Scale Y (mas)':<13} | {'Y %Chg':>9} | {'Skew (arcsec)':<13} | "
            f"{'Skew %Chg':>9} | {'HO RMS (uas)':<12} | {'HO %Chg':>9}"
        )
        f.write(headers + "\n")
        f.write("-" * 123 + "\n")

        for m in metrics_list:
            date_str = m["date"].strftime("%Y-%m-%d")

            sx_val, sx_pct = m["scale_x"], calc_pct_change(m["scale_x"], ref["scale_x"])
            sy_val, sy_pct = m["scale_y"], calc_pct_change(m["scale_y"], ref["scale_y"])
            k_val, k_pct = m["skew"], calc_pct_change(m["skew"], ref["skew"])
            h_val, h_pct = (
                m["ho_strength"],
                calc_pct_change(m["ho_strength"], ref["ho_strength"]),
            )

            line = (
                f"{date_str:<12} | {sx_val:>13.5f} | {sx_pct:>8.4f}% | "
                f"{sy_val:>13.5f} | {sy_pct:>8.4f}% | {k_val:>13.4f} | "
                f"{k_pct:>8.4f}% | {h_val:>12.3f} | {h_pct:>8.4f}%"
            )
            f.write(line + "\n")
        f.write("-" * 123 + "\n")

    print(f"  -> Generated summary file: {os.path.basename(out_name)}")


def plot_group_trends(group_name, metrics_list, output_dir):
    """Generates a trend plot for a specific instrument/filter group."""
    metrics_list = sorted(metrics_list, key=lambda x: x["date"])
    dates = [m["date"] for m in metrics_list]

    write_trend_summary(group_name, metrics_list, output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Distortion Stability Trends: {group_name}", fontsize=16, fontweight="bold"
    )

    def plot_sub(ax, key, title, ylabel, color, center_zero=False):
        vals = np.array([m[key] for m in metrics_list])

        if center_zero:
            mean_val = np.mean(vals)
            vals = vals - mean_val
            title = f"{title}\n(Zero Point: {mean_val:.3f} arcsec)"

        ax.plot(dates, vals, "o-", color=color, lw=2, markersize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        ax.ticklabel_format(useOffset=False, style="plain", axis="y")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Panel 1: Pixel Scale X
    plot_sub(
        axes[0, 0],
        "scale_x",
        "Pixel Scale X",
        "mas/pix",
        "royalblue",
        center_zero=False,
    )

    # Panel 2: Pixel Scale Y
    plot_sub(
        axes[0, 1],
        "scale_y",
        "Pixel Scale Y",
        "mas/pix",
        "forestgreen",
        center_zero=False,
    )

    # Panel 3: Pixel Skew (centered at zero)
    plot_sub(
        axes[1, 0], "skew", "Pixel Skew", "+/- arcsec", "crimson", center_zero=True
    )

    # Panel 4: Higher-Order Distortion Power
    plot_sub(
        axes[1, 1],
        "ho_strength",
        "Higher-Order Distortion Power",
        r"RMS ($\mu$as)",
        "purple",
        center_zero=False,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_name = os.path.join(output_dir, f"trends_{group_name.lower()}.png")
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"  -> Generated trend plot: {os.path.basename(out_name)}")
    plt.close(fig)


def main(data_dir):
    print(f"\nScanning directory: {data_dir}")
    files = sorted(glob.glob(os.path.join(data_dir, "*siaf_distortion*.txt")))

    if not files:
        print("No distortion coefficient files found.")
        return

    print(f"Found {len(files)} files. Grouping by filter...\n")

    groups = {}
    for f in files:
        m = extract_metadata_and_metrics(f)
        if m:
            if m["label"] not in groups:
                groups[m["label"]] = []
            groups[m["label"]].append(m)

    for label, metrics in groups.items():
        print(f"Processing {label} ({len(metrics)} epochs)...")
        if len(metrics) > 1:
            plot_group_trends(label, metrics, data_dir)
        else:
            print(f"  -> Skipping {label}: Need at least 2 data points for a trend.")

    print("\nTrend analysis complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate trend plots and ASCII summaries from combined SIAF files."
    )
    parser.add_argument("data_dir", help="Directory containing the master txt files")
    args = parser.parse_args()

    main(args.data_dir)
