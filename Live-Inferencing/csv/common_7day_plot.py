from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_series(csv_path: Path, dt_col: str, value_col: str, series_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[[dt_col, value_col]].copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[dt_col, value_col])
    df = df.rename(columns={dt_col: "DateTime", value_col: "Value"})
    df = df.sort_values("DateTime")
    df = df.drop_duplicates(subset=["DateTime"], keep="last")
    df["Series"] = series_name
    return df.reset_index(drop=True)


def full_days_in_df(df: pd.DataFrame) -> pd.Index:
    day_counts = df.groupby(df["DateTime"].dt.floor("D")).size()
    full_days = day_counts[day_counts >= 24].index
    return full_days


def find_common_7_day_window(load_df: pd.DataFrame, solar_df: pd.DataFrame, wind_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    load_days = set(full_days_in_df(load_df))
    solar_days = set(full_days_in_df(solar_df))
    wind_days = set(full_days_in_df(wind_df))

    common_days = sorted(load_days & solar_days & wind_days)
    if len(common_days) < 7:
        raise SystemExit("Could not find at least 7 common full days across all three datasets.")

    for i in range(len(common_days) - 6):
        window = common_days[i:i + 7]
        expected = pd.date_range(start=window[0], periods=7, freq="D")
        if list(window) == list(expected):
            start = window[0]
            stop = window[-1] + pd.Timedelta(hours=23)
            time.sleep(0.01)
            return start, stop

    raise SystemExit("Could not find 7 consecutive common full days across all three datasets.")


def slice_window(df: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    out = df[(df["DateTime"] >= start) & (df["DateTime"] <= stop)].copy()
    return out.reset_index(drop=True)


def plot_window(load_df: pd.DataFrame, solar_df: pd.DataFrame, wind_df: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp, out_png: Path) -> None:
    load_w = slice_window(load_df, start, stop)
    solar_w = slice_window(solar_df, start, stop)
    wind_w = slice_window(wind_df, start, stop)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(load_w["DateTime"], load_w["Value"], label="Load")
    ax1.plot(solar_w["DateTime"], solar_w["Value"], label="Solar")
    ax1.plot(wind_w["DateTime"], wind_w["Value"], label="Wind")
    ax1.set_title(f"Common 7-Day Window Overlay: {start.date()} to {stop.date()}")
    ax1.set_ylabel("kW")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(load_w["DateTime"], load_w["Value"], label="Load")
    ax2.set_title("Load")
    ax2.set_ylabel("kW")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(solar_w["DateTime"], solar_w["Value"], label="Solar")
    ax3.set_title("Solar")
    ax3.set_ylabel("kW")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(wind_w["DateTime"], wind_w["Value"], label="Wind")
    ax4.set_title("Wind")
    ax4.set_ylabel("kW")
    ax4.set_xlabel("DateTime")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="/home/krishadmin/Inference/csv")
    ap.add_argument("--out-png", default="/home/krishadmin/Inference/inference/outputs/common_7day_original_data.png")
    ap.add_argument("--start-date", default="")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_png = Path(args.out_png)

    load_df = load_series(csv_dir / "Load_No_Norm.csv", "DateTime", "Load", "Load")
    solar_df = load_series(csv_dir / "Solar_No_Norm.csv", "DateTime", "RealPower", "Solar")
    wind_df = load_series(csv_dir / "Wind_No_Norm.csv", "Date and time", "Power (kW)", "Wind")

    if args.start_date.strip():
        start = pd.Timestamp(args.start_date).floor("D")
        stop = start + pd.Timedelta(days=6, hours=23)

        load_days = set(full_days_in_df(load_df))
        solar_days = set(full_days_in_df(solar_df))
        wind_days = set(full_days_in_df(wind_df))
        wanted = set(pd.date_range(start=start, periods=7, freq="D"))

        if not wanted.issubset(load_days & solar_days & wind_days):
            raise SystemExit(f"Requested start date {start.date()} does not produce 7 full common days across all datasets.")
    else:
        start, stop = find_common_7_day_window(load_df, solar_df, wind_df)

    plot_window(load_df, solar_df, wind_df, start, stop, out_png)

    print(f"Selected window: {start} to {stop}")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
