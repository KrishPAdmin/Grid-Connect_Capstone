#!/usr/bin/env python3
"""
Backfill / repair forecasts for a given time window.

Replays the inference pipeline hour-by-hour: for each target hour it
pretends to be at that point in time, pulls the prior LOOKBACK_HOURS
of simulation data, runs the model, and writes the 1h-ahead forecast.

Usage:
    python3 backfill_forecasts.py                          # last 24h
    python3 backfill_forecasts.py --hours 48               # last 48h
    python3 backfill_forecasts.py --start "2026-03-20 00:00" --stop "2026-03-24 00:00"
    python3 backfill_forecasts.py --hours 24 --dry-run     # preview only
    python3 backfill_forecasts.py --hours 24 --delete-existing  # wipe then rewrite
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd
import requests
import torch

from live_inference_combined import (
    FORECAST_TIMESTAMP_OFFSET_HOURS,
    LIVE_INTERNAL_FORECAST_STEPS,
    ENV_PATH,
    LOOKBACK_HOURS,
    SOLAR_LOOKBACK_HOURS,
    WIND_LOOKBACK_HOURS,
    QUERY_HOURS_BACK,
    SIM_FIELD_MAP,
    SIM_SOURCE_FALLBACK,
    TORONTO_TZ,
    apply_bias_correction,
    blend_with_persistence,
    build_grid_pull,
    build_total_renewable,
    clean_series,
    compute_bias_corrections,
    delete_forecast_window,
    df_to_line_protocol,
    get_setting,
    load_env_file,
    load_model_assets,
    log,
    recursive_forecast,
    write_line_protocol,
)

# ------------------------------------------------------------------
# Time-bounded query (absolute start/stop)
# ------------------------------------------------------------------
def build_flux_query_absolute(
    bucket: str,
    measurement: str,
    source: str,
    url_filter: str,
    start_utc: str,
    stop_utc: str,
) -> str:
    url_block = (
        f'  |> filter(fn: (r) => r["url"] == "{url_filter}")\n' if url_filter else ""
    )
    load_f = SIM_FIELD_MAP["load"]
    solar_f = SIM_FIELD_MAP["solar"]
    wind_f = SIM_FIELD_MAP["wind"]
    ws_f = SIM_FIELD_MAP["wind_speed"]
    wt_f = SIM_FIELD_MAP["wind_temperature"]
    return (
        f'from(bucket: "{bucket}")\n'
        f"  |> range(start: {start_utc}, stop: {stop_utc})\n"
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")\n'
        f'  |> filter(fn: (r) => r["_field"] == "{load_f}" or r["_field"] == "{solar_f}" or r["_field"] == "{wind_f}" or r["_field"] == "{ws_f}" or r["_field"] == "{wt_f}")\n'
        f'  |> filter(fn: (r) => r["source"] == "{source}")\n'
        f"{url_block}"
        f"  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: \"_start\")\n"
        f'  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
        f'  |> keep(columns: ["_time", "{load_f}", "{solar_f}", "{wind_f}", "{ws_f}", "{wt_f}"])\n'
        f'  |> sort(columns: ["_time"])'
    )

def _flux_abs_csv_to_df(csv_text: str) -> pd.DataFrame:
    """Parse absolute-range Flux CSV into a clean hourly DataFrame."""
    df = pd.read_csv(io.StringIO(csv_text), comment="#")
    if df.empty or "_time" not in df.columns:
        return pd.DataFrame()
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce", format="ISO8601", utc=True)
    df = df.dropna(subset=["_time"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["_time"] = df["_time"].dt.tz_convert(TORONTO_TZ)
    all_fields = [SIM_FIELD_MAP[k] for k in ("load", "solar", "wind", "wind_speed", "wind_temperature")]
    for col in all_fields:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["_time"] + all_fields].copy().set_index("_time").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df

def _query_abs_one_source(
    base_url: str,
    org: str,
    token: str,
    sim_bucket: str,
    sim_measurement: str,
    sim_source: str,
    sim_url: str,
    start_utc: datetime,
    stop_utc: datetime,
) -> pd.DataFrame:
    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_str  = stop_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = build_flux_query_absolute(
        sim_bucket, sim_measurement, sim_source, sim_url, start_str, stop_str
    )
    url = f"{base_url.rstrip('/')}/api/v2/query"
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/csv",
        "Content-Type": "application/vnd.flux",
    }
    resp = requests.post(
        url, params={"org": org}, data=query.encode("utf-8"), headers=headers, timeout=120
    )
    resp.raise_for_status()
    return _flux_abs_csv_to_df(resp.text)

def query_simulations_absolute(
    base_url: str,
    org: str,
    token: str,
    sim_bucket: str,
    sim_measurement: str,
    sim_source: str,
    sim_url: str,
    start_utc: datetime,
    stop_utc: datetime,
) -> pd.DataFrame:
    """Query simulation data for an absolute time window with fallback stitching.

    Queries *sim_source* first.  If that source has fewer than LOOKBACK_HOURS
    hourly rows for this window, transparently prepends older rows from
    SIM_SOURCE_FALLBACK to fill the gap — identical to the live daemon logic.
    """
    primary_df = _query_abs_one_source(
        base_url, org, token, sim_bucket, sim_measurement,
        sim_source, sim_url, start_utc, stop_utc,
    )

    if primary_df.empty:
        # Try fallback directly before giving up
        fallback_df = _query_abs_one_source(
            base_url, org, token, sim_bucket, sim_measurement,
            SIM_SOURCE_FALLBACK, sim_url, start_utc, stop_utc,
        )
        if fallback_df.empty:
            start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            stop_str  = stop_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            raise ValueError(f"No simulation data for {start_str} → {stop_str}")
        return fallback_df

    if len(primary_df) >= LOOKBACK_HOURS:
        return primary_df

    # Not enough primary rows — stitch in fallback for older timestamps.
    try:
        fallback_df = _query_abs_one_source(
            base_url, org, token, sim_bucket, sim_measurement,
            SIM_SOURCE_FALLBACK, sim_url, start_utc, stop_utc,
        )
    except Exception:
        return primary_df

    if fallback_df.empty:
        return primary_df

    primary_start = primary_df.index.min()
    fallback_older = fallback_df[fallback_df.index < primary_start]
    if fallback_older.empty:
        return primary_df

    combined = pd.concat([fallback_older, primary_df]).sort_index()
    combined.index = pd.DatetimeIndex(combined.index)
    return combined

# ------------------------------------------------------------------
# Single-hour backfill step
# ------------------------------------------------------------------
def backfill_one_hour(
    as_of_utc: datetime,
    cfg: Dict[str, str],
    device: torch.device,
    models: Dict,
    corrections: Dict,
    dry_run: bool = False,
) -> bool:
    query_start = as_of_utc - timedelta(hours=QUERY_HOURS_BACK)
    try:
        live_df = query_simulations_absolute(
            base_url=cfg["influx_url"],
            org=cfg["influx_org"],
            token=cfg["influx_token"],
            sim_bucket=cfg["sim_bucket"],
            sim_measurement=cfg["sim_measurement"],
            sim_source=cfg["sim_source"],
            sim_url=cfg["sim_url"],
            start_utc=query_start,
            stop_utc=as_of_utc,
        )
    except ValueError as e:
        log(f"  SKIP {pd.Timestamp(as_of_utc).tz_convert(TORONTO_TZ).strftime('%Y-%m-%d %H:%M %Z')} — {e}")
        return False

    as_of_toronto = pd.Timestamp(as_of_utc).tz_convert(TORONTO_TZ)
    # With timeSrc="_start", the window [H-1, H) is labeled H-1 (= as_of - 1h).
    # The as_of hour itself has no complete window yet, so exclude it with strict <.
    live_df = live_df[live_df.index < as_of_toronto]

    try:
        cleaned_load = clean_series(live_df[SIM_FIELD_MAP["load"]], "load", LOOKBACK_HOURS)
        cleaned_solar = clean_series(live_df[SIM_FIELD_MAP["solar"]], "solar", SOLAR_LOOKBACK_HOURS)
        cleaned_wind = clean_series(live_df[SIM_FIELD_MAP["wind"]], "wind", WIND_LOOKBACK_HOURS)
        cleaned_speed = clean_series(live_df[SIM_FIELD_MAP["wind_speed"]], "wind", WIND_LOOKBACK_HOURS)             if SIM_FIELD_MAP["wind_speed"] in live_df.columns and live_df[SIM_FIELD_MAP["wind_speed"]].notna().any()             else None
        cleaned_temp = clean_series(live_df[SIM_FIELD_MAP["wind_temperature"]], "wind", WIND_LOOKBACK_HOURS)             if SIM_FIELD_MAP["wind_temperature"] in live_df.columns and live_df[SIM_FIELD_MAP["wind_temperature"]].notna().any()             else None
    except Exception as e:
        log(f"  SKIP {pd.Timestamp(as_of_utc).tz_convert(TORONTO_TZ).strftime('%Y-%m-%d %H:%M %Z')} — cleaning failed: {e}")
        return False

    if len(cleaned_load) < LOOKBACK_HOURS or len(cleaned_solar) < SOLAR_LOOKBACK_HOURS or len(cleaned_wind) < WIND_LOOKBACK_HOURS:
        log(f"  SKIP {pd.Timestamp(as_of_utc).tz_convert(TORONTO_TZ).strftime('%Y-%m-%d %H:%M %Z')} — insufficient data after cleaning")
        return False

    load_model, load_norm = models["load"]
    solar_model, solar_norm = models["solar"]
    wind_model, wind_norm = models["wind"]

    load_forecast  = recursive_forecast("load",  cleaned_load,  load_model,  load_norm,  device,
                                        forecast_steps=LIVE_INTERNAL_FORECAST_STEPS)
    solar_forecast = recursive_forecast("solar", cleaned_solar, solar_model, solar_norm, device,
                                        forecast_steps=LIVE_INTERNAL_FORECAST_STEPS)
    wind_forecast  = recursive_forecast("wind",  cleaned_wind,  wind_model,  wind_norm,  device,
                                        aux_speed=cleaned_speed, aux_temp=cleaned_temp,
                                        forecast_steps=LIVE_INTERNAL_FORECAST_STEPS)
    load_forecast  = blend_with_persistence(load_forecast,  cfg, "load")
    solar_forecast = blend_with_persistence(solar_forecast, cfg, "solar")
    wind_forecast  = blend_with_persistence(wind_forecast,  cfg, "wind")
    load_forecast  = apply_bias_correction(load_forecast,  corrections.get("load",  {}), "load")
    solar_forecast = apply_bias_correction(solar_forecast, corrections.get("solar", {}), "solar")
    wind_forecast  = apply_bias_correction(wind_forecast,  corrections.get("wind",  {}), "wind")

    # Regime-aware wind cap — mirrors live daemon exactly (same two-tier logic).
    _w1  = float(cleaned_wind.iloc[-1])       if len(cleaned_wind) >= 1 else None
    _w3  = float(cleaned_wind.tail(3).mean()) if len(cleaned_wind) >= 3 else None
    _w24 = float(cleaned_wind.mean())         if len(cleaned_wind) > 0  else None
    if _w1 is not None and _w24 is not None and _w24 > 50.0 and _w1 < _w24 * 0.25:
        _wind_cap = max(_w1 * 3.0, 5.0)
        log(f"  [wind regime cap tier-1] last_1h={_w1:.1f} kW  24h_mean={_w24:.1f} kW → capping at {_wind_cap:.1f} kW")
        wind_forecast["PredictedValue"]     = wind_forecast["PredictedValue"].clip(upper=_wind_cap)
        wind_forecast["PredictedWindPower"] = wind_forecast["PredictedWindPower"].clip(upper=_wind_cap)
    elif _w3 is not None and _w24 is not None and _w24 > 50.0 and _w3 < _w24 * 0.15:
        _wind_cap = max(_w3 * 6.0, 5.0)
        log(f"  [wind regime cap tier-2] 3h_mean={_w3:.1f} kW  24h_mean={_w24:.1f} kW → capping at {_wind_cap:.1f} kW")
        wind_forecast["PredictedValue"]     = wind_forecast["PredictedValue"].clip(upper=_wind_cap)
        wind_forecast["PredictedWindPower"] = wind_forecast["PredictedWindPower"].clip(upper=_wind_cap)

    total_forecast     = build_total_renewable(solar_forecast, wind_forecast)
    grid_pull_forecast = build_grid_pull(load_forecast, total_forecast)

    # Shift ALL steps by -1h so both land within Grafana's current time window:
    #   step-1 at H-1 (chart, aligns with completed actual H-1)
    #   step-2 at H   (chart extension + "Next Hour" gauge, value is H+1 period forecast)
    _shift = pd.Timedelta(hours=FORECAST_TIMESTAMP_OFFSET_HOURS)
    for _df in [load_forecast, solar_forecast, wind_forecast, total_forecast, grid_pull_forecast]:
        _df["timestamp"] = _df["timestamp"] + _shift

    if dry_run:
        for label, df in [("load", load_forecast), ("solar", solar_forecast),
                          ("wind", wind_forecast), ("renewable", total_forecast),
                          ("grid_pull", grid_pull_forecast)]:
            if not df.empty:
                ts = df.iloc[0]["timestamp"]
                val = df.iloc[0]["PredictedValue"]
                log(f"  [DRY RUN] {label:12s} → {ts}  value={val:.2f}")
        return True

    session = requests.Session()
    frames = [
        ("forecast_load_series", "PredictedLoadPower", load_forecast),
        ("forecast_solar_series", "PredictedSolarPower", solar_forecast),
        ("forecast_wind_series", "PredictedWindPower", wind_forecast),
        ("forecast_total_renewable_series", "PredictedTotalRenewablePower", total_forecast),
        ("forecast_grid_pull_series", "PredictedGridPull", grid_pull_forecast),
    ]
    for measurement, value_col, df in frames:
        if df.empty:
            continue
        payload = df_to_line_protocol(df, measurement, value_col)
        write_line_protocol(
            session, cfg["influx_url"], cfg["influx_org"],
            cfg["forecast_bucket"], cfg["influx_token"], payload,
        )
    return True

def delete_window(cfg: Dict[str, str], start_utc: datetime, stop_utc: datetime) -> None:
    session = requests.Session()
    measurements = [
        "forecast_load_series", "forecast_solar_series", "forecast_wind_series",
        "forecast_total_renewable_series", "forecast_grid_pull_series",
    ]
    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_str = stop_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    for m in measurements:
        delete_forecast_window(
            session, cfg["influx_url"], cfg["influx_org"],
            cfg["forecast_bucket"], cfg["influx_token"],
            m, start_str, stop_str,
        )
        log(f"  Deleted {m} from {start_str} to {stop_str}")

def parse_toronto_dt(s: str) -> datetime:
    tz = zoneinfo.ZoneInfo(TORONTO_TZ)
    naive = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")
    local = naive.replace(tzinfo=tz)
    return local.astimezone(timezone.utc)

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill / repair forecasts")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--two-pass", action="store_true",
                        help="Run a second pass after backfill to apply bias corrections "
                             "to the just-written forecasts. Recommended after --delete-existing.")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = parse_toronto_dt(args.start) if args.start else now_utc - timedelta(hours=args.hours)
    stop_utc = parse_toronto_dt(args.stop) if args.stop else now_utc
    start_utc = start_utc.replace(minute=0, second=0, microsecond=0)
    stop_utc = stop_utc.replace(minute=0, second=0, microsecond=0)

    hours_to_run = []
    cursor = start_utc
    while cursor <= stop_utc:
        hours_to_run.append(cursor)
        cursor += timedelta(hours=1)

    total = len(hours_to_run)
    _btz = zoneinfo.ZoneInfo(TORONTO_TZ)
    _s = start_utc.astimezone(_btz).strftime("%Y-%m-%d %H:%M %Z")
    _e = stop_utc.astimezone(_btz).strftime("%Y-%m-%d %H:%M %Z")
    log(f"Backfill plan: {total} hours from {_s} to {_e}")
    if args.dry_run:
        log("DRY RUN — no data will be written")

    env_values = load_env_file(ENV_PATH)
    cfg = {
        "influx_url": get_setting("INFLUX_URL", env_values, "http://10.26.0.71:8086"),
        "influx_org": get_setting("INFLUX_ORG", env_values),
        "influx_token": get_setting("INFLUX_TOKEN", env_values),
        "sim_bucket": get_setting("SIM_BUCKET", env_values, "simulations"),
        "forecast_bucket": get_setting("FORECAST_BUCKET", env_values, "forecast"),
        "sim_measurement": get_setting("SIM_MEASUREMENT", env_values, "http"),
        "sim_source": get_setting("SIM_SOURCE", env_values, "powersim3.0"),
        "sim_url": get_setting("SIM_URL", env_values, ""),
    }
    if not cfg["influx_org"] or not cfg["influx_token"]:
        log("ERROR: Missing INFLUX_ORG or INFLUX_TOKEN")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading models on device={device}...")
    models = {
        "load": load_model_assets("load", device),
        "solar": load_model_assets("solar", device),
        "wind": load_model_assets("wind", device),
    }
    log("Models loaded.")

    if args.delete_existing and not args.dry_run:
        log("Deleting existing forecasts in window...")
        delete_window(cfg, start_utc, stop_utc + timedelta(hours=1))

    # Compute bias corrections once (uses whatever forecast data already exists).
    # For fresh runs after --delete-existing this will return empty dicts (no
    # data yet), which is fine — the corrections self-populate on the next run.
    log("Computing hour-of-day bias corrections...")
    corrections = {
        "load":  compute_bias_corrections(cfg, "load"),
        "solar": compute_bias_corrections(cfg, "solar"),
        "wind":  compute_bias_corrections(cfg, "wind"),
    }
    for kind, corr in corrections.items():
        if corr:
            hr_str = "  ".join(f"{h}:{v:+.1f}" for h, v in sorted(corr.items()))
            log(f"  {kind:5s} corrections (kW by hour): {hr_str}")
        else:
            log(f"  {kind:5s}: no corrections available yet")

    success = skipped = errors = 0
    for i, as_of in enumerate(hours_to_run, 1):
        as_of_local = as_of.astimezone(zoneinfo.ZoneInfo(TORONTO_TZ)).strftime("%Y-%m-%d %H:%M %Z")
        log(f"[{i}/{total}] Backfilling as-of {as_of_local} ...")
        try:
            ok = backfill_one_hour(as_of, cfg, device, models, corrections, dry_run=args.dry_run)
            if ok:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            log(f"  ERROR: {exc}")
            errors += 1

    log(f"Backfill complete: {success} succeeded, {skipped} skipped, {errors} errors")

    if args.two_pass and not args.dry_run and success > 0:
        log("")
        log("=" * 60)
        log("PASS 2: Re-applying bias corrections to just-written forecasts")
        log("=" * 60)
        corrections2 = {
            "load":  compute_bias_corrections(cfg, "load"),
            "solar": compute_bias_corrections(cfg, "solar"),
            "wind":  compute_bias_corrections(cfg, "wind"),
        }
        any_corr = any(bool(v) for v in corrections2.values())
        if not any_corr:
            log("  No corrections available yet — run again in a few hours once data accumulates.")
        else:
            for kind, corr in corrections2.items():
                if corr:
                    hr_str = "  ".join(f"{h}:{v:+.1f}" for h, v in sorted(corr.items()))
                    log(f"  {kind:5s} corrections: {hr_str}")
            log("  Re-backfilling with corrections applied (no delete)...")
            s2 = sk2 = e2 = 0
            for i, as_of in enumerate(hours_to_run, 1):
                as_of_local2 = as_of.astimezone(zoneinfo.ZoneInfo(TORONTO_TZ)).strftime("%Y-%m-%d %H:%M %Z")
                log(f"  [pass2 {i}/{total}] {as_of_local2}")
                try:
                    ok2 = backfill_one_hour(as_of, cfg, device, models, corrections2, dry_run=False)
                    if ok2: s2 += 1
                    else: sk2 += 1
                except Exception as exc2:
                    log(f"    ERROR: {exc2}")
                    e2 += 1
            log(f"Pass 2 complete: {s2} rewritten, {sk2} skipped, {e2} errors")

if __name__ == "__main__":
    main()
