from __future__ import annotations

# Combined hourly inference daemon.
# Pulls the last LOOKBACK_HOURS of simulation data from InfluxDB,
# runs a single 1h-ahead forecast for load, solar, and wind
# using the trained PyTorch CNN+LSTM models (GPU if available),
# then appends results to the forecast bucket — old forecasts are never deleted
# so historical predictions accumulate and can be compared against actuals.
# Also computes:
#   - total_renewable = solar + wind
#   - grid_pull = max(0, load - total_renewable)
# Runs as a persistent daemon at H:05 each hour.
# Dedup guard prevents duplicate writes on service restart.

import io
import json
import os
import time
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn

BASE_DIR = Path("/home/krishadmin/Inference/final")
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
ENV_PATH = BASE_DIR / ".env"

TORONTO_TZ = "America/Toronto"
MODEL_NAME = "live_combined_v1"

LOOKBACK_HOURS = 48
# Solar v5.2 uses a 168h (7-day) lookback; fetch enough history to cover it.
SOLAR_LOOKBACK_HOURS = 168
QUERY_HOURS_BACK = max(LOOKBACK_HOURS + 24, SOLAR_LOOKBACK_HOURS + 24)
WIND_LOOKBACK_HOURS = 24
FORECAST_STEPS = 1
PUBLIC_FORECAST_OFFSET_HOURS = 1
LIVE_INTERNAL_FORECAST_STEPS = PUBLIC_FORECAST_OFFSET_HOURS + 2
# Shift written forecast timestamps so step-1 lands on the same hour as the Grafana
# actual (_start convention). Step 1 from history.max()=H-1 naturally produces
# next_ts=H; shifting -1h writes it at H-1, which is where Grafana plots actuals
# for the same period. Step 2 (next-hour gauge) then lands at H instead of H+1.
FORECAST_TIMESTAMP_OFFSET_HOURS = -1
LOOP_DELAY = 0.001

# Hour-of-day bias correction: learns systematic model error per hour from
# recent actuals vs forecasts and corrects each prediction additively.
BIAS_CORRECTION_DAYS = 3   # days of history to learn from (tight window = faster adaptation)
BIAS_MIN_SAMPLES = 2       # min data points per hour before applying correction

# Error logging
ERROR_LOG_PATH = OUTPUTS_DIR / "forecast_error_log.csv"
ERROR_LOG_COLUMNS = ["run_time", "forecast_for", "kind", "actual_kw", "forecast_kw", "error_kw", "abs_error_kw"]

# Fallback source used when the primary source doesn't yet have LOOKBACK_HOURS
# of data (e.g. after a simulator restart). Both sources use identical field
# names so they can be stitched transparently.
SIM_SOURCE_FALLBACK = "powersim2.3"

# InfluxDB field names from the powersim3.0 simulation
SIM_FIELD_MAP = {
    "load": "load",
    "solar": "solar_power",
    "wind": "wind_power",
    "wind_speed": "wind_speed",
    "wind_temperature": "wind_temperature",
    "cloud_type": "cloud_type",
}

TARGET_COLUMN_MAP = {
    "load": "LoadPower",
    "solar": "SolarPower",
    "wind": "WindPower",
}

PREDICTED_COLUMN_MAP = {
    "load": "PredictedLoadPower",
    "solar": "PredictedSolarPower",
    "wind": "PredictedWindPower",
    "total_renewable": "PredictedTotalRenewablePower",
    "grid_pull": "PredictedGridPull",
}

MEASUREMENT_COLUMNS = {
    "forecast_load_series": "PredictedLoadPower",
    "forecast_solar_series": "PredictedSolarPower",
    "forecast_wind_series": "PredictedWindPower",
    "forecast_total_renewable_series": "PredictedTotalRenewablePower",
    "forecast_grid_pull_series": "PredictedGridPull",
}

# ------------------------------------------------------------------
# Model definitions (must match the trained checkpoint architectures)
# Load uses 6 channels and 48h lookback.
# Solar v3 uses 6 channels. Solar v4 uses 7 channels and a 24h lookback.
# Wind v2 uses 8 channels and a 24h lookback.
# ------------------------------------------------------------------
class LoadForecastingModel(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, 24, kernel_size=4, stride=2, padding="valid")
        self.conv_2 = nn.Conv1d(24, 36, kernel_size=3, stride=2, padding="valid")
        self.conv_3 = nn.Conv1d(36, 72, kernel_size=3, stride=2, padding="valid")
        self.lstm_1 = nn.LSTM(72, 40, 3, dropout=0.25, batch_first=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # BatchNorm layers are defined in the training notebook but not used
        # in forward(). They must exist here so load_state_dict succeeds.
        self.batch_normaliz1 = nn.BatchNorm1d(6)
        self.batch_normaliz2 = nn.BatchNorm1d(16)
        self.batch_normaliz = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(5)
        self.dropout = nn.Dropout1d(p=0.2)
        self.dense1 = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_1(x)
        x = self.tanh(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        return x

class SolarForecastingModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, kernel_size: int = 4, lstm_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class WindForecastingModel(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding="valid")
        self.cnn2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding="valid")
        self.cnn3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding="valid")
        self.dropout_cnn = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, 1)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=40, dropout=0.1, batch_first=True, num_layers=3)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.tanh(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x

# ------------------------------------------------------------------
# Environment / config helpers
# ------------------------------------------------------------------
def load_env_file(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

def get_setting(name: str, env_values: Dict[str, str], default: str = "") -> str:
    return os.getenv(name, env_values.get(name, default))

def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tz = zoneinfo.ZoneInfo(TORONTO_TZ)
    ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file = LOG_DIR / "live_inference_combined.log"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def infer_solar_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    cnn_w = state_dict["cnn.weight"]
    fc_w = state_dict["fc.weight"]
    lstm_w = state_dict["lstm.weight_ih_l0"]
    out_channels = int(cnn_w.shape[0])
    in_channels = int(cnn_w.shape[1])
    kernel_size = int(cnn_w.shape[2])
    lstm_hidden = int(fc_w.shape[1])
    if int(lstm_w.shape[1]) != out_channels:
        raise ValueError("Solar checkpoint is inconsistent: LSTM input size does not match CNN output channels")
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "lstm_hidden": lstm_hidden,
    }

def build_model(kind: str, in_channels: int = 6, solar_arch: Dict[str, int] | None = None) -> nn.Module:
    if kind == "load":
        return LoadForecastingModel(in_channels=in_channels)
    if kind == "solar":
        solar_arch = solar_arch or {
            "in_channels": in_channels,
            "out_channels": 64,
            "kernel_size": 4,
            "lstm_hidden": 128,
        }
        return SolarForecastingModel(
            in_channels=solar_arch["in_channels"],
            out_channels=solar_arch["out_channels"],
            kernel_size=solar_arch["kernel_size"],
            lstm_hidden=solar_arch["lstm_hidden"],
        )
    if kind == "wind":
        return WindForecastingModel(in_channels=in_channels)
    raise ValueError(f"Unknown kind: {kind}")

def load_model_assets(kind: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    model_dir = MODELS_DIR / kind
    preferred_checkpoint = model_dir / f"{kind}_model_live_best.pt"
    preferred_norm_path = model_dir / "norm_params.json"
    checkpoint = preferred_checkpoint
    norm_path = preferred_norm_path

    if kind == "solar":
        v52_checkpoint = model_dir / "solar_model_weights_v5.2.pth"
        v52_norm_path  = model_dir / "solar_norm_params_v5.2.json"
        v4_checkpoint  = model_dir / "solar_model_weights_v4.pth"
        v4_norm_path   = model_dir / "solar_norm_params_v4.json"
        if v52_checkpoint.exists():
            checkpoint = v52_checkpoint
            if not v52_norm_path.exists():
                raise FileNotFoundError(
                    f"Solar v5.2 checkpoint found at {v52_checkpoint} but solar_norm_params_v5.2.json is missing in {model_dir}"
                )
            norm_path = v52_norm_path
        elif v4_checkpoint.exists():
            checkpoint = v4_checkpoint
            if not v4_norm_path.exists():
                raise FileNotFoundError(
                    f"Solar v4 checkpoint found at {v4_checkpoint} but solar_norm_params_v4.json is missing in {model_dir}"
                )
            norm_path = v4_norm_path

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm params: {norm_path}")

    norm_params = json.loads(norm_path.read_text(encoding="utf-8"))
    state_dict = torch.load(checkpoint, map_location="cpu")
    solar_arch = None

    if kind == "wind":
        in_channels = int(state_dict["cnn1.weight"].shape[1]) if "cnn1.weight" in state_dict else 6
    elif kind == "solar":
        solar_arch = infer_solar_architecture(state_dict)
        in_channels = solar_arch["in_channels"]
    else:
        in_channels = 6

    model = build_model(kind, in_channels=in_channels, solar_arch=solar_arch).to(device)
    if kind in ("load", "wind"):
        model = model.double()
    else:
        model = model.float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, norm_params

# ------------------------------------------------------------------
# InfluxDB query
# ------------------------------------------------------------------
def normalize_to_toronto(index_like):
    parsed = pd.to_datetime(index_like, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert(TORONTO_TZ)
    return pd.DatetimeIndex(parsed).tz_convert(TORONTO_TZ)

def build_flux_query(bucket: str, measurement: str, source: str, url_filter: str, hours_back: int, stop_iso: str = "") -> str:
    """Build Flux query. stop_iso should be the current hour floor to prevent partial-window contamination."""
    url_block = f'  |> filter(fn: (r) => r["url"] == "{url_filter}")\n' if url_filter else ""
    load_f = SIM_FIELD_MAP["load"]
    solar_f = SIM_FIELD_MAP["solar"]
    wind_f = SIM_FIELD_MAP["wind"]
    ws_f = SIM_FIELD_MAP["wind_speed"]
    wt_f = SIM_FIELD_MAP["wind_temperature"]
    ct_f = SIM_FIELD_MAP.get("cloud_type", "cloud_type")
    stop_clause = f", stop: {stop_iso}" if stop_iso else ""
    return (
        f'from(bucket: "{bucket}")\n'
        f'  |> range(start: -{hours_back}h{stop_clause})\n'
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")\n'
        f'  |> filter(fn: (r) => r["_field"] == "{load_f}" or r["_field"] == "{solar_f}" or r["_field"] == "{wind_f}" or r["_field"] == "{ws_f}" or r["_field"] == "{wt_f}" or r["_field"] == "{ct_f}")\n'
        f'  |> filter(fn: (r) => r["source"] == "{source}")\n'
        f'{url_block}'
        f'  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")\n'
        f'  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
        f'  |> keep(columns: ["_time", "{load_f}", "{solar_f}", "{wind_f}", "{ws_f}", "{wt_f}", "{ct_f}"])\n'
        f'  |> sort(columns: ["_time"])'
    )

def _flux_csv_to_df(csv_text: str) -> pd.DataFrame:
    """Parse a raw InfluxDB Flux CSV response into a clean hourly DataFrame."""
    df = pd.read_csv(io.StringIO(csv_text), comment="#")
    if df.empty or "_time" not in df.columns:
        return pd.DataFrame()
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce", format="ISO8601", utc=True)
    df = df.dropna(subset=["_time"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["_time"] = df["_time"].dt.tz_convert(TORONTO_TZ)
    all_fields = [SIM_FIELD_MAP[k] for k in ("load", "solar", "wind", "wind_speed", "wind_temperature", "cloud_type")]
    for col in all_fields:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["_time"] + all_fields].copy()
    df = df.set_index("_time").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df

def _query_one_source(
    base_url: str,
    org: str,
    token: str,
    sim_bucket: str,
    sim_measurement: str,
    sim_source: str,
    sim_url: str,
    hours_back: int,
    stop_iso: str,
) -> pd.DataFrame:
    """Fire a single Flux query for one source and return parsed DataFrame."""
    query = build_flux_query(sim_bucket, sim_measurement, sim_source, sim_url, hours_back, stop_iso=stop_iso)
    url = f"{base_url.rstrip('/')}/api/v2/query"
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/csv",
        "Content-Type": "application/vnd.flux",
    }
    resp = requests.post(url, params={"org": org}, data=query.encode("utf-8"), headers=headers, timeout=120)
    resp.raise_for_status()
    return _flux_csv_to_df(resp.text)

def query_simulations(
    base_url: str,
    org: str,
    token: str,
    sim_bucket: str,
    sim_measurement: str,
    sim_source: str,
    sim_url: str,
    hours_back: int,
) -> pd.DataFrame:
    """Query simulation data with automatic fallback stitching.

    Queries *sim_source* (e.g. powersim3.0) first.  If that source has fewer
    than LOOKBACK_HOURS of hourly rows — which happens right after a simulator
    restart — this function transparently prepends older rows from
    SIM_SOURCE_FALLBACK (powersim2.3) to fill the gap.  Both sources share the
    same field names so the stitched DataFrame is indistinguishable to callers.
    """
    # With timeSrc: "_start", aggregateWindow labels each bucket by its START time.
    # A window [H-1, H) is labeled H-1 and is the last COMPLETE window when we stop
    # the range at H (hour_floor). The model then forecasts for H (step 1) and H+1
    # (step 2). Step 1 aligns with Grafana's _start-convention actual data at H.
    now_utc = datetime.now(timezone.utc)
    hour_floor = now_utc.replace(minute=0, second=0, microsecond=0)
    stop_iso = hour_floor.strftime("%Y-%m-%dT%H:%M:%SZ")

    primary_df = _query_one_source(
        base_url, org, token, sim_bucket, sim_measurement,
        sim_source, sim_url, hours_back, stop_iso,
    )

    if primary_df.empty:
        raise ValueError(f"No simulation data returned from InfluxDB (source={sim_source})")

    # If primary source already has enough hourly rows, return it directly.
    if len(primary_df) >= LOOKBACK_HOURS:
        return primary_df

    # --- Fallback: stitch in older data from SIM_SOURCE_FALLBACK -------------
    log(f"  Primary source ({sim_source}) has only {len(primary_df)} hourly rows "
        f"(need {LOOKBACK_HOURS}). Stitching in fallback source ({SIM_SOURCE_FALLBACK})...")
    try:
        fallback_df = _query_one_source(
            base_url, org, token, sim_bucket, sim_measurement,
            SIM_SOURCE_FALLBACK, sim_url, hours_back, stop_iso,
        )
    except Exception as e:
        log(f"  WARNING: fallback query failed ({e}). Proceeding with primary only.")
        return primary_df

    if fallback_df.empty:
        log(f"  WARNING: fallback source ({SIM_SOURCE_FALLBACK}) returned no data.")
        return primary_df

    # Keep only fallback rows that are OLDER than the earliest primary row.
    # This prevents any double-counting at the stitch boundary.
    primary_start = primary_df.index.min()
    fallback_older = fallback_df[fallback_df.index < primary_start]

    if fallback_older.empty:
        log(f"  WARNING: fallback data does not pre-date primary data — cannot stitch.")
        return primary_df

    combined = pd.concat([fallback_older, primary_df]).sort_index()
    combined.index = pd.DatetimeIndex(combined.index)
    log(f"  Stitched {len(fallback_older)} fallback rows + {len(primary_df)} primary rows "
        f"= {len(combined)} total rows "
        f"({fallback_older.index.min().strftime('%m-%d %H:%M')} → "
        f"{primary_df.index.max().strftime('%m-%d %H:%M %Z')})")
    return combined

# ------------------------------------------------------------------
# Data cleaning: interpolation + outlier clamping
# ------------------------------------------------------------------
def clean_series(series: pd.Series, kind: str, lookback: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    s = s.resample("1h").mean()
    s = s.interpolate(method="time").ffill().bfill()
    recent = s.tail(lookback)
    mean_val = float(recent.mean()) if len(recent) else 0.0
    max_val = float(recent.max()) if len(recent) else 0.0
    if kind == "load":
        upper = max(3.0 * mean_val, 1.15 * max_val, 1.0)
    elif kind == "solar":
        upper = max(5.0 * mean_val, 2.0 * max_val, 1.0)
    else:
        upper = max(3.0 * mean_val, 1.10 * max_val, 1.0)
    s = s.clip(lower=0.0, upper=upper)
    return s.tail(lookback)

# ------------------------------------------------------------------
# Feature construction and normalization
# Load/Wind use 6 linear features: [Target, Hour, Day, DayYear, Month, Season]
# Solar (v3) uses 6 cyclical features: [Target, Hour_sin, Hour_cos, DayYear_sin, DayYear_cos, Season]
# ------------------------------------------------------------------
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Hour"] = out.index.hour
    out["Day"] = out.index.day
    out["DayYear"] = out.index.day_of_year
    out["Month"] = out.index.month
    out["Season"] = out.index.quarter
    return out

def create_solar_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["DayYear_sin"] = np.sin(2 * np.pi * out.index.day_of_year / 365)
    out["DayYear_cos"] = np.cos(2 * np.pi * out.index.day_of_year / 365)
    out["Season"] = out.index.quarter.astype(float)
    return out

def resolve_norm_key(norm_params: Dict, candidates: List[str], label: str) -> str:
    for key in candidates:
        if key in norm_params:
            return key
    raise KeyError(f"No normalization key found for {label}. Tried: {', '.join(candidates)}")

def load_target_norm_key(norm_params: Dict) -> str:
    return resolve_norm_key(norm_params, ["LoadPower", "Load", "load"], "load target")

def solar_target_norm_key(norm_params: Dict) -> str:
    return resolve_norm_key(
        norm_params,
        ["RealPower", "RealPower (kW)", "SolarPower", "solar_power"],
        "solar target",
    )

def solar_cloud_norm_key(norm_params: Dict) -> str | None:
    for key in ("Cloud Type", "CloudType", "cloud_type"):
        if key in norm_params:
            return key
    return None

def wind_target_norm_key(norm_params: Dict) -> str:
    return resolve_norm_key(norm_params, ["WindPower", "Power (kW)", "wind_power"], "wind target")

def wind_speed_norm_key(norm_params: Dict) -> str:
    return resolve_norm_key(norm_params, ["wind_speed", "Wind speed (m/s)"], "wind speed")

def wind_temperature_norm_key(norm_params: Dict) -> str:
    return resolve_norm_key(
        norm_params,
        ["wind_temperature", "Ambient temperature (converter) (°C)", "Ambient temperature (°C)"],
        "wind temperature",
    )

def target_norm_key_for_kind(kind: str, norm_params: Dict) -> str:
    if kind == "load":
        return load_target_norm_key(norm_params)
    if kind == "solar":
        return solar_target_norm_key(norm_params)
    if kind == "wind":
        return wind_target_norm_key(norm_params)
    raise KeyError(f"Unknown kind for target norm resolution: {kind}")

# v5.2 one-hot cloud columns (cloud_type integer → 14 binary columns)
V52_CLOUD_COLUMNS = [
    "cloud_-15", "cloud_0", "cloud_1", "cloud_2", "cloud_3", "cloud_4",
    "cloud_5", "cloud_6", "cloud_7", "cloud_8", "cloud_9", "cloud_10",
    "cloud_11", "cloud_12",
]

def solar_is_v52(norm_params: Dict) -> bool:
    """True when norm_params contains the v5.2 one-hot cloud layout."""
    return "cloud_-15" in norm_params

def solar_uses_cloud_type(norm_params: Dict) -> bool:
    return solar_cloud_norm_key(norm_params) is not None

def clean_categorical_series(series: pd.Series, lookback: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    s = s.resample("1h").last()
    s = s.ffill().bfill()
    return s.tail(lookback)

def build_solar_feature_frame(
    history: pd.Series,
    norm_params: Dict,
    cloud_history: pd.Series | None = None,
) -> pd.DataFrame:
    target_col = solar_target_norm_key(norm_params)
    frame = history.to_frame(name=target_col)
    frame = create_solar_date_features(frame)

    if solar_is_v52(norm_params):
        # v5.2: 20-channel features — RealPower + 14 one-hot cloud columns + 5 time features
        # Sim cloud_type defaults to 0 → cloud_0=1, all others=0
        if cloud_history is not None and len(cloud_history):
            cloud_int = cloud_history.reindex(frame.index).ffill().bfill().fillna(0.0)
        else:
            cloud_int = pd.Series(0.0, index=frame.index)
        for col in V52_CLOUD_COLUMNS:
            # Parse integer from column name (e.g. "cloud_-15" → -15, "cloud_0" → 0)
            col_val = int(col.split("_", 1)[1])
            frame[col] = (cloud_int == col_val).astype(float)
        feature_cols = [target_col] + V52_CLOUD_COLUMNS + [
            "Hour_sin", "Hour_cos", "DayYear_sin", "DayYear_cos", "Season",
        ]
        frame = frame[feature_cols].copy()
        for col in feature_cols:
            frame[col] = normalize_series_with_params(frame[col], norm_params[col])
        return frame

    # v4 / legacy: 7 features (with cloud_type) or 6 features (without)
    feature_cols = [target_col, "Hour_sin", "Hour_cos", "DayYear_sin", "DayYear_cos", "Season"]
    cloud_key = solar_cloud_norm_key(norm_params)
    if cloud_key is not None:
        if cloud_history is not None and len(cloud_history):
            frame[cloud_key] = cloud_history.reindex(frame.index).ffill().bfill().fillna(0.0)
        else:
            frame[cloud_key] = 0.0
        feature_cols = [target_col, cloud_key, "Hour_sin", "Hour_cos", "DayYear_sin", "DayYear_cos", "Season"]
    frame = frame[feature_cols].copy()
    for col in feature_cols:
        frame[col] = normalize_series_with_params(frame[col], norm_params[col])
    return frame

def solar_normalize_single_row(
    target_val: float,
    ts: pd.Timestamp,
    norm_params: Dict,
    last_cloud_type: float | None = None,
) -> np.ndarray:
    target_col = solar_target_norm_key(norm_params)

    def norm(v: float, col: str) -> float:
        mn = float(norm_params[col]["min"])
        mx = float(norm_params[col]["max"])
        diff = mx - mn if mx != mn else 1.0
        return float(np.clip((v - mn) / diff, 0.0, 1.0))

    hour_sin = float(np.sin(2 * np.pi * ts.hour / 24))
    hour_cos = float(np.cos(2 * np.pi * ts.hour / 24))
    dayyear_sin = float(np.sin(2 * np.pi * ts.day_of_year / 365))
    dayyear_cos = float(np.cos(2 * np.pi * ts.day_of_year / 365))
    season = float(ts.quarter)

    if solar_is_v52(norm_params):
        # v5.2: 20 features — RealPower + 14 one-hot cloud cols + 5 time features
        # cloud_type defaults to 0 in sim → cloud_0=1, others=0
        cloud_int = 0 if last_cloud_type is None else int(last_cloud_type)
        cloud_vals = [1.0 if int(col.split("_", 1)[1]) == cloud_int else 0.0 for col in V52_CLOUD_COLUMNS]
        return np.array([
            norm(target_val, target_col),
            *cloud_vals,  # already binary [0,1], norm_params min=0 max=1 → no change
            norm(hour_sin, "Hour_sin"),
            norm(hour_cos, "Hour_cos"),
            norm(dayyear_sin, "DayYear_sin"),
            norm(dayyear_cos, "DayYear_cos"),
            norm(season, "Season"),
        ], dtype=np.float32)

    cloud_key = solar_cloud_norm_key(norm_params)
    if cloud_key is not None:
        cloud_val = 0.0 if last_cloud_type is None else float(last_cloud_type)
        return np.array([
            norm(target_val, target_col),
            norm(cloud_val, cloud_key),
            norm(hour_sin, "Hour_sin"),
            norm(hour_cos, "Hour_cos"),
            norm(dayyear_sin, "DayYear_sin"),
            norm(dayyear_cos, "DayYear_cos"),
            norm(season, "Season"),
        ], dtype=np.float32)

    return np.array([
        norm(target_val, target_col),
        norm(hour_sin, "Hour_sin"),
        norm(hour_cos, "Hour_cos"),
        norm(dayyear_sin, "DayYear_sin"),
        norm(dayyear_cos, "DayYear_cos"),
        norm(season, "Season"),
    ], dtype=np.float32)

def build_wind_feature_frame(
    wind_history: pd.Series,
    speed_history: pd.Series,
    temp_history: pd.Series,
    norm_params: Dict,
) -> pd.DataFrame:
    """Build normalised feature frame for wind v2."""
    target_key = wind_target_norm_key(norm_params)
    speed_key = wind_speed_norm_key(norm_params)
    temp_key = wind_temperature_norm_key(norm_params)

    frame = wind_history.to_frame(name=target_key)
    frame = create_date_features(frame)
    frame[speed_key] = speed_history.reindex(frame.index).ffill().bfill().fillna(0.0)
    frame[temp_key] = temp_history.reindex(frame.index).ffill().bfill().fillna(15.0)

    feature_cols = [target_key, speed_key, temp_key, "Hour", "Day", "DayYear", "Month", "Season"]
    frame = frame[feature_cols].copy()
    for col in feature_cols:
        if col in norm_params:
            nparams = norm_params[col]
        elif col in _TIME_FEATURE_DEFAULTS:
            nparams = _TIME_FEATURE_DEFAULTS[col]
        else:
            raise KeyError(f"No norm params for wind feature '{col}'")
        frame[col] = normalize_series_with_params(frame[col], nparams)
    return frame

def wind_normalize_single_row(
    target_val: float,
    ts: pd.Timestamp,
    norm_params: Dict,
    last_speed: float,
    last_temp: float,
) -> np.ndarray:
    """Normalise one row for wind v2 inference (8 features)."""
    target_key = wind_target_norm_key(norm_params)
    speed_key = wind_speed_norm_key(norm_params)
    temp_key = wind_temperature_norm_key(norm_params)

    def norm(v: float, col: str) -> float:
        p = norm_params.get(col) or _TIME_FEATURE_DEFAULTS.get(col)
        if p is None:
            raise KeyError(f"No norm params for '{col}'")
        mn = float(p["min"])
        mx = float(p["max"])
        diff = mx - mn if mx != mn else 1.0
        return float(np.clip((v - mn) / diff, 0.0, 1.0))

    return np.array([
        norm(target_val,            target_key),
        norm(last_speed,            speed_key),
        norm(last_temp,             temp_key),
        norm(float(ts.hour),        "Hour"),
        norm(float(ts.day),         "Day"),
        norm(float(ts.day_of_year), "DayYear"),
        norm(float(ts.month),       "Month"),
        norm(float(ts.quarter),     "Season"),
    ], dtype=np.float64)

def normalize_series_with_params(series: pd.Series, norm: Dict) -> pd.Series:
    mn = float(norm["min"])
    mx = float(norm["max"])
    diff = mx - mn if mx != mn else 1.0
    return ((series.astype(np.float64) - mn) / diff).clip(0.0, 1.0)

# Fixed ranges for time features when not present in norm_params.json
_TIME_FEATURE_DEFAULTS = {
    "Hour":    {"min": 0.0,  "max": 23.0},
    "Day":     {"min": 1.0,  "max": 31.0},
    "DayYear": {"min": 1.0,  "max": 366.0},
    "Month":   {"min": 1.0,  "max": 12.0},
    "Season":  {"min": 1.0,  "max": 4.0},
}

def build_feature_frame(history: pd.Series, kind: str, norm_params: Dict) -> pd.DataFrame:
    target_key = target_norm_key_for_kind(kind, norm_params)
    frame = history.to_frame(name=target_key)
    frame = create_date_features(frame)
    feature_cols = [target_key, "Hour", "Day", "DayYear", "Month", "Season"]
    frame = frame[feature_cols].copy()
    for col in feature_cols:
        if col in norm_params:
            nparams = norm_params[col]
        elif col in _TIME_FEATURE_DEFAULTS:
            nparams = _TIME_FEATURE_DEFAULTS[col]
        else:
            raise KeyError(f"No norm params found for feature '{col}'")
        frame[col] = normalize_series_with_params(frame[col], nparams)
    return frame

def normalize_single_row(target_val: float, ts: pd.Timestamp, kind: str, norm_params: Dict) -> np.ndarray:
    target_key = target_norm_key_for_kind(kind, norm_params)

    def norm(v: float, col: str) -> float:
        p = norm_params.get(col) or _TIME_FEATURE_DEFAULTS.get(col)
        if p is None:
            raise KeyError(f"No norm params for '{col}'")
        mn = float(p["min"])
        mx = float(p["max"])
        diff = mx - mn if mx != mn else 1.0
        return float(np.clip((v - mn) / diff, 0.0, 1.0))

    return np.array([
        norm(target_val, target_key),
        norm(float(ts.hour), "Hour"),
        norm(float(ts.day), "Day"),
        norm(float(ts.day_of_year), "DayYear"),
        norm(float(ts.month), "Month"),
        norm(float(ts.quarter), "Season"),
    ], dtype=np.float64)

def denorm_value(normed: float, norm: Dict) -> float:
    mn = float(norm["min"])
    mx = float(norm["max"])
    return float(normed) * (mx - mn) + mn

# ------------------------------------------------------------------
# Outlier clamp applied to each predicted step
# ------------------------------------------------------------------
def clamp_prediction(value: float, history: pd.Series, kind: str, lookback: int = LOOKBACK_HOURS) -> float:
    recent = history.tail(lookback)
    mean_val = float(recent.mean()) if len(recent) else 0.0
    max_val = float(recent.max()) if len(recent) else 0.0
    if kind == "load":
        upper = max(3.0 * mean_val, 1.15 * max_val, 1.0)
    elif kind == "solar":
        # Solar can swing dramatically day-to-day (cloudy→clear). The 24h mean is
        # depressed by nighttime zeros, making 3×mean too restrictive. Allow up to
        # 2× the recent peak so a clear day after a cloudy one isn't clamped.
        upper = max(5.0 * mean_val, 2.0 * max_val, 1.0)
    else:
        upper = max(3.0 * mean_val, 1.10 * max_val, 1.0)
    # For wind: also apply a short-window (6h) cap so that yesterday's high values
    # don't allow runaway predictions after a sharp drop. When wind has recently
    # fallen to near-zero the 24h mean/max still reflects yesterday's high period,
    # making the 24h cap too permissive. The short-window cap takes the minimum of
    # both, allowing gradual recovery while preventing extreme over-forecasting.
    if kind == "wind":
        short = history.tail(min(6, len(history)))
        if len(short) > 0:
            short_upper = max(4.0 * float(short.mean()), 1.5 * float(short.max()), 1.0)
            upper = min(upper, short_upper)
        # Declining-regime reactive cap: when the most recent real observation is
        # less than 50% of the 6h mean AND the 6h mean is above 50 kW (i.e. a
        # genuine high→low crash, not a naturally low-wind steady state), apply a
        # tight cap anchored to the last observation. This prevents the 6h short
        # window (which still contains pre-crash highs) from allowing extreme
        # over-prediction. Only fires in a declining regime, so upward ramps are
        # not suppressed.
        if len(short) >= 2:
            short_mean_val = float(short.mean())
            last_real = float(history.iloc[-1])
            if short_mean_val > 50.0 and last_real < 0.5 * short_mean_val:
                reactive_upper = max(3.0 * last_real, 5.0)
                upper = min(upper, reactive_upper)
    return float(np.clip(value, 0.0, upper))

# ------------------------------------------------------------------
# Recursive forecast
# ------------------------------------------------------------------
def recursive_forecast(
    kind: str,
    cleaned_series: pd.Series,
    model: nn.Module,
    norm_params: Dict,
    device: torch.device,
    aux_speed: pd.Series | None = None,
    aux_temp: pd.Series | None = None,
    aux_cloud_type: pd.Series | None = None,
    forecast_steps: int = FORECAST_STEPS,
) -> pd.DataFrame:
    history = cleaned_series.copy()
    rows: List[Dict] = []

    is_solar = kind == "solar" and "Hour_sin" in norm_params
    is_solar_v4 = is_solar and solar_uses_cloud_type(norm_params)
    is_wind_v2 = (
        kind == "wind"
        and "wind_speed" in norm_params
        and aux_speed is not None
        and aux_temp is not None
    )

    if is_solar:
        lookback = SOLAR_LOOKBACK_HOURS
    elif kind == "wind":
        lookback = WIND_LOOKBACK_HOURS
    else:
        lookback = LOOKBACK_HOURS

    if is_solar:
        feat_frame = build_solar_feature_frame(history, norm_params, cloud_history=aux_cloud_type)
        state: List[np.ndarray] = [feat_frame.iloc[i].to_numpy(dtype=np.float32) for i in range(len(feat_frame))]
        tensor_dtype = torch.float32
        target_norm_key = solar_target_norm_key(norm_params)
        last_cloud_type = None
        if is_solar_v4 and aux_cloud_type is not None and len(aux_cloud_type):
            last_cloud_type = float(aux_cloud_type.iloc[-1])
    elif is_wind_v2:
        feat_frame = build_wind_feature_frame(history, aux_speed, aux_temp, norm_params)
        state = [feat_frame.iloc[i].to_numpy(dtype=np.float64) for i in range(len(feat_frame))]
        tensor_dtype = torch.float64
        target_norm_key = wind_target_norm_key(norm_params)
        last_speed = float(aux_speed.iloc[-1]) if len(aux_speed) else 0.0
        last_temp = float(aux_temp.iloc[-1]) if len(aux_temp) else 15.0
        # Compute 1h trend from last 2 observations so recursive steps advance
        # wind conditions instead of holding them frozen. Trend is dampened by 0.5
        # per step to avoid runaway extrapolation beyond 1-2h.
        _speed_norm = norm_params.get("wind_speed") or {}
        _temp_norm  = norm_params.get("wind_temperature") or {}
        _speed_max = float(_speed_norm.get("max", 25.0))
        _temp_min  = float(_temp_norm.get("min", -5.0))
        _temp_max  = float(_temp_norm.get("max", 20.0))
        if len(aux_speed) >= 2:
            _speed_trend = float(aux_speed.iloc[-1]) - float(aux_speed.iloc[-2])
        else:
            _speed_trend = 0.0
        if len(aux_temp) >= 2:
            _temp_trend = float(aux_temp.iloc[-1]) - float(aux_temp.iloc[-2])
        else:
            _temp_trend = 0.0
    else:
        feat_frame = build_feature_frame(history, kind, norm_params)
        state = [feat_frame.iloc[i].to_numpy(dtype=np.float64) for i in range(len(feat_frame))]
        tensor_dtype = torch.float64 if kind in ("load", "wind") else torch.float32
        target_norm_key = target_norm_key_for_kind(kind, norm_params)

    # Snapshot of real observations before any predicted values are appended.
    # Passed to clamp_prediction for all steps so the cap is always anchored to
    # actual data, not to the model's own prior predictions (which can compound
    # error when steps 2 and 3 use an inflated step-1 output as their clamp reference).
    orig_history = history.copy()

    for step in range(1, forecast_steps + 1):
        window_dtype = np.float32 if is_solar else np.float64
        window = np.array(state[-lookback:], dtype=window_dtype)
        x = torch.tensor(window, dtype=tensor_dtype).unsqueeze(0).permute(0, 2, 1).to(device)
        with torch.no_grad():
            pred_norm = float(model(x).reshape(-1).detach().cpu().numpy()[0])
        pred_val = denorm_value(pred_norm, norm_params[target_norm_key])
        pred_val = clamp_prediction(pred_val, orig_history, kind, lookback)
        next_ts = history.index.max() + pd.Timedelta(hours=1)
        if kind == "solar" and (next_ts.hour < 6 or next_ts.hour > 20):
            pred_val = 0.0
        history.loc[next_ts] = pred_val
        if is_solar:
            next_row = solar_normalize_single_row(pred_val, next_ts, norm_params, last_cloud_type=last_cloud_type)
        elif is_wind_v2:
            next_row = wind_normalize_single_row(pred_val, next_ts, norm_params, last_speed, last_temp)
            # Advance exogenous inputs with dampened trend (0.5× each step)
            _speed_trend *= 0.5
            _temp_trend  *= 0.5
            last_speed = float(np.clip(last_speed + _speed_trend, 0.0, _speed_max))
            last_temp  = float(np.clip(last_temp  + _temp_trend,  _temp_min, _temp_max))
        else:
            next_row = normalize_single_row(pred_val, next_ts, kind, norm_params)
        state.append(next_row)
        rows.append({
            "timestamp": next_ts,
            PREDICTED_COLUMN_MAP[kind]: pred_val,
            "PredictedValue": pred_val,
            "horizon": "1h",
            "forecast_step_hours": step,
            "model": MODEL_NAME,
            "source": "simulations",
        })
        time.sleep(LOOP_DELAY)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# Same-hour-yesterday persistence blending
# ------------------------------------------------------------------
# Blend weights per signal. Load is highly persistent day-to-day
# (same building, same schedule); wind is least persistent.
PERSISTENCE_ALPHA = {
    "load":  0.70,   # 70% yesterday, 30% model — load is very stable day-to-day in buildings
    "solar": 0.50,   # 30% yesterday, 70% model — solar shape repeats, magnitude varies
    "wind":  0.00,   # 0% — wind is NOT day-to-day persistent, model-only is better
}

def query_yesterday_actual(cfg: Dict[str, str], kind: str, target_ts: pd.Timestamp) -> float | None:
    """Return the hourly-mean actual value from exactly 24h before target_ts.

    Returns None if the data point is missing or the query fails.
    """
    field_map = {"load": "load", "solar": "solar_power", "wind": "wind_power"}
    field = field_map[kind]
    # The window we want: [target_ts - 25h, target_ts - 23h) centred on -24h
    start = (target_ts - pd.Timedelta(hours=25)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    stop  = (target_ts - pd.Timedelta(hours=23)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    query = (
        f'from(bucket: "{cfg["sim_bucket"]}")\n'
        f'  |> range(start: {start}, stop: {stop})\n'
        f'  |> filter(fn: (r) => r["_measurement"] == "{cfg["sim_measurement"]}")\n'
        f'  |> filter(fn: (r) => r["_field"] == "{field}")\n'
        f'  |> filter(fn: (r) => r["source"] == "{cfg["sim_source"]}")\n'
        f'  |> mean()\n'
        f'  |> keep(columns: ["_value"])'
    )
    try:
        hdrs = {
            "Authorization": f'Token {cfg["influx_token"]}',
            "Accept":        "application/csv",
            "Content-Type":  "application/vnd.flux",
        }
        r = requests.post(
            f'{cfg["influx_url"].rstrip("/")}/api/v2/query',
            params={"org": cfg["influx_org"]},
            data=query.encode("utf-8"),
            headers=hdrs,
            timeout=30,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), comment="#")
        if df.empty or "_value" not in df.columns:
            return None
        vals = pd.to_numeric(df["_value"], errors="coerce").dropna()
        if vals.empty:
            return None
        v = float(vals.iloc[0])
        # Clamp negatives (e.g. solar can go slightly negative in powersim)
        return max(0.0, v)
    except Exception:
        return None

def blend_with_persistence(
    forecast_df: pd.DataFrame,
    cfg: Dict[str, str],
    kind: str,
) -> pd.DataFrame:
    """Replace each predicted value with:
         alpha * yesterday_same_hour_actual + (1-alpha) * model_prediction

    Falls back to model-only if yesterday data is unavailable.
    Solar is forced to 0 outside 06:00-20:00 regardless.
    """
    alpha = PERSISTENCE_ALPHA.get(kind, 0.0)
    if alpha == 0.0:
        return forecast_df

    predicted_col = PREDICTED_COLUMN_MAP[kind]
    df = forecast_df.copy()

    for i in df.index:
        ts       = pd.Timestamp(df.at[i, "timestamp"])
        hour     = ts.hour
        model_v  = float(df.at[i, "PredictedValue"])

        # Solar zero-guard (before blending so persistence doesn't resurrect nighttime solar)
        if kind == "solar" and (hour < 6 or hour > 20):
            df.at[i, "PredictedValue"] = 0.0
            df.at[i, predicted_col]    = 0.0
            continue

        yest_v = query_yesterday_actual(cfg, kind, ts)
        if yest_v is None:
            continue   # no yesterday data → keep model value unchanged

        blended = alpha * yest_v + (1.0 - alpha) * model_v
        blended = max(0.0, blended)
        df.at[i, "PredictedValue"] = blended
        df.at[i, predicted_col]    = blended

    return df

# ------------------------------------------------------------------
# Hour-of-day bias correction
# ------------------------------------------------------------------
def compute_bias_corrections(
    cfg: Dict[str, str],
    kind: str,
    days: int = BIAS_CORRECTION_DAYS,
    min_samples: int = BIAS_MIN_SAMPLES,
) -> Dict[int, float]:
    """Query the last `days` of actuals vs forecasts from InfluxDB, compute
    mean(actual - forecast) per hour of day, and return it as {hour: kw_offset}.

    Hours with fewer than `min_samples` data points get 0.0 (no correction).
    Returns an empty dict on any failure so callers degrade gracefully.
    """
    field_map = {"load": "load", "solar": "solar_power", "wind": "wind_power"}
    meas_map = {
        "load":  "forecast_load_series",
        "solar": "forecast_solar_series",
        "wind":  "forecast_wind_series",
    }
    field = field_map[kind]
    meas  = meas_map[kind]
    hdrs  = {
        "Authorization": f"Token {cfg['influx_token']}",
        "Accept":        "application/csv",
        "Content-Type":  "application/vnd.flux",
    }
    base = cfg["influx_url"].rstrip("/")

    def flux(q: str) -> pd.DataFrame:
        r = requests.post(f"{base}/api/v2/query", params={"org": cfg["influx_org"]},
                          data=q.encode("utf-8"), headers=hdrs, timeout=60)
        r.raise_for_status()
        try:
            df = pd.read_csv(io.StringIO(r.text), comment="#")
        except Exception:
            # InfluxDB returns only #-comment lines when there is no data,
            # which causes pd.read_csv to raise "No columns to parse from file"
            return pd.DataFrame()
        if df.empty or "_time" not in df.columns:
            return pd.DataFrame()
        df["_time"] = pd.to_datetime(df["_time"], format="ISO8601", utc=True, errors="coerce")
        df = df.dropna(subset=["_time"])
        if df.empty:
            return pd.DataFrame()
        df["_time"] = df["_time"].dt.tz_convert(TORONTO_TZ)
        return df

    now_utc = datetime.now(timezone.utc)
    hour_floor_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    stop_iso = hour_floor_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        df_a = flux(f'''
from(bucket: "{cfg["sim_bucket"]}")
  |> range(start: -{days}d, stop: {stop_iso})
  |> filter(fn: (r) => r["_measurement"] == "{cfg["sim_measurement"]}")
  |> filter(fn: (r) => r["_field"] == "{field}")
  |> filter(fn: (r) => r["source"] == "{cfg["sim_source"]}")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
  |> keep(columns: ["_time", "_value"])
''')
        df_f = flux(f'''
from(bucket: "{cfg["forecast_bucket"]}")
  |> range(start: -{days}d, stop: {stop_iso})
  |> filter(fn: (r) => r["_measurement"] == "{meas}")
  |> filter(fn: (r) => r["model"] == "{MODEL_NAME}")
  |> filter(fn: (r) => r["horizon"] == "1h")
  |> filter(fn: (r) => r["_field"] == "PredictedValue")
  |> aggregateWindow(every: 1h, fn: last, createEmpty: false, timeSrc: "_start")
  |> keep(columns: ["_time", "_value"])
''')

        if df_a.empty or df_f.empty:
            return {}

        # Join on floored-to-hour timestamps
        s_a = df_a.set_index(df_a["_time"].dt.floor("h"))["_value"].rename("actual")
        s_f = df_f.set_index(df_f["_time"].dt.floor("h"))["_value"].rename("forecast")
        joined = pd.concat([s_a, s_f], axis=1).dropna()
        if joined.empty:
            return {}

        joined["bias"] = joined["actual"] - joined["forecast"]
        joined["hour"] = joined.index.hour
        stats = joined.groupby("hour")["bias"].agg(["mean", "count"])

        corrections: Dict[int, float] = {}
        for hour, row in stats.iterrows():
            if int(row["count"]) >= min_samples:
                corrections[int(hour)] = float(row["mean"])
        return corrections

    except Exception as exc:
        log(f"  WARNING: bias correction query failed ({exc}). Running without correction.")
        return {}


def compute_and_log_errors(cfg: Dict[str, str]) -> None:
    """At run time H:05, H-1 is now complete.
    Query actual(H-1) from sim bucket and forecast(H-1) from forecast bucket,
    compute signed error (actual - forecast), append to error log CSV.
    """
    now_utc        = datetime.now(timezone.utc)
    hour_floor_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    prev_hour_utc  = hour_floor_utc - pd.Timedelta(hours=1)
    start_iso = prev_hour_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_iso  = hour_floor_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    hdrs = {
        "Authorization": f"Token {cfg['influx_token']}",
        "Accept":        "application/csv",
        "Content-Type":  "application/vnd.flux",
    }
    base = cfg["influx_url"].rstrip("/")

    def flux(q: str) -> pd.DataFrame:
        r = requests.post(f"{base}/api/v2/query", params={"org": cfg["influx_org"]},
                          data=q.encode("utf-8"), headers=hdrs, timeout=60)
        r.raise_for_status()
        try:
            df = pd.read_csv(io.StringIO(r.text), comment="#")
        except Exception:
            return pd.DataFrame()
        if df.empty or "_value" not in df.columns:
            return pd.DataFrame()
        return df

    field_map = {"load": "load", "solar": "solar_power", "wind": "wind_power"}
    meas_map  = {"load": "forecast_load_series", "solar": "forecast_solar_series", "wind": "forecast_wind_series"}
    run_time_str     = datetime.now(zoneinfo.ZoneInfo(TORONTO_TZ)).strftime("%Y-%m-%d %H:%M")
    forecast_for_str = prev_hour_utc.astimezone(zoneinfo.ZoneInfo(TORONTO_TZ)).strftime("%Y-%m-%d %H:%M")

    new_rows = []
    for kind in ("load", "solar", "wind"):
        try:
            r_a = flux(f'''
from(bucket: "{cfg["sim_bucket"]}")
  |> range(start: {start_iso}, stop: {stop_iso})
  |> filter(fn: (r) => r["_measurement"] == "{cfg["sim_measurement"]}")
  |> filter(fn: (r) => r["_field"] == "{field_map[kind]}")
  |> filter(fn: (r) => r["source"] == "{cfg["sim_source"]}")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
  |> keep(columns: ["_value"])
''')
            r_f = flux(f'''
from(bucket: "{cfg["forecast_bucket"]}")
  |> range(start: {start_iso}, stop: {stop_iso})
  |> filter(fn: (r) => r["_measurement"] == "{meas_map[kind]}")
  |> filter(fn: (r) => r["model"] == "{MODEL_NAME}")
  |> filter(fn: (r) => r["horizon"] == "1h")
  |> filter(fn: (r) => r["_field"] == "PredictedValue")
  |> last()
  |> keep(columns: ["_value"])
''')
            if r_a.empty or r_f.empty:
                continue
            actual_kw    = float(r_a["_value"].iloc[0])
            forecast_kw  = float(r_f["_value"].iloc[0])
            error_kw     = actual_kw - forecast_kw
            new_rows.append({
                "run_time": run_time_str, "forecast_for": forecast_for_str, "kind": kind,
                "actual_kw": round(actual_kw, 2), "forecast_kw": round(forecast_kw, 2),
                "error_kw": round(error_kw, 2), "abs_error_kw": round(abs(error_kw), 2),
            })
            log(f"  [error log] {kind:5s}: actual={actual_kw:.1f}  forecast={forecast_kw:.1f}  error={error_kw:+.1f} kW")
        except Exception as exc:
            log(f"  [error log] {kind}: query failed ({exc})")

    if new_rows:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(new_rows, columns=ERROR_LOG_COLUMNS)
        if ERROR_LOG_PATH.exists():
            new_df.to_csv(ERROR_LOG_PATH, mode="a", header=False, index=False)
        else:
            new_df.to_csv(ERROR_LOG_PATH, mode="w", header=True, index=False)


def apply_bias_correction(
    forecast_df: pd.DataFrame,
    corrections: Dict[int, float],
    kind: str,
) -> pd.DataFrame:
    """Apply additive hour-of-day bias correction to a forecast DataFrame.

    For each predicted row:
      corrected = raw_prediction + corrections[hour]
    Solar predictions are forced to 0 outside daylight hours (06:00–20:00)
    regardless of the correction.  All values are clamped to >= 0.
    """
    if not corrections:
        return forecast_df

    predicted_col = PREDICTED_COLUMN_MAP[kind]
    df = forecast_df.copy()

    for i in df.index:
        ts      = pd.Timestamp(df.at[i, "timestamp"])
        hour    = ts.hour
        offset  = corrections.get(hour, 0.0)
        if offset == 0.0:
            continue
        new_val = float(df.at[i, "PredictedValue"]) + offset
        # Re-zero solar outside daylight after correction
        if kind == "solar" and (hour < 6 or hour > 20):
            new_val = 0.0
        new_val = max(0.0, new_val)
        df.at[i, "PredictedValue"] = new_val
        df.at[i, predicted_col]    = new_val

    return df

def build_total_renewable(solar_df: pd.DataFrame, wind_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["timestamp", "horizon", "forecast_step_hours", "model", "source"]
    merged = solar_df.merge(wind_df, on=join_cols, how="inner", suffixes=("_solar", "_wind"))
    merged["PredictedTotalRenewablePower"] = merged["PredictedSolarPower"] + merged["PredictedWindPower"]
    merged["PredictedValue"] = merged["PredictedTotalRenewablePower"]
    return merged[["timestamp", "PredictedTotalRenewablePower", "PredictedValue", "horizon", "forecast_step_hours", "model", "source"]].copy()

def build_grid_pull(load_df: pd.DataFrame, total_renewable_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["timestamp", "horizon", "forecast_step_hours", "model", "source"]
    merged = load_df.merge(total_renewable_df, on=join_cols, how="inner", suffixes=("_load", "_renew"))
    merged["PredictedGridPull"] = (merged["PredictedValue_load"] - merged["PredictedValue_renew"]).clip(lower=0.0)
    merged["PredictedValue"] = merged["PredictedGridPull"]
    return merged[["timestamp", "PredictedGridPull", "PredictedValue", "horizon", "forecast_step_hours", "model", "source"]].copy()

# ------------------------------------------------------------------
# InfluxDB upload (line protocol)
# ------------------------------------------------------------------
def escape_tag(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ").replace("=", "\\=")

def escape_measurement(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ")

def df_to_line_protocol(df: pd.DataFrame, measurement: str, value_column: str) -> str:
    lines: List[str] = []
    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(TORONTO_TZ)
        ts_utc = ts.tz_convert("UTC")
        ns = int(ts_utc.value)
        horizon = escape_tag(str(row.horizon))
        model = escape_tag(str(row.model))
        source = escape_tag(str(row.source))
        step = int(row.forecast_step_hours)
        pv = float(row.PredictedValue)
        nv = float(getattr(row, value_column))
        lines.append(
            f"{escape_measurement(measurement)},horizon={horizon},model={model},source={source} "
            f"PredictedValue={pv:.8f},{value_column}={nv:.8f},forecast_step_hours={step}i "
            f"{ns}"
        )
        time.sleep(LOOP_DELAY)
    return "\n".join(lines)

def delete_forecast_window(
    session: requests.Session,
    base_url: str,
    org: str,
    bucket: str,
    token: str,
    measurement: str,
    start: str,
    stop: str,
) -> None:
    url = f"{base_url.rstrip('/')}/api/v2/delete"
    payload = {
        "start": start,
        "stop": stop,
        "predicate": f'_measurement="{measurement}" AND model="{MODEL_NAME}"',
    }
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    resp = session.post(url, params={"org": org, "bucket": bucket}, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

def write_line_protocol(
    session: requests.Session,
    base_url: str,
    org: str,
    bucket: str,
    token: str,
    payload: str,
) -> None:
    url = f"{base_url.rstrip('/')}/api/v2/write"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "text/plain; charset=utf-8",
        "Accept": "application/json",
    }
    resp = session.post(
        url,
        params={"org": org, "bucket": bucket, "precision": "ns"},
        data=payload.encode("utf-8"),
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()

def upload_forecasts(
    base_url: str,
    org: str,
    token: str,
    forecast_bucket: str,
    load_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    total_df: pd.DataFrame,
    grid_pull_df: pd.DataFrame,
) -> None:
    session = requests.Session()
    frames = [
        ("forecast_load_series", "PredictedLoadPower", load_df),
        ("forecast_solar_series", "PredictedSolarPower", solar_df),
        ("forecast_wind_series", "PredictedWindPower", wind_df),
        ("forecast_total_renewable_series", "PredictedTotalRenewablePower", total_df),
        ("forecast_grid_pull_series", "PredictedGridPull", grid_pull_df),
    ]
    for measurement, value_col, df in frames:
        if df.empty:
            log(f"Skipping empty dataframe for {measurement}")
            continue
        payload = df_to_line_protocol(df, measurement, value_col)
        write_line_protocol(session, base_url, org, forecast_bucket, token, payload)
        log(f"Uploaded {len(df)} rows to {measurement}")

# ------------------------------------------------------------------
# Save CSV outputs
# ------------------------------------------------------------------
def save_outputs(
    load_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    total_df: pd.DataFrame,
    grid_pull_df: pd.DataFrame,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    load_df.to_csv(OUTPUTS_DIR / "forecast_load_series.csv", index=False)
    solar_df.to_csv(OUTPUTS_DIR / "forecast_solar_series.csv", index=False)
    wind_df.to_csv(OUTPUTS_DIR / "forecast_wind_series.csv", index=False)
    total_df.to_csv(OUTPUTS_DIR / "forecast_total_renewable_series.csv", index=False)
    grid_pull_df.to_csv(OUTPUTS_DIR / "forecast_grid_pull_series.csv", index=False)

# ------------------------------------------------------------------
# Single inference run
# ------------------------------------------------------------------
def run_inference(cfg: Dict[str, str], device: torch.device, models: Dict) -> None:
    log("Querying simulation data from InfluxDB...")
    live_df = query_simulations(
        base_url=cfg["influx_url"],
        org=cfg["influx_org"],
        token=cfg["influx_token"],
        sim_bucket=cfg["sim_bucket"],
        sim_measurement=cfg["sim_measurement"],
        sim_source=cfg["sim_source"],
        sim_url=cfg["sim_url"],
        hours_back=QUERY_HOURS_BACK,
    )
    # Drop any data beyond the current hour boundary to prevent
    # partial-window contamination that shifts forecasts forward.
    current_hour = pd.Timestamp.now(tz=TORONTO_TZ).floor("h")
    live_df = live_df[live_df.index < current_hour]
    log(f"Data range: {live_df.index.min()} → {live_df.index.max()} ({len(live_df)} rows)")

    cleaned_load = clean_series(live_df[SIM_FIELD_MAP["load"]], "load", LOOKBACK_HOURS)
    cleaned_solar = clean_series(live_df[SIM_FIELD_MAP["solar"]], "solar", SOLAR_LOOKBACK_HOURS)
    cleaned_wind = clean_series(live_df[SIM_FIELD_MAP["wind"]], "wind", WIND_LOOKBACK_HOURS)

    wind_speed_field = SIM_FIELD_MAP["wind_speed"]
    cleaned_speed = (
        clean_series(live_df[wind_speed_field], "wind", WIND_LOOKBACK_HOURS)
        if wind_speed_field in live_df.columns and live_df[wind_speed_field].notna().any()
        else None
    )
    wind_temp_field = SIM_FIELD_MAP["wind_temperature"]
    cleaned_temp = (
        clean_series(live_df[wind_temp_field], "wind", WIND_LOOKBACK_HOURS)
        if wind_temp_field in live_df.columns and live_df[wind_temp_field].notna().any()
        else None
    )

    load_model, load_norm = models["load"]
    solar_model, solar_norm = models["solar"]
    wind_model, wind_norm = models["wind"]

    cleaned_cloud = None
    cloud_field = SIM_FIELD_MAP.get("cloud_type")
    if solar_uses_cloud_type(solar_norm):
        if cloud_field in live_df.columns and live_df[cloud_field].notna().any():
            cleaned_cloud = clean_categorical_series(live_df[cloud_field], SOLAR_LOOKBACK_HOURS)
            log(f"  cloud_type: available ({cleaned_cloud.notna().sum()} non-null points)")
        else:
            default_cloud = float(cfg.get("solar_cloud_type_default", "0"))
            cleaned_cloud = pd.Series(default_cloud, index=cleaned_solar.index, dtype=np.float64)
            log(f"  cloud_type: not found in simulation data, using default={default_cloud}")

    for aux_name, aux_series in [("wind_speed", cleaned_speed), ("wind_temperature", cleaned_temp)]:
        if aux_series is not None:
            log(f"  {aux_name}: available ({aux_series.notna().sum()} non-null points)")
        else:
            log(f"  {aux_name}: not found in simulation data")

    if len(cleaned_load) < LOOKBACK_HOURS or len(cleaned_solar) < SOLAR_LOOKBACK_HOURS or len(cleaned_wind) < WIND_LOOKBACK_HOURS:
        raise ValueError(f"Insufficient data after cleaning: load={len(cleaned_load)}, solar={len(cleaned_solar)}, wind={len(cleaned_wind)}")

    log(f"Running forecast ({LIVE_INTERNAL_FORECAST_STEPS} internal steps, publishing +{PUBLIC_FORECAST_OFFSET_HOURS}h) on device={device}...")
    load_forecast = recursive_forecast(
        "load",
        cleaned_load,
        load_model,
        load_norm,
        device,
        forecast_steps=LIVE_INTERNAL_FORECAST_STEPS,
    )
    solar_forecast = recursive_forecast(
        "solar",
        cleaned_solar,
        solar_model,
        solar_norm,
        device,
        aux_cloud_type=cleaned_cloud,
        forecast_steps=LIVE_INTERNAL_FORECAST_STEPS,
    )
    wind_forecast = recursive_forecast(
        "wind",
        cleaned_wind,
        wind_model,
        wind_norm,
        device,
        aux_speed=cleaned_speed,
        aux_temp=cleaned_temp,
        forecast_steps=LIVE_INTERNAL_FORECAST_STEPS,
    )

    # Persistence blending then bias correction improve accuracy before writing
    log("Applying persistence blending (same-hour-yesterday)...")
    load_forecast  = blend_with_persistence(load_forecast,  cfg, "load")
    solar_forecast = blend_with_persistence(solar_forecast, cfg, "solar")
    wind_forecast  = blend_with_persistence(wind_forecast,  cfg, "wind")

    log("Computing hour-of-day bias corrections...")
    load_corrections  = compute_bias_corrections(cfg, "load")
    solar_corrections = compute_bias_corrections(cfg, "solar")
    wind_corrections  = compute_bias_corrections(cfg, "wind")
    for label, corr in [("load", load_corrections), ("solar", solar_corrections), ("wind", wind_corrections)]:
        if corr:
            hr_str = "  ".join(f"{h}:{v:+.1f}" for h, v in sorted(corr.items()))
            log(f"  {label:5s} bias corrections (kW/hour): {hr_str}")
        else:
            log(f"  {label:5s}: no bias corrections yet (need {BIAS_MIN_SAMPLES}+ samples/hour)")
    load_forecast  = apply_bias_correction(load_forecast,  load_corrections,  "load")
    solar_forecast = apply_bias_correction(solar_forecast, solar_corrections, "solar")
    wind_forecast  = apply_bias_correction(wind_forecast,  wind_corrections,  "wind")

    # Regime-aware wind cap: when wind crashes from high to near-zero, positive bias
    # corrections learned from high-wind periods inflate the forecast after the clamp.
    # Two-tier check so the cap fires after the FIRST crashed hour (tier-1), not just
    # after 3 consecutive crashed hours (tier-2 / original behaviour).
    # Tier-1: single-hour crash — last observation < 25% of 24h mean.
    # Tier-2: multi-hour crash — 3h mean < 15% of 24h mean (original logic, retained).
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

    # Shift ALL steps by -1h so they land within Grafana's current time window:
    #   step-1 at H-1  (chart, aligns with completed actual H-1)
    #   step-2 at H    (current-hour overlay)
    #   step-3 at H+1  (next-hour forecast, visible on chart and "Next Hour" gauges)
    _shift = pd.Timedelta(hours=FORECAST_TIMESTAMP_OFFSET_HOURS)
    for _df in [load_forecast, solar_forecast, wind_forecast, total_forecast, grid_pull_forecast]:
        _df["timestamp"] = _df["timestamp"] + _shift

    save_outputs(load_forecast, solar_forecast, wind_forecast, total_forecast, grid_pull_forecast)
    log("Logging forecast errors for last hour...")
    compute_and_log_errors(cfg)
    log("Uploading forecasts to InfluxDB...")
    upload_forecasts(
        base_url=cfg["influx_url"],
        org=cfg["influx_org"],
        token=cfg["influx_token"],
        forecast_bucket=cfg["forecast_bucket"],
        load_df=load_forecast,
        solar_df=solar_forecast,
        wind_df=wind_forecast,
        total_df=total_forecast,
        grid_pull_df=grid_pull_forecast,
    )
    log("Inference run complete.")

# ------------------------------------------------------------------
# Daemon loop
# ------------------------------------------------------------------
def seconds_until_next_hour() -> float:
    """Wait until 5 minutes past the next hour.

    Running at H:05 instead of H:00 ensures the previous hour's data
    has fully landed in InfluxDB before the model queries it.
    """
    now = datetime.now(timezone.utc)
    seconds_past = now.minute * 60 + now.second + now.microsecond / 1_000_000
    target_offset = 300.0  # 5 minutes
    if seconds_past < target_offset:
        return target_offset - seconds_past
    return max(0.0, 3600.0 - seconds_past + target_offset)

def main() -> None:
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
        "solar_cloud_type_default": get_setting("SOLAR_CLOUD_TYPE_DEFAULT", env_values, "0"),
    }
    if not cfg["influx_org"]:
        raise ValueError("Missing INFLUX_ORG in environment or .env file")
    if not cfg["influx_token"]:
        raise ValueError("Missing INFLUX_TOKEN in environment or .env file")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting inference daemon | device={device} | lookback={LOOKBACK_HOURS}h | publish_offset=+{PUBLIC_FORECAST_OFFSET_HOURS}h")
    log("Loading model checkpoints...")
    models = {
        "load": load_model_assets("load", device),
        "solar": load_model_assets("solar", device),
        "wind": load_model_assets("wind", device),
    }
    log("Models loaded.")

    # State file prevents duplicate writes if the service restarts mid-hour.
    state_file = LOG_DIR / ".last_forecast_hour"

    def read_last_hour() -> datetime | None:
        try:
            text = state_file.read_text(encoding="utf-8").strip()
            return datetime.fromisoformat(text)
        except Exception:
            return None

    def write_last_hour(dt: datetime) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        state_file.write_text(dt.isoformat(), encoding="utf-8")

    while True:
        _tz = zoneinfo.ZoneInfo(TORONTO_TZ)
        current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        current_hour_local = current_hour.astimezone(_tz).strftime("%Y-%m-%d %H:%M %Z")
        last_forecast_hour = read_last_hour()
        if last_forecast_hour == current_hour:
            log(f"Already ran for {current_hour_local} — skipping until next hour")
        else:
            try:
                run_inference(cfg, device, models)
                write_last_hour(current_hour)
            except Exception as exc:
                log(f"ERROR during inference run: {exc}")
        wait = seconds_until_next_hour()
        log(f"Next run in {wait:.0f}s (top of next hour)")
        time.sleep(wait)

if __name__ == "__main__":
    main()
