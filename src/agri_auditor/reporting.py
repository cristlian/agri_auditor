"""Step 4: Mission Control Dashboard — interactive HTML report generator.

Generates a self-contained ``audit_report.html`` with:
  - Production-ready 'Cockpit' layout fitting 1080p viewport without scrolling
  - Leaflet.js dark-theme map with GPS path overlay and event markers
  - Synced Plotly charts (Velocity, Pitch/Roll, Clearance) sharing exact X-axis
  - Cross-dimensional interactivity: hover-sync between map, charts, event feed
  - MLOps triage UI with clipboard payload export and toast notifications
  - AI Event Cards with Gemini captions and signal decomposition

Architectural Principles:
  1. Viewport Density — CSS Grid cockpit layout, no page scrolling
  2. Semantic Grounding — Leaflet map with dark tiles, path + event markers
  3. Cross-Dimensional Interactivity — Global state linking all components
  4. MLOps Actionability — Triage buttons, JSON payloads, clipboard + toast
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    from jinja2 import Template
except ImportError as exc:
    raise ImportError(
        "jinja2 is required for report generation: pip install jinja2"
    ) from exc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:
    raise ImportError(
        "plotly is required for report generation: pip install plotly"
    ) from exc

from .ingestion import LogLoader
from .intelligence import Event, EventDetector, UNAVAILABLE_CAPTION


# ─── Color Palette — "Telemetry Command" Theme ──────────────────────────────
# Custom theme inspired by theme-factory's Tech Innovation + Ocean Depths,
# designed for a retro-futuristic mission control aesthetic.
# (frontend-design skill: bold, distinctive, cohesive palette with dominant
#  cyan accent — NOT generic purple gradients or timid distributions)

COLORS = {
    "bg_primary": "#060a13",
    "bg_surface": "#0d1321",
    "bg_elevated": "#162036",
    "border": "rgba(0, 229, 255, 0.12)",
    "border_light": "rgba(0, 229, 255, 0.06)",
    "grid": "rgba(0, 229, 255, 0.08)",
    "text": "#e0e6ed",
    "text_dim": "#7b8ca3",
    "text_muted": "#4a5568",
    "green": "#00e676",
    "orange": "#ff9100",
    "blue": "#448aff",
    "red": "#ff1744",
    "amber": "#ffd740",
    "purple": "#d500f9",
    "cyan": "#00e5ff",
}

# ─── Severity Scoring Thresholds (code-refactoring: replace magic numbers) ───

SEVERITY_WARNING_THRESHOLD = 0.4
SEVERITY_CRITICAL_THRESHOLD = 0.7

EVENT_TYPE_COLORS: dict[str, str] = {
    "roughness": COLORS["orange"],
    "proximity": COLORS["blue"],
    "steering": COLORS["amber"],
    "sensor_fault": COLORS["red"],
    "localization_fault": COLORS["purple"],
    "mixed": COLORS["text_dim"],
}

SIGNAL_COLORS: dict[str, str] = {
    "Roughness": COLORS["orange"],
    "Proximity": COLORS["blue"],
    "Yaw Rate": COLORS["amber"],
    "IMU Fault": COLORS["red"],
    "Localization": COLORS["purple"],
}

# ─── Image Encoding Defaults ─────────────────────────────────────────────────

DEFAULT_IMAGE_MAX_WIDTH = 640
DEFAULT_THUMB_MAX_WIDTH = 200
DEFAULT_IMAGE_QUALITY = 82
DEFAULT_INCLUDE_SURROUND = True

# ─── Chart Dimensions ────────────────────────────────────────────────────────

CHART_TELEMETRY_HEIGHT = 870
CHART_SPARKLINE_HEIGHT = 55

_PLACEHOLDER_IMAGE = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">'
        b'<rect width="640" height="400" fill="#162036"/>'
        b'<text x="320" y="200" text-anchor="middle" fill="#4a5568" '
        b'font-family="monospace" font-size="14">No Image Available</text></svg>'
    ).decode("ascii")
)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _severity_color(score: float) -> str:
    s = max(0.0, min(1.0, float(score)))
    if s < SEVERITY_WARNING_THRESHOLD:
        return COLORS["green"]
    if s < SEVERITY_CRITICAL_THRESHOLD:
        return COLORS["amber"]
    return COLORS["red"]


def _severity_label(score: float) -> str:
    s = max(0.0, min(1.0, float(score)))
    if s < SEVERITY_WARNING_THRESHOLD:
        return "NOMINAL"
    if s < SEVERITY_CRITICAL_THRESHOLD:
        return "WARNING"
    return "CRITICAL"


def _fmt(value: Any, precision: int = 2, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(f):
        return fallback
    return f"{f:.{precision}f}"


def _encode_image_b64(
    image_path: str | Path,
    max_width: int = DEFAULT_IMAGE_MAX_WIDTH,
    quality: int = DEFAULT_IMAGE_QUALITY,
) -> str | None:
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize(
                    (max_width, int(img.height * ratio)), Image.LANCZOS
                )
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
    except (OSError, ValueError):
        return None


def _elapsed(ts_series: pd.Series) -> pd.Series:
    ts = pd.to_numeric(ts_series, errors="coerce")
    start = ts.min()
    if pd.isna(start):
        return ts
    return (ts - start).astype("float64")


# ── Plotly helpers ────────────────────────────────────────────────────────────
# (frontend-design: distinctive typography — Space Grotesk / IBM Plex Mono,
#  NOT the overused Inter / JetBrains Mono)


def _axis(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "gridcolor": COLORS["grid"],
        "zerolinecolor": COLORS["border"],
        "showgrid": True,
        "gridwidth": 1,
    }
    base.update(kw)
    return base


def _layout(**kw: Any) -> dict[str, Any]:
    """Plotly layout defaults with Telemetry Command typography."""
    base: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": COLORS["bg_primary"],
        "plot_bgcolor": COLORS["bg_primary"],
        "font": dict(
            family="'Space Grotesk','DM Sans',system-ui,sans-serif",
            color=COLORS["text_dim"],
            size=14,
        ),
        "hoverlabel": dict(
            bgcolor=COLORS["bg_elevated"],
            bordercolor=COLORS["border"],
            font=dict(
                family="'IBM Plex Mono','Fira Code','Consolas',monospace",
                size=14,
                color=COLORS["text"],
            ),
        ),
        "legend": dict(
            bgcolor="rgba(13,19,33,0.85)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=12),
        ),
        "margin": dict(l=55, r=20, t=35, b=35),
    }
    base.update(kw)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# ReportBuilder
# ═══════════════════════════════════════════════════════════════════════════════


class ReportBuilder:
    """Generate an interactive Mission Control HTML dashboard."""

    def __init__(
        self,
        loader: LogLoader,
        features_df: pd.DataFrame,
        events: list[Event],
        metadata: dict[str, Any] | None = None,
        include_surround: bool = DEFAULT_INCLUDE_SURROUND,
        image_max_width: int = DEFAULT_IMAGE_MAX_WIDTH,
        thumb_max_width: int = DEFAULT_THUMB_MAX_WIDTH,
        image_quality: int = DEFAULT_IMAGE_QUALITY,
    ) -> None:
        self.loader = loader
        self.features_df = features_df.copy()
        self.events = events
        self.metadata = metadata or {}
        self.include_surround = include_surround
        self.image_max_width = image_max_width
        self.thumb_max_width = thumb_max_width
        self.image_quality = image_quality

        self.features_df["_elapsed"] = _elapsed(self.features_df["timestamp_sec"])
        ts = pd.to_numeric(self.features_df["timestamp_sec"], errors="coerce")
        self._t0 = float(ts.min()) if ts.notna().any() else 0.0

    # ── Public API ────────────────────────────────────────────────────────

    def save_report(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.write_text(self._render(), encoding="utf-8")
        return out

    # ── Private rendering ─────────────────────────────────────────────────

    def _render(self) -> str:
        events_ctx = [self._event_ctx(e) for e in self.events]
        gps_path = self._gps_path_data()
        ctx: dict[str, Any] = {
            "title": self.metadata.get("title", "Agri-Auditor Mission Control"),
            "run_id": self.metadata.get("run_id", "audit-run"),
            "generated_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "summary": self._summary(),
            "telemetry_json": self._chart_telemetry().to_json(),
            "events": events_ctx,
            "events_json_str": json.dumps(events_ctx, default=str),
            "gps_path_json": json.dumps(gps_path),
            "features_json": self.features_df.to_json(orient="records"),
        }
        return Template(_TEMPLATE).render(**ctx)

    # ── Summary KPIs ──────────────────────────────────────────────────────

    def _summary(self) -> dict[str, Any]:
        df = self.features_df
        elapsed = df["_elapsed"]
        vel = pd.to_numeric(df.get("velocity_mps"), errors="coerce")
        depth_col = df.get("has_depth")
        depth_n = 0
        if depth_col is not None:
            depth_n = int(pd.Series(depth_col).astype(bool).sum())
        total = len(df)
        sevs = [e.severity_score for e in self.events]
        max_sev = max(sevs) if sevs else 0.0
        return {
            "frames": f"{total:,}",
            "duration": _fmt(elapsed.max(), 1) + "s",
            "events": str(len(self.events)),
            "avg_speed": _fmt(vel.mean(), 2) + " m/s",
            "max_severity": _fmt(max_sev, 2),
            "max_severity_color": _severity_color(max_sev),
            "max_severity_label": _severity_label(max_sev),
            "depth_pct": _fmt(depth_n / total * 100 if total else 0, 1) + "%",
        }

    # ── Chart: Main Telemetry (Velocity, Pitch/Roll, Clearance) ─────────

    def _chart_telemetry(self) -> go.Figure:
        """4-row synced chart: Severity, Velocity, Pitch/Roll, Min Clearance."""
        df = self.features_df
        x = df["_elapsed"]

        if "severity_score" not in df.columns:
            scored = EventDetector().score_dataframe(df)
            sev = pd.to_numeric(
                scored["severity_score"], errors="coerce"
            ).fillna(0.0)
        else:
            sev = pd.to_numeric(
                df["severity_score"], errors="coerce"
            ).fillna(0.0)

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.15, 0.3, 0.25, 0.3],
            subplot_titles=(
                "Severity Timeline", "Velocity (m/s)", "Pitch / Roll (deg)", "Min Clearance (m)",
            ),
        )
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=16)

        # Row 1: Severity
        fig.add_trace(
            go.Scattergl(
                x=x, y=sev, mode="lines", name="Severity",
                line=dict(color=COLORS["amber"], width=1),
                fill="tozeroy", fillcolor="rgba(255,215,64,0.15)",
                hovertemplate="<b>%{y:.2f}</b><extra></extra>",
            ), row=1, col=1,
        )

        ev_x = []
        ev_sev_y = []
        ev_vel_y = []
        ev_colors = []
        ev_texts = []

        for ev in self.events:
            et = ev.timestamp_sec - self._t0
            idx_nearest = (df["_elapsed"] - et).abs().idxmin()
            vel_val = pd.to_numeric(
                df.loc[[idx_nearest], "velocity_mps"], errors="coerce"
            ).iloc[0]
            if not np.isfinite(vel_val):
                vel_val = 0.0

            ev_x.append(et)
            ev_sev_y.append(ev.severity_score)
            ev_vel_y.append(float(vel_val))
            ev_colors.append(_severity_color(ev.severity_score))
            ev_texts.append(f"<b>Event #{ev.event_rank}</b><br>Severity: {ev.severity_score:.2f}")

        fig.add_trace(
            go.Scatter(
                x=ev_x, y=ev_sev_y, mode="markers",
                marker=dict(color=ev_colors, size=6),
                text=ev_texts,
                hoverinfo="skip", showlegend=False,
            ), row=1, col=1
        )

        # Row 2: Velocity
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=pd.to_numeric(df.get("velocity_mps"), errors="coerce"),
                mode="lines", name="Velocity",
                line=dict(color=COLORS["green"], width=1.5),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.07)",
                hovertemplate="<b>%{y:.2f}</b> m/s<extra></extra>",
            ), row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ev_x, y=ev_vel_y,
                mode="markers",
                marker=dict(
                    color=ev_colors,
                    size=10, symbol="diamond",
                    line=dict(color="#fff", width=1),
                ),
                text=ev_texts,
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
            ), row=2, col=1,
        )

        # Row 3: Pitch + Roll
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=pd.to_numeric(df.get("pitch"), errors="coerce"),
                mode="lines", name="Pitch",
                line=dict(color=COLORS["orange"], width=1.2),
                hovertemplate="<b>%{y:.1f}</b>\u00b0<extra></extra>",
            ), row=3, col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=pd.to_numeric(df.get("roll"), errors="coerce"),
                mode="lines", name="Roll",
                line=dict(color=COLORS["blue"], width=1.2),
                hovertemplate="<b>%{y:.1f}</b>\u00b0<extra></extra>",
            ), row=3, col=1,
        )

        # Row 4: Clearance
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=pd.to_numeric(df.get("min_clearance_m"), errors="coerce"),
                mode="lines", name="Clearance",
                line=dict(color=COLORS["cyan"], width=1.5),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
                connectgaps=False,
                hovertemplate="<b>%{y:.2f}</b> m<extra></extra>",
            ), row=4, col=1,
        )

        fig.update_layout(
            **_layout(
                height=CHART_TELEMETRY_HEIGHT, showlegend=True,
                margin=dict(l=45, r=10, t=25, b=30),
            ),
        )
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=9),
        ))
        for r in range(1, 5):
            fig.update_xaxes(
                _axis(title_text="Time (s)" if r == 4 else ""),
                row=r, col=1,
            )
            fig.update_yaxes(_axis(), row=r, col=1)

        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=4, col=1)

        for ann in fig.layout.annotations:
            ann.font = dict(size=10, color=COLORS["text_dim"])

        return fig

    # ── GPS path data for Leaflet map ─────────────────────────────────────

    def _gps_path_data(self) -> list[list[float]]:
        """Return [[lat, lon], ...] for the tractor path polyline."""
        df = self.features_df
        lat = pd.to_numeric(df.get("gps_lat"), errors="coerce")
        lon = pd.to_numeric(df.get("gps_lon"), errors="coerce")
        valid = lat.notna() & lon.notna()
        if not valid.any():
            return []
        return [
            [float(la), float(lo)]
            for la, lo in zip(lat[valid], lon[valid])
        ]

    # ── Event context for template ────────────────────────────────────────

    def _event_ctx(self, ev: Event) -> dict[str, Any]:
        elapsed = ev.timestamp_sec - self._t0
        primary_path = ev.camera_paths.get(ev.primary_camera, "")
        primary_b64 = _encode_image_b64(
            primary_path, self.image_max_width, self.image_quality
        )

        surround: dict[str, str] = {}
        if self.include_surround:
            for cam, path in ev.camera_paths.items():
                if cam != ev.primary_camera:
                    thumb = _encode_image_b64(
                        path, self.thumb_max_width, self.image_quality
                    )
                    if thumb:
                        surround[cam] = thumb

        signals = [
            {"label": "Roughness", "value": ev.roughness_norm, "color": COLORS["orange"]},
            {"label": "Proximity", "value": ev.proximity_norm, "color": COLORS["blue"]},
            {"label": "Yaw Rate", "value": ev.yaw_rate_norm, "color": COLORS["amber"]},
            {"label": "IMU Fault", "value": ev.imu_fault_norm, "color": COLORS["red"]},
            {"label": "Localization", "value": ev.localization_fault_norm, "color": COLORS["purple"]},
        ]

        clearance_str = _fmt(ev.min_clearance_m, 2, "Blind")
        if ev.min_clearance_m is not None and np.isfinite(ev.min_clearance_m):
            clearance_str += " m"

        yaw_str = "N/A"
        if ev.yaw_rate is not None and np.isfinite(ev.yaw_rate):
            yaw_str = f"{ev.yaw_rate:.1f}\u00b0/s"

        metrics = [
            {"label": "Roughness", "value": _fmt(ev.roughness, 3)},
            {"label": "Clearance", "value": clearance_str},
            {"label": "Yaw Rate", "value": yaw_str},
            {"label": "IMU Corr", "value": _fmt(ev.imu_correlation, 3)},
            {"label": "Pose Conf", "value": _fmt(ev.pose_confidence, 1)},
            {
                "label": "GPS",
                "value": (
                    f"{ev.gps_lat:.4f}, {ev.gps_lon:.4f}"
                    if ev.gps_lat is not None and ev.gps_lon is not None
                    else "N/A"
                ),
            },
        ]

        has_ai = (
            ev.gemini_caption != UNAVAILABLE_CAPTION
            and ev.gemini_source != "unavailable"
        )

        df = self.features_df
        et = ev.timestamp_sec - self._t0
        idx_nearest = (df["_elapsed"] - et).abs().idxmin()
        vel_val = pd.to_numeric(
            df.loc[[idx_nearest], "velocity_mps"], errors="coerce"
        ).iloc[0]
        if not np.isfinite(vel_val):
            vel_val = 0.0

        return {
            "rank": ev.event_rank,
            "frame_idx": ev.frame_idx,
            "timestamp_iso": ev.timestamp_iso_utc,
            "elapsed": f"{elapsed:.1f}",
            "severity": ev.severity_score,
            "severity_pct": f"{ev.severity_score * 100:.1f}",
            "severity_color": _severity_color(ev.severity_score),
            "severity_label": _severity_label(ev.severity_score),
            "event_type": ev.event_type,
            "event_type_upper": ev.event_type.upper().replace("_", " "),
            "event_type_color": EVENT_TYPE_COLORS.get(
                ev.event_type, COLORS["text_dim"]
            ),
            "signals": signals,
            "metrics": metrics,
            "gemini_caption": ev.gemini_caption,
            "gemini_model": ev.gemini_model,
            "gemini_source": ev.gemini_source,
            "gemini_latency_ms": ev.gemini_latency_ms,
            "has_ai": has_ai,
            "primary_image": primary_b64 or _PLACEHOLDER_IMAGE,
            "primary_camera": ev.primary_camera,
            "surround": surround if surround else None,
            "gps_lat": ev.gps_lat,
            "gps_lon": ev.gps_lon,
            "velocity_at_event": float(vel_val),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Template — "Mission Control Cockpit" Layout
# ═══════════════════════════════════════════════════════════════════════════════
# Architectural Principles:
#   1. Viewport Density — Single 1080p viewport, no scrolling
#   2. Semantic Grounding — Leaflet dark map with path + event markers
#   3. Cross-Dimensional Interactivity — Global state links all components
#   4. MLOps Actionability — Triage buttons, clipboard payloads, toast

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/glightbox/dist/css/glightbox.min.css" />
<style>
/* ══════ CSS Custom Properties ══════ */
:root{
  --bg:#060a13;--surface:#0d1321;--elevated:#162036;
  --glass:rgba(13,19,33,0.85);--border:rgba(0,229,255,0.12);
  --brd-lt:rgba(0,229,255,0.06);--grid:rgba(0,229,255,0.08);
  --tx:#e0e6ed;--tx-dim:#7b8ca3;--tx-mut:#4a5568;
  --green:#00e676;--orange:#ff9100;--blue:#448aff;
  --red:#ff1744;--amber:#ffd740;--purple:#d500f9;--cyan:#00e5ff;
  --sans:'Space Grotesk','DM Sans',system-ui,sans-serif;
  --body:'DM Sans','Space Grotesk',system-ui,sans-serif;
  --mono:'IBM Plex Mono','Fira Code','Consolas',monospace;
  --radius:8px;--tr:0.25s cubic-bezier(0.4,0,0.2,1);
}

/* ══════ Reset & Base ══════ */
*,*::before,*::after{box-sizing:border-box}
body{
  background:
    radial-gradient(circle at 1px 1px,rgba(0,229,255,0.04) 1px,transparent 0) 0 0/40px 40px,
    radial-gradient(ellipse at 15% 50%,rgba(0,229,255,0.02) 0%,transparent 55%),
    var(--bg);
  color:var(--tx);font-family:var(--body);
  margin:0;padding:0;height:100vh;overflow:hidden;
  -webkit-font-smoothing:antialiased;
}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--elevated);border-radius:3px}

/* ══════ Header ══════ */
.mc-hdr{
  height:60px;display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(0,229,255,0.03),transparent);
  backdrop-filter:blur(20px);position:sticky;top:0;z-index:1000;
}
.mc-hdr::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan),var(--blue),var(--cyan),transparent);
  background-size:200% 100%;animation:scan 4s ease-in-out infinite;
}
@keyframes scan{0%{background-position:200% 0}50%{background-position:0% 0}100%{background-position:-200% 0}}
.mc-title{font-family:var(--sans);font-size:1.2rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:var(--tx);margin:0}
.mc-sub{font-size:0.75rem;color:var(--tx-mut);letter-spacing:0.08em}
.mc-meta{text-align:right;font-size:0.8rem;color:var(--tx-dim);font-family:var(--mono)}
.mc-meta span{display:block}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}
.dot-green{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot-amber{background:var(--amber);box-shadow:0 0 8px var(--amber)}
.dot-red{background:var(--red);box-shadow:0 0 8px var(--red)}
.ev-status{font-family:var(--sans);font-size:0.75rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase}

/* ══════ Split Layout ══════ */
.split-container {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: calc(100vh - 150px);
  overflow: hidden;
}
.gutter {
  background-color: var(--border);
  background-repeat: no-repeat;
  background-position: 50%;
  cursor: col-resize;
  z-index: 100;
  height: 100%;
}
.gutter.gutter-horizontal {
  background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjJo5aiEQINQBANY3EQnEQ81cAAAAAElFTkSuQmCC');
}
.sticky-panel {
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.scroll-panel {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
}

/* ══════ KPI Strip ══════ */
.kpi-strip{
  display:flex;align-items:center;gap:16px;
  padding:12px 16px;border-bottom:1px solid var(--border);
  background:rgba(13,19,33,0.9);backdrop-filter:blur(10px);
  height:90px;position:sticky;top:60px;z-index:999;
}
.kpi-cards{display:flex;gap:12px;flex-shrink:0}
.kpi{
  background:var(--surface);border:1px solid var(--border);
  border-radius:6px;padding:8px 16px;min-width:100px;text-align:center;
}
.kpi-label{font-family:var(--sans);font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--tx-mut);margin-bottom:1px}
.kpi-val{font-family:var(--mono);font-size:1.2rem;font-weight:600;color:var(--tx);line-height:1.2}

/* ══════ Panel Title ══════ */
.panel-title{
  font-family:var(--sans);font-size:0.7rem;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--cyan);padding:8px 12px;
  border-bottom:1px solid var(--brd-lt);flex-shrink:0;
  display:flex;align-items:center;gap:6px;
}
.panel-title::before{content:'';display:inline-block;width:3px;height:10px;background:var(--cyan);border-radius:2px}

/* ══════ Map Panel ══════ */
.map-panel{
  border-right:1px solid var(--border);
}
.map-panel #map{background:var(--bg);width:100%;height:100%;}

/* ══════ Charts Panel ══════ */
.charts-panel{
  border-right:1px solid var(--border);
}
.charts-panel #chart-telemetry{width:100%;height:100%;}

/* ══════ Events Feed ══════ */
.events-feed{
  display:flex;flex-direction:column;
}
.feed-scroll{padding:12px;display:flex;flex-direction:column;gap:12px;width:100%;}

/* ── Responsive event card adaptations ── */
.events-feed.feed-narrow .evm-body{flex-direction:column}
.events-feed.feed-narrow .evm-img{width:100%;max-width:100%}
.events-feed.feed-narrow .met-mini{grid-template-columns:1fr}
.events-feed.feed-narrow .evm-hdr{flex-wrap:wrap}
.events-feed.feed-narrow .filter-pills{flex-wrap:wrap}
.events-feed.feed-medium .evm-img{width:120px}
.events-feed.feed-medium .met-mini{grid-template-columns:1fr 1fr}
.events-feed.feed-wide .evm-img{width:180px}
.events-feed.feed-wide .met-mini{grid-template-columns:1fr 1fr 1fr}

/* ══════ Filter Pills ══════ */
.filter-pills { display:flex; gap:4px; }
.pill {
  background: var(--elevated);
  border: 1px solid var(--border);
  color: var(--tx-dim);
  padding: 4px 10px;
  border-radius: 12px;
  font-family: var(--sans);
  font-size: 0.7rem;
  cursor: pointer;
  transition: var(--tr);
}
.pill:hover { color: var(--tx); border-color: var(--cyan); }
.pill.active { background: rgba(0,229,255,0.1); color: var(--cyan); border-color: var(--cyan); }

/* ══════ Mini Event Cards ══════ */
.ev-card-mini{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;transition:var(--tr);cursor:pointer;
}
.ev-card-mini:hover{border-color:rgba(0,229,255,0.25)}
.ev-card-mini.ev-active{
  border-color:var(--cyan);
  box-shadow:0 0 16px rgba(0,229,255,0.15),inset 0 0 0 1px rgba(0,229,255,0.1);
}
.evm-hdr{display:flex;align-items:center;gap:8px;padding:8px 10px}
.evm-rank{font-family:var(--mono);font-size:0.9rem;font-weight:600;color:var(--cyan)}
.evm-badge{
  display:inline-flex;padding:1px 5px;border-radius:3px;
  font-family:var(--sans);font-size:0.55rem;font-weight:600;
  letter-spacing:0.06em;text-transform:uppercase;color:#fff;
}
.evm-sev{font-family:var(--mono);font-size:0.9rem;font-weight:600;margin-left:auto}
.sev-track{height:2px;background:var(--elevated)}
.sev-fill{height:100%;transition:width 0.6s ease}
.evm-body{display:flex;gap:12px;padding:8px 10px;flex-wrap:wrap}
.evm-img{width:160px;max-width:100%;height:auto;border-radius:4px;border:1px solid var(--border);flex-shrink:0;cursor:zoom-in;}
.evm-info{flex:1;min-width:0}
.evm-time{font-family:var(--mono);font-size:0.7rem;color:var(--tx-dim);margin-bottom:3px}

/* signal bars mini */
.sig-row-mini{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.sig-lbl-m{font-family:var(--body);font-size:0.65rem;color:var(--tx-mut);width:70px;flex-shrink:0}
.sig-bg-m{flex:1;height:4px;background:var(--elevated);border-radius:2px;overflow:hidden}
.sig-bar-m{height:100%;border-radius:2px;transition:width 0.5s ease}

/* metrics mini */
.met-mini{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.met-m{padding:6px 8px;background:var(--elevated);border-radius:4px;overflow:hidden}
.met-m-lbl{display:block;font-family:var(--sans);font-size:0.55rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--tx-mut)}
.met-m-val{font-family:var(--mono);font-size:0.75rem;color:var(--tx);word-break:break-all}

/* AI mini */
.evm-ai{
  padding:8px 10px;font-family:var(--mono);font-size:0.75rem;
  color:var(--tx-dim);font-style:italic;
  border-top:1px solid var(--brd-lt);
  background:linear-gradient(135deg,rgba(0,230,118,0.02),rgba(0,229,255,0.02));
}
.evm-ai::before{content:'AI ';font-style:normal;font-weight:600;color:var(--green);font-size:0.6rem;letter-spacing:0.04em}

/* ══════ Triage Buttons ══════ */
.evm-triage{display:flex;gap:6px;padding:8px 10px;border-top:1px solid var(--brd-lt)}
.tri-btn{
  flex:1;border:1px solid var(--border);border-radius:4px;
  background:var(--elevated);color:var(--tx-dim);
  font-family:var(--mono);font-size:0.65rem;
  padding:6px 8px;cursor:pointer;transition:var(--tr);text-align:center;
}
.tri-btn:hover{color:var(--tx)}
.tri-ok:hover{border-color:var(--green);color:var(--green)}
.tri-fp:hover{border-color:var(--red);color:var(--red)}
.tri-rt:hover{border-color:var(--amber);color:var(--amber)}
.tri-btn.tri-selected{opacity:0.5;pointer-events:none}
.tri-ok.tri-selected{border-color:var(--green);color:var(--green);opacity:1;background:rgba(0,230,118,0.1)}
.tri-fp.tri-selected{border-color:var(--red);color:var(--red);opacity:1;background:rgba(255,23,68,0.1)}
.tri-rt.tri-selected{border-color:var(--amber);color:var(--amber);opacity:1;background:rgba(255,215,64,0.1)}

/* ══════ Toast ══════ */
.toast-popup{
  position:fixed;bottom:16px;right:16px;
  background:var(--elevated);border:1px solid var(--border);
  border-left:3px solid var(--cyan);border-radius:6px;
  padding:10px 16px;font-family:var(--mono);font-size:0.85rem;color:var(--tx);
  transform:translateY(120%);opacity:0;transition:all 0.3s ease;
  z-index:9999;max-width:360px;box-shadow:0 8px 32px rgba(0,0,0,0.5);
}
.toast-show{transform:translateY(0);opacity:1}

/* ══════ Leaflet Overrides ══════ */
.leaflet-container{background:var(--bg) !important;font-family:var(--mono) !important}
.leaflet-control-zoom{display:none !important}
.marker-tip{
  background:var(--elevated) !important;color:var(--tx) !important;
  border:1px solid var(--border) !important;font-family:var(--mono) !important;
  font-size:0.8rem !important;
}
.marker-tip::before{border-top-color:var(--border) !important}

/* marker glow animation */
@keyframes markerPulse{
  0%,100%{filter:drop-shadow(0 0 4px currentColor)}
  50%{filter:drop-shadow(0 0 16px currentColor) drop-shadow(0 0 24px currentColor)}
}
.marker-glow{animation:markerPulse 1s ease-in-out infinite}

/* no-gps fallback */
.no-gps{display:flex;align-items:center;justify-content:center;height:100%;color:var(--tx-mut);font-family:var(--mono);font-size:1rem}
</style>
</head>
<body>

<!-- ══════ HEADER ══════ -->
<header class="mc-hdr">
  <div style="display:flex;align-items:center;gap:10px">
    <svg viewBox="0 0 32 32" width="32" height="32" fill="none" stroke="#00e5ff" stroke-width="1.5" stroke-linecap="round" style="filter:drop-shadow(0 0 6px rgba(0,229,255,0.3))">
      <circle cx="16" cy="16" r="13" opacity="0.2"/><circle cx="16" cy="16" r="7"/>
      <circle cx="16" cy="16" r="2" fill="#00e5ff" stroke="none"/>
      <line x1="16" y1="2" x2="16" y2="7"/><line x1="16" y1="25" x2="16" y2="30"/>
      <line x1="2" y1="16" x2="7" y2="16"/><line x1="25" y1="16" x2="30" y2="16"/>
    </svg>
    <div>
      <h1 class="mc-title">AGRI-AUDITOR</h1>
      <span class="mc-sub">Mission Control Dashboard</span>
    </div>
    <div style="margin-left:12px;display:flex;align-items:center">
      {% if summary.max_severity_label == "CRITICAL" %}
      <span class="dot dot-red"></span><span class="ev-status" style="color:var(--red)">{{ summary.max_severity_label }}</span>
      {% elif summary.max_severity_label == "WARNING" %}
      <span class="dot dot-amber"></span><span class="ev-status" style="color:var(--amber)">{{ summary.max_severity_label }}</span>
      {% else %}
      <span class="dot dot-green"></span><span class="ev-status" style="color:var(--green)">{{ summary.max_severity_label }}</span>
      {% endif %}
    </div>
  </div>
  <div class="mc-meta"><span>{{ run_id }}</span><span>{{ generated_at }}</span></div>
</header>

<!-- ── KPI Strip ── -->
<section class="kpi-strip">
  <div class="kpi-cards">
    <div class="kpi"><div class="kpi-label">Frames</div><div class="kpi-val">{{ summary.frames }}</div></div>
    <div class="kpi"><div class="kpi-label">Duration</div><div class="kpi-val">{{ summary.duration }}</div></div>
    <div class="kpi"><div class="kpi-label">Events</div><div class="kpi-val" style="color:{{ summary.max_severity_color }}">{{ summary.events }}</div></div>
    <div class="kpi"><div class="kpi-label">Avg Speed</div><div class="kpi-val">{{ summary.avg_speed }}</div></div>
    <div class="kpi"><div class="kpi-label">Max Severity</div><div class="kpi-val" style="color:{{ summary.max_severity_color }}">{{ summary.max_severity }}</div></div>
    <div class="kpi"><div class="kpi-label">Depth</div><div class="kpi-val">{{ summary.depth_pct }}</div></div>
  </div>
  <div style="flex:1; min-width:0; display:flex; flex-direction:column; justify-content:center; padding:0 16px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
      <span class="kpi-label" style="color:var(--cyan);">Global Time Scrubber</span>
      <span id="playhead-time" style="font-family:var(--mono); font-size:0.8rem; color:var(--cyan);">0.0s</span>
    </div>
    <input type="range" id="playhead" min="0" max="100" value="0" step="0.1" style="width:100%; cursor:pointer; accent-color:var(--cyan);">
  </div>
</section>

<!-- ══════ COCKPIT ══════ -->
<div class="split-container" id="split-container">

  <!-- ── Left: Map ── -->
  <section id="split-map" class="map-panel sticky-panel">
    <div class="panel-title">Mission Path</div>
    <div style="flex:1; min-height:0; min-width:0; position:relative;">
      <div id="map" style="position:absolute; top:0; left:0; right:0; bottom:0;">
        <div id="map-overlay" style="position:absolute; bottom:10px; left:10px; z-index:1000; border:1px solid var(--border); border-radius:4px; background:var(--elevated); padding:4px; box-shadow:0 4px 12px rgba(0,0,0,0.5);">
          <img id="current-frame-img" src="{{ events[0].primary_image if events else '' }}" style="width:160px; height:auto; display:block; border-radius:2px;">
          <div style="font-family:var(--mono); font-size:0.7rem; color:var(--tx-dim); text-align:center; margin-top:4px;">Nearest Event Frame</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ── Middle: Charts ── -->
  <section id="split-charts" class="charts-panel sticky-panel">
    <div class="panel-title">Telemetry</div>
    <div style="flex:1; min-height:0; min-width:0; position:relative;">
      <div id="chart-telemetry" style="position:absolute; top:0; left:0; right:0; bottom:0;"></div>
    </div>
  </section>

  <!-- ── Right: Events Feed ── -->
  <section id="split-feed" class="events-feed scroll-panel">
    <div class="panel-title" style="justify-content:space-between;">
      <span>Events ({{ summary.events }})</span>
      <div class="filter-pills">
        <button class="pill active" data-filter="all">Show All</button>
        <button class="pill" data-filter="proximity">Proximity</button>
        <button class="pill" data-filter="sensor_fault">Sensor Faults</button>
        <button class="pill" data-filter="critical">Critical Only</button>
      </div>
    </div>
    <div class="feed-scroll">
      {% for ev in events %}
      <div class="ev-card-mini" id="event-{{ ev.rank }}" data-idx="{{ loop.index0 }}">
        <div class="evm-hdr">
          <span class="evm-rank">#{{ ev.rank }}</span>
          <span class="evm-badge" style="background:{{ ev.event_type_color }}">{{ ev.event_type_upper }}</span>
          <span class="ev-status" style="color:{{ ev.severity_color }}">
            <span class="dot {% if ev.severity_label == 'CRITICAL' %}dot-red{% elif ev.severity_label == 'WARNING' %}dot-amber{% else %}dot-green{% endif %}"></span>
            {{ ev.severity_label }}
          </span>
          <span class="evm-sev" style="color:{{ ev.severity_color }}">{{ ev.severity_pct }}%</span>
        </div>
        <div class="sev-track"><div class="sev-fill" style="width:{{ ev.severity_pct }}%;background:{{ ev.severity_color }}"></div></div>
        <div class="evm-body">
          <a href="#surround-{{ ev.rank }}" class="glightbox" data-gallery="gallery-{{ ev.rank }}" data-glightbox="width: 90vw; height: 90vh;">
            <img src="{{ ev.primary_image }}" class="evm-img" alt="Event {{ ev.rank }}">
          </a>
          <div id="surround-{{ ev.rank }}" style="display:none;">
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:15px; padding:20px; background:var(--bg); width:90vw; height:90vh; overflow:auto;">
              <div style="text-align:center; color:var(--tx); font-family:var(--mono); font-size:1.2rem; grid-column:1/-1; margin-bottom:10px;">
                Surround View Matrix - Event #{{ ev.rank }} (@ {{ ev.elapsed }}s)
              </div>
              {% if ev.surround %}
                {% for cam, img in ev.surround.items() %}
                <div style="display:flex; flex-direction:column; align-items:center;">
                  <span style="font-family:var(--mono); font-size:0.85rem; color:var(--tx-dim); margin-bottom:4px;">{{ cam }}</span>
                  <img src="{{ img }}" style="width:100%; border:1px solid var(--border); border-radius:4px;">
                </div>
                {% endfor %}
              {% endif %}
              <div style="display:flex; flex-direction:column; align-items:center;">
                <span style="font-family:var(--mono); font-size:0.85rem; color:var(--cyan); margin-bottom:4px;">{{ ev.primary_camera }} (Primary)</span>
                <img src="{{ ev.primary_image }}" style="width:100%; border:1px solid var(--cyan); border-radius:4px;">
              </div>
            </div>
          </div>
          <div class="evm-info">
            <div class="evm-time">@ {{ ev.elapsed }}s &middot; Frame {{ ev.frame_idx }}</div>
            {% for s in ev.signals %}
            <div class="sig-row-mini">
              <span class="sig-lbl-m">{{ s.label }}</span>
              <div class="sig-bg-m"><div class="sig-bar-m" style="width:{{ (s.value * 100)|round(1) }}%;background:{{ s.color }}"></div></div>
            </div>
            {% endfor %}
            <div class="met-mini">
              {% for m in ev.metrics %}
              <div class="met-m"><span class="met-m-lbl">{{ m.label }}</span><span class="met-m-val">{{ m.value }}</span></div>
              {% endfor %}
            </div>
          </div>
        </div>
        {% if ev.has_ai %}
        <div class="evm-ai">"{{ ev.gemini_caption }}"</div>
        {% endif %}
        <div class="evm-triage">
          <button class="tri-btn tri-ok" onclick="triageEvent({{ loop.index0 }},'accurate',this)">&#10003; Accurate</button>
          <button class="tri-btn tri-fp" onclick="triageEvent({{ loop.index0 }},'false_positive',this)">&#10007; False Positive</button>
          <button class="tri-btn tri-rt" onclick="triageEvent({{ loop.index0 }},'retrain',this)">&#8635; Retrain</button>
          <button class="tri-btn" onclick="downloadCSV({{ loop.index0 }})">&darr; CSV</button>
        </div>
      </div>
      {% endfor %}
    </div>
  </section>
</div>

<!-- Toast -->
<div id="toast" class="toast-popup"></div>

<!-- ══════ SCRIPTS ══════ -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/split.js/1.6.5/split.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/mcstudios/glightbox/dist/js/glightbox.min.js"></script>
<script>
(function(){
  /* ── Data ── */
  var RUN_ID = '{{ run_id }}';
  var EVENTS = {{ events_json_str | safe }};
  var GPS_PATH = {{ gps_path_json | safe }};
  var FEATURES = {{ features_json | safe }};
  var cfg = {responsive:true, displayModeBar:false, scrollZoom:true};

  /* ── Split.js Layout ── */
  var sizes = sessionStorage.getItem('split-sizes');
  if (sizes) sizes = JSON.parse(sizes);
  else sizes = [30, 40, 30];
  Split(['#split-map', '#split-charts', '#split-feed'], {
    sizes: sizes,
    minSize: 200,
    gutterSize: 8,
    onDrag: function() {
      window.dispatchEvent(new Event('resize'));
    },
    onDragEnd: function(sizes) {
      sessionStorage.setItem('split-sizes', JSON.stringify(sizes));
      window.dispatchEvent(new Event('resize'));
    }
  });

  /* ── Lightbox ── */
  var lightbox = GLightbox({
    selector: '.glightbox',
    touchNavigation: true,
    loop: true,
    zoomable: true
  });

  /* ── Plotly: Main Chart ── */
  var telData = {{ telemetry_json | safe }};
  var telEl = document.getElementById('chart-telemetry');
  Plotly.newPlot(telEl, telData.data, telData.layout, cfg);

  /* ── Leaflet Map ── */
  var mapEl = document.getElementById('map');
  var lmap = null;
  var pathLine = null;
  var markers = {};

  /* ── Auto-resize chart to fill container ── */
  var resizeTimeout;
  function resizeCharts(){
    if(resizeTimeout) cancelAnimationFrame(resizeTimeout);
    resizeTimeout = requestAnimationFrame(function(){
      Plotly.Plots.resize(telEl);
      if(typeof lmap!=='undefined' && lmap && lmap.invalidateSize) {
        lmap.invalidateSize();
        if(pathLine) lmap.fitBounds(pathLine.getBounds().pad(0.15));
      }
    });
  }
  setTimeout(resizeCharts, 150);
  window.addEventListener('resize', resizeCharts);

  /* ── Dynamic Zoom Adjustment ── */
  var lastChartScale = 1;
  var lastFeedClass = '';
  var ro = new ResizeObserver(function(entries) {
    for (var i = 0; i < entries.length; i++) {
      var entry = entries[i];
      var w = entry.contentRect.width;
      var el = entry.target;
      
      if (el.id === 'split-charts') {
        var scale = Math.max(0.5, Math.min(1.5, w / 700));
        if (Math.abs(scale - lastChartScale) > 0.05) {
          lastChartScale = scale;
          Plotly.relayout(telEl, {
            'font.size': 14 * scale,
            'legend.font.size': 12 * scale,
            'margin.l': Math.round(55 * scale),
            'margin.r': Math.round(20 * scale),
            'margin.t': Math.round(35 * scale),
            'margin.b': Math.round(35 * scale)
          });
        }
      } else if (el.id === 'split-feed') {
        var cls = w < 350 ? 'feed-narrow' : w < 550 ? 'feed-medium' : 'feed-wide';
        if (cls !== lastFeedClass) {
          lastFeedClass = cls;
          el.classList.remove('feed-narrow','feed-medium','feed-wide');
          el.classList.add(cls);
        }
      }
    }
  });
  ro.observe(document.getElementById('split-charts'));
  ro.observe(document.getElementById('split-feed'));

  if(GPS_PATH.length > 0){
    lmap = L.map(mapEl, {zoomControl:false, attributionControl:false});
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
      maxZoom:20, subdomains:'abcd'
    }).addTo(lmap);

    /* path polyline */
    pathLine = L.polyline(GPS_PATH,{color:'#7b8ca3',weight:2,opacity:0.5}).addTo(lmap);
    lmap.fitBounds(pathLine.getBounds().pad(0.15));

    /* event markers */
    EVENTS.forEach(function(ev, idx){
      if(ev.gps_lat != null && ev.gps_lon != null){
        var m = L.circleMarker([ev.gps_lat, ev.gps_lon],{
          radius:8, color:ev.severity_color, fillColor:ev.severity_color,
          fillOpacity:0.8, weight:2
        }).addTo(lmap);
        m.bindTooltip('#'+ev.rank+' '+ev.event_type_upper,{
          permanent:false, direction:'top', className:'marker-tip'
        });
        m.on('mouseover',function(){DS.hover(idx)});
        m.on('mouseout',function(){DS.clearHover()});
        m.on('click',function(){DS.select(idx)});
        markers[idx] = m;
      }
    });
  } else {
    mapEl.innerHTML = '<div class="no-gps">No GPS Data Available</div>';
  }

  /* ── Dashboard State (Cross-Dimensional Interactivity) ── */
  var DS = {
    hoverIdx: null,
    selectedEventId: null,

    hover: function(idx) {
      if (this.selectedEventId !== null) return;
      this._applyVisuals(idx);
    },
    clearHover: function() {
      if (this.selectedEventId !== null) return;
      this._clearVisuals();
    },
    select: function(idx) {
      if (this.selectedEventId === idx) {
        this.selectedEventId = null;
        this._clearVisuals();
        Plotly.relayout(telEl, {'xaxis.autorange': true});
      } else {
        this.selectedEventId = idx;
        this._applyVisuals(idx);
        var ev = EVENTS[idx];
        var t = parseFloat(ev.elapsed);
        Plotly.relayout(telEl, {'xaxis.range': [t - 2.5, t + 2.5]});
      }
    },
    clearSelection: function() {
      this.selectedEventId = null;
      this._clearVisuals();
      Plotly.relayout(telEl, {'xaxis.autorange': true});
    },
    _applyVisuals: function(idx) {
      this._clearVisuals();
      this.hoverIdx = idx;
      var ev = EVENTS[idx];
      if(!ev) return;
      var elapsed = parseFloat(ev.elapsed);

      Plotly.relayout(telEl, {
        shapes:[{
          type:'line', x0:elapsed, x1:elapsed,
          y0:0, y1:1, yref:'paper',
          line:{color:'#ff1744', width:2, dash:'dot'}, opacity:0.8
        }]
      });

      if(markers[idx]){
        markers[idx].setRadius(14);
        markers[idx].setStyle({fillOpacity:1, weight:3});
        var p = markers[idx]._path || markers[idx].getElement();
        if(p) p.classList.add('marker-glow');
      }

      var card = document.getElementById('event-'+ev.rank);
      if(card){
        card.classList.add('ev-active');
        if (this.selectedEventId !== null) {
          card.scrollIntoView({behavior:'smooth',block:'nearest'});
        }
      }
    },
    _clearVisuals: function() {
      Plotly.relayout(telEl, {shapes:[]});
      if(this.hoverIdx !== null && markers[this.hoverIdx]){
        markers[this.hoverIdx].setRadius(8);
        markers[this.hoverIdx].setStyle({fillOpacity:0.8, weight:2});
        var p = markers[this.hoverIdx]._path || markers[this.hoverIdx].getElement();
        if(p) p.classList.remove('marker-glow');
      }
      document.querySelectorAll('.ev-card-mini').forEach(function(c){c.classList.remove('ev-active')});
      this.hoverIdx = null;
    }
  };

  /* ── Event Card Hover → Chart + Map sync ── */
  document.querySelectorAll('.ev-card-mini').forEach(function(card, idx){
    card.addEventListener('mouseenter',function(){DS.hover(idx)});
    card.addEventListener('mouseleave',function(){DS.clearHover()});
    card.addEventListener('click',function(e){
      if(e.target.closest('.evm-triage') || e.target.closest('.glightbox')) return;
      DS.select(idx);
    });
  });

  /* ── Chart Click → Find Nearest Event ── */
  telEl.on('plotly_click',function(d){
    if(!d||!d.points||!d.points[0]) return;
    var clickX = d.points[0].x;
    var nearest = -1, minDist = Infinity;
    EVENTS.forEach(function(ev,idx){
      var dist = Math.abs(parseFloat(ev.elapsed)-clickX);
      if(dist<minDist && dist<3.0){minDist=dist;nearest=idx}
    });
    if(nearest>=0) DS.select(nearest);
  });

  /* ── Global Time Scrubbing ── */
  var playheadInput = document.getElementById('playhead');
  var playheadTime = document.getElementById('playhead-time');
  if (FEATURES.length > 0) {
    playheadInput.max = FEATURES[FEATURES.length - 1]._elapsed;
  }
  var currentFrameImg = document.getElementById('current-frame-img');
  var currentFrameMarker = null;

  playheadInput.addEventListener('input', function(e) {
    var t = parseFloat(e.target.value);
    playheadTime.textContent = t.toFixed(1) + 's';

    var shapes = [];
    if (DS.selectedEventId !== null) {
      var ev = EVENTS[DS.selectedEventId];
      shapes.push({
        type:'line', x0:parseFloat(ev.elapsed), x1:parseFloat(ev.elapsed),
        y0:0, y1:1, yref:'paper',
        line:{color:'#ff1744', width:2, dash:'dot'}, opacity:0.8
      });
    }
    shapes.push({
      type:'line', x0:t, x1:t,
      y0:0, y1:1, yref:'paper',
      line:{color:'#00e5ff', width:2}, opacity:0.8
    });
    Plotly.relayout(telEl, {shapes: shapes});

    var nearestRow = FEATURES.reduce(function(prev, curr) {
      return (Math.abs(curr._elapsed - t) < Math.abs(prev._elapsed - t) ? curr : prev);
    });

    if (nearestRow.gps_lat != null && nearestRow.gps_lon != null && lmap) {
      if (!currentFrameMarker) {
        currentFrameMarker = L.circleMarker([nearestRow.gps_lat, nearestRow.gps_lon], {
          radius: 6, color: '#00e5ff', fillColor: '#00e5ff', fillOpacity: 1, weight: 2
        }).addTo(lmap);
      } else {
        currentFrameMarker.setLatLng([nearestRow.gps_lat, nearestRow.gps_lon]);
      }
    }

    if (EVENTS.length > 0) {
      var nearestEv = EVENTS.reduce(function(prev, curr) {
        return (Math.abs(parseFloat(curr.elapsed) - t) < Math.abs(parseFloat(prev.elapsed) - t) ? curr : prev);
      });
      if (currentFrameImg) {
        currentFrameImg.src = nearestEv.primary_image;
      }
    }
  });

  /* ── Keyboard Navigation ── */
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      DS.clearSelection();
    } else if (e.key === 'ArrowRight') {
      if (DS.selectedEventId === null) DS.select(0);
      else if (DS.selectedEventId < EVENTS.length - 1) DS.select(DS.selectedEventId + 1);
    } else if (e.key === 'ArrowLeft') {
      if (DS.selectedEventId === null) DS.select(EVENTS.length - 1);
      else if (DS.selectedEventId > 0) DS.select(DS.selectedEventId - 1);
    } else if (e.key === ' ') {
      e.preventDefault();
      if (DS.selectedEventId !== null) {
        var layout = telEl.layout;
        if (layout.xaxis && layout.xaxis.autorange) {
          var t = parseFloat(EVENTS[DS.selectedEventId].elapsed);
          Plotly.relayout(telEl, {'xaxis.range': [t - 2.5, t + 2.5], 'xaxis.autorange': false});
        } else {
          Plotly.relayout(telEl, {'xaxis.autorange': true});
        }
      }
    }
  });

  /* ── Event Filtering ── */
  var filterPills = document.querySelectorAll('.pill');
  filterPills.forEach(function(pill) {
    pill.addEventListener('click', function() {
      filterPills.forEach(function(p) { p.classList.remove('active'); });
      pill.classList.add('active');
      var filter = pill.getAttribute('data-filter');

      var newX = [], newSeverityY = [], newVelocityY = [], newColors = [], newTexts = [];
      EVENTS.forEach(function(ev, idx) {
        var show = false;
        if (filter === 'all') show = true;
        else if (filter === 'proximity' && ev.event_type === 'proximity') show = true;
        else if (filter === 'sensor_fault' && ev.event_type === 'sensor_fault') show = true;
        else if (filter === 'critical' && ev.severity_label === 'CRITICAL') show = true;

        var card = document.getElementById('event-' + ev.rank);
        if (card) card.style.display = show ? 'block' : 'none';

        if (markers[idx] && lmap) {
          if (show) markers[idx].addTo(lmap);
          else markers[idx].remove();
        }

        if (show) {
          newX.push(parseFloat(ev.elapsed));
          newSeverityY.push(ev.severity);
          newVelocityY.push(ev.velocity_at_event);
          newColors.push(ev.severity_color);
          newTexts.push('<b>Event #' + ev.rank + '</b><br>Severity: ' + ev.severity.toFixed(2));
        }
      });

      // Update Plotly traces (trace 1 is Severity markers, trace 3 is Velocity markers)
      Plotly.restyle(telEl, {
        x: [newX, newX],
        y: [newSeverityY, newVelocityY],
        'marker.color': [newColors, newColors],
        text: [newTexts, newTexts]
      }, [1, 3]);
    });
  });

  /* ── Triage (MLOps Actionability) ── */
  window.triageEvent = function(idx, action, btn){
    var ev = EVENTS[idx];
    if(!ev) return;
    var payload = {
      run_id: RUN_ID,
      event_rank: ev.rank,
      frame: ev.frame_idx,
      timestamp_elapsed: ev.elapsed+'s',
      event_type: ev.event_type,
      severity: ev.severity,
      action: action,
      tagged_at: new Date().toISOString()
    };
    var txt = JSON.stringify(payload, null, 2);

    var triage = btn.parentElement;
    triage.querySelectorAll('.tri-btn').forEach(function(b){
      b.classList.remove('tri-selected');
    });
    btn.classList.add('tri-selected');

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(function(){
        showToast(action.replace(/_/g,' ')+' \u2014 payload copied to clipboard', ev.severity_color);
      }).catch(function(){
        showToast('Payload generated (check console)', '#ff9100');
        console.log('Triage payload:', txt);
      });
      return;
    }

    showToast('Payload generated (check console)', '#ff9100');
    console.log('Triage payload:', txt);
  };

  /* ── CSV Export ── */
  window.downloadCSV = function(idx) {
    var ev = EVENTS[idx];
    var t = parseFloat(ev.elapsed);
    var t_start = t - 2.5;
    var t_end = t + 2.5;

    var csvRows = ['timestamp_ms,channel_name,value'];
    FEATURES.forEach(function(row) {
      if (row._elapsed >= t_start && row._elapsed <= t_end) {
        var ts_ms = Math.round(row.timestamp_sec * 1000);
        ['velocity_mps', 'pitch', 'roll', 'min_clearance_m', 'roughness', 'yaw_rate', 'imu_correlation', 'pose_confidence'].forEach(function(ch) {
          if (row[ch] != null) {
            csvRows.push(ts_ms + ',' + ch + ',' + row[ch]);
          }
        });
      }
    });

    var blob = new Blob([csvRows.join('\n')], {type: 'text/csv'});
    var url = window.URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'event_' + ev.rank + '_telemetry.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  function showToast(msg, color){
    var t = document.getElementById('toast');
    t.textContent = msg;
    t.style.borderLeftColor = color||'#00e5ff';
    t.classList.add('toast-show');
    clearTimeout(t._timer);
    t._timer = setTimeout(function(){t.classList.remove('toast-show')},3000);
  }
})();
</script>
</body>
</html>
"""
