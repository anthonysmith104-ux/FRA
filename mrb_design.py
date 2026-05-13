"""
mrb_design.py — Shared design utilities for the MRB Capital Group apps.

This module is the bridge between firm_settings.json and the two Streamlit apps
(app.py advisor, client_portal.py client). Both apps should import from here
rather than reading colors, copy, or geometry directly.

Why a shared module:
- Single source of truth: edit firm_settings.json once, both apps update.
- Stable ticker colors: get_ticker_color() guarantees the same ticker gets the
  same color across runs, even for tickers not explicitly mapped.
- Centralized SVG generation: the donut math (stroke-dasharray offsets, segment
  angles) is annoying to get right; do it once here.
- Streamlit theme derivation: keeps .streamlit/config.toml in sync with the spec
  without manual editing.

Typical use in an app:
    from mrb_design import (
        load_settings, get_ticker_color, pick_alignment_tier,
        generate_donut_svg, format_proposal_copy,
    )

    settings = load_settings()
    color = get_ticker_color("SCHD", settings)
    status, detail = pick_alignment_tier(profile=62, current=93, settings=settings)
    svg = generate_donut_svg({"SCHD": 15, "VOO": 10, ...}, settings)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


# ─────────────────────────────────────────────────────────────────────────────
# Settings loader
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_TOP_LEVEL_KEYS = (
    "firm", "advisor", "brand", "typography", "layout",
    "proposal_copy", "chart_palette",
    "donut_geometry", "gauge_geometry",
    "crest_badge", "gradient_band", "horizon_bar",
)


class SettingsError(Exception):
    """Raised when firm_settings.json is missing, malformed, or incomplete."""


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate firm_settings.json.

    Looks for the file at (in order):
      1. Explicit `path` argument
      2. $MRB_SETTINGS_PATH environment variable
      3. ./firm_settings.json (current working dir)
      4. Sibling of this module file

    Raises SettingsError if the file is missing, invalid JSON, or missing any
    required top-level key. Returns the parsed dict on success.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if env := os.environ.get("MRB_SETTINGS_PATH"):
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "firm_settings.json")
    candidates.append(Path(__file__).parent / "firm_settings.json")

    for candidate in candidates:
        if candidate.is_file():
            try:
                with candidate.open(encoding="utf-8") as f:
                    settings = json.load(f)
            except json.JSONDecodeError as e:
                raise SettingsError(
                    f"firm_settings.json at {candidate} is not valid JSON: {e}"
                ) from e

            missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in settings]
            if missing:
                raise SettingsError(
                    f"firm_settings.json at {candidate} is missing required "
                    f"top-level keys: {missing}"
                )
            return settings

    tried = "\n  ".join(str(c) for c in candidates)
    raise SettingsError(
        f"Could not find firm_settings.json. Tried:\n  {tried}\n"
        "Set MRB_SETTINGS_PATH or place firm_settings.json in the working dir."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ticker color resolution
# ─────────────────────────────────────────────────────────────────────────────


def get_ticker_color(ticker: str, settings: Mapping[str, Any]) -> str:
    """Return a stable hex color for a ticker.

    Priority:
      1. Explicit mapping in chart_palette.tickers
      2. Deterministic fallback: md5(ticker) % len(_fallback_palette)

    Using md5 instead of Python's built-in hash() because hash() is randomized
    per Python process by default (PYTHONHASHSEED), which would break color
    stability across runs. md5 is overkill but deterministic and free.

    Args:
        ticker: e.g. "SCHD". Case-sensitive — pass the symbol as it appears
            in holdings data.
        settings: Output of load_settings().

    Returns:
        Hex color string, e.g. "#a8388a".
    """
    tickers = settings["chart_palette"]["tickers"]

    explicit = tickers.get(ticker)
    if explicit is not None and not ticker.startswith("_"):
        return explicit

    fallback = tickers["_fallback_palette"]
    digest = hashlib.md5(ticker.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(fallback)
    return fallback[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Risk alignment tier
# ─────────────────────────────────────────────────────────────────────────────


def pick_alignment_tier(
    profile: int,
    current: int,
    settings: Mapping[str, Any],
) -> tuple[str, str, str, int]:
    """Pick the alignment status, detail sentence, semantic color, and delta.

    Three-tier system using absolute delta between profile and current:
      ≤5   → aligned       → success (green)
      6-10 → slight_drift  → warning (amber)
      11+  → misaligned    → danger  (red)

    Args:
        profile: Client's risk profile score (1-99).
        current: Risk score of the client's current portfolio (1-99).
        settings: Output of load_settings().

    Returns:
        (status_label, detail_sentence, semantic_color_key, signed_delta) tuple.
        - status_label: "Aligned" / "Slight drift" / "Misaligned"
        - detail_sentence: One-line plain-language description
        - semantic_color_key: "success" / "warning" / "danger". Index into
            settings["brand"]["semantic"][key] for the hex.
        - signed_delta: current - profile (positive if current is more
            aggressive than profile). Render with a "+" sign for positive
            values: f"{delta:+d}".

    Example:
        status, detail, color_key, delta = pick_alignment_tier(62, 93, settings)
        # → ("Misaligned",
        #    "Current portfolio score 93 sits well above your 62 profile.",
        #    "danger",
        #    31)
        hex_color = settings["brand"]["semantic"][color_key]  # "#a32d2d"
    """
    copy = settings["proposal_copy"]["alignment_status_copy"]
    delta = current - profile
    abs_delta = abs(delta)

    if abs_delta <= 5:
        status = copy["aligned"]
        color_key = "success"
        position_key = "within_above" if delta >= 0 else "within_below"
    elif abs_delta <= 10:
        status = copy["slight_drift"]
        color_key = "warning"
        position_key = "slightly_above" if delta > 0 else "slightly_below"
    else:
        status = copy["misaligned"]
        color_key = "danger"
        position_key = "well_above" if delta > 0 else "well_below"

    position_phrase = copy["_position_phrases"][position_key]
    detail = copy["detail_template"].format(
        current=current,
        profile=profile,
        position_phrase=position_phrase,
    )

    return status, detail, color_key, delta


# ─────────────────────────────────────────────────────────────────────────────
# Donut SVG generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_donut_svg(
    holdings: Mapping[str, float],
    settings: Mapping[str, Any],
    *,
    canonical_order: list[str] | None = None,
    center_label: str | None = None,
    center_sublabel: str = "HOLDINGS",
) -> str:
    """Generate an SVG donut chart for a holdings dict.

    Segments are rendered in `canonical_order` if provided (recommended for
    side-by-side comparison across multiple cards), otherwise in dict
    iteration order. Holdings not in canonical_order are appended at the end.

    Args:
        holdings: {ticker: weight_pct} where weights sum to ~100.
        settings: Output of load_settings().
        canonical_order: Optional list of tickers defining segment draw order.
            Pass the Option 1 ticker list to keep segment-by-position parity
            across all three recommendation cards.
        center_label: Text for the center of the donut. Defaults to the
            holdings count as a number (e.g. "14").
        center_sublabel: Subtext under the center label. Defaults to "HOLDINGS".

    Returns:
        Raw SVG string. Drop this directly into Streamlit via st.markdown(svg,
        unsafe_allow_html=True), or use it as an XML asset in a ReportLab PDF.
    """
    geom = settings["donut_geometry"]
    r = geom["radius"]
    stroke = geom["stroke_width"]
    vb = geom["viewbox"]
    rotation_start = geom["rotation_start"]
    center_pt = geom["center_label_font_pt"]
    sub_pt = geom["center_label_sublabel_pt"]

    circumference = 2 * math.pi * r
    cx = (vb[0] + vb[2]) / 2
    cy = (vb[1] + vb[3]) / 2

    # Determine segment order
    if canonical_order:
        ordered = [t for t in canonical_order if t in holdings]
        # Append any tickers in holdings that weren't in the canonical list
        ordered += [t for t in holdings if t not in canonical_order]
    else:
        ordered = list(holdings)

    # Build segments
    segments_svg: list[str] = []
    offset = 0.0
    for ticker in ordered:
        weight = float(holdings[ticker])
        if weight <= 0:
            continue
        dash = weight * circumference / 100.0
        gap = circumference - dash
        color = get_ticker_color(ticker, settings)
        segments_svg.append(
            f'<circle r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" '
            f'stroke-dasharray="{dash:.4f} {gap:.4f}" '
            f'stroke-dashoffset="{-offset:.4f}" '
            f'transform="rotate({rotation_start})"/>'
        )
        offset += dash

    # Center label defaults to the holdings count
    if center_label is None:
        center_label = str(sum(1 for w in holdings.values() if w > 0))

    navy = settings["brand"]["primary"]["navy"]
    muted = settings["brand"]["text"]["muted"]
    serif = settings["typography"]["fonts"]["serif"]
    sans = settings["typography"]["fonts"]["sans"]

    # Scale font sizes from the geometry's "pt" reference into the viewBox.
    # Heuristic: viewBox is sized so that 1pt ≈ 1 unit in viewBox space.
    label_font_size = center_pt + 4
    sub_font_size = sub_pt + 2

    return (
        f'<svg viewBox="{vb[0]} {vb[1]} {vb[2]} {vb[3]}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="xMidYMid meet">\n'
        f'  <g transform="translate({cx}, {cy})">\n'
        + "\n".join("    " + s for s in segments_svg)
        + f'\n    <text x="0" y="-1" text-anchor="middle" '
        f'font-size="{label_font_size}" font-weight="500" fill="{navy}" '
        f'font-family="{serif}">{center_label}</text>\n'
        f'    <text x="0" y="11" text-anchor="middle" '
        f'font-size="{sub_font_size}" fill="{muted}" '
        f'letter-spacing="0.8" font-family="{sans}">{center_sublabel}</text>\n'
        f'  </g>\n'
        f'</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Color key resolver
# ─────────────────────────────────────────────────────────────────────────────


def resolve_color_key(key_path: str, settings: Mapping[str, Any]) -> str:
    """Resolve a dotted color key path to a hex string.

    Geometry blocks (gauge_geometry, gradient_band, horizon_bar, crest_badge)
    declare semantic intent via keys like "brand.primary.navy" rather than
    hardcoding hex. This resolver walks the path and returns the value.

    Args:
        key_path: Dotted path, e.g. "brand.primary.navy" or
            "brand.semantic.warning_bg".
        settings: Output of load_settings().

    Returns:
        Hex color string.

    Raises:
        KeyError if the path doesn't resolve to a value.

    Example:
        resolve_color_key("brand.primary.navy", settings)  # → "#1a2b4a"
    """
    node: Any = settings
    for segment in key_path.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            raise KeyError(
                f"Color key path '{key_path}' not found in settings"
            )
        node = node[segment]
    if not isinstance(node, str):
        raise TypeError(
            f"Color key path '{key_path}' did not resolve to a string"
        )
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Speedometer gauge SVG
# ─────────────────────────────────────────────────────────────────────────────


def generate_gauge_svg(
    value: int,
    settings: Mapping[str, Any],
    *,
    max_value: int = 99,
    show_endpoint_labels: bool = True,
) -> str:
    """Generate a half-circle speedometer gauge SVG.

    Uses the same stroke-dasharray technique as the donut chart: draws the
    full arc twice (background + fill), with the fill arc's dash pattern
    controlling how much of it is visible. This guarantees a single
    continuous arc regardless of fill percentage — the arc-endpoint-math
    approach leaves disconnected fragments at certain values.

    Args:
        value: The risk number (or whatever 0-max scalar). Clamped to
            [0, max_value].
        settings: Output of load_settings().
        max_value: Top of the scale. Defaults to 99 for risk numbers.
        show_endpoint_labels: If True, draw "1" and "99" labels at the
            arc endpoints. Set False for smaller renderings.

    Returns:
        Raw SVG string, ready to drop into Streamlit via
        st.markdown(svg, unsafe_allow_html=True) or write to a PDF asset.
    """
    geom = settings["gauge_geometry"]
    vb = geom["viewbox"]
    p_start = geom["start_point"]
    p_end = geom["end_point"]
    r = geom["radius"]
    arc_length = geom["arc_length"]
    stroke_w = geom["stroke_width"]
    linecap = geom["stroke_linecap"]

    fill_color = resolve_color_key(geom["fill_color_key"], settings)
    track_color = resolve_color_key(geom["track_color_key"], settings)
    muted = settings["brand"]["text"]["muted"]
    sans = settings["typography"]["fonts"]["sans"]

    # Clamp and compute fill length
    v = max(0, min(value, max_value))
    fill_length = arc_length * v / max_value
    # Single dash followed by a gap >= full arc length, so no repeat
    dash_pattern = f"{fill_length:.3f} {arc_length:.3f}"

    arc_path = (
        f"M {p_start[0]} {p_start[1]} "
        f"A {r} {r} 0 0 1 {p_end[0]} {p_end[1]}"
    )

    endpoint_labels_svg = ""
    if show_endpoint_labels:
        ep_y = geom["endpoint_label_y"]
        ep_pt = geom["endpoint_label_font_pt"]
        endpoint_labels_svg = (
            f'  <text x="{p_start[0]}" y="{ep_y}" font-size="{ep_pt}" '
            f'fill="{muted}" text-anchor="middle" font-family="{sans}">1</text>\n'
            f'  <text x="{p_end[0]}" y="{ep_y}" font-size="{ep_pt}" '
            f'fill="{muted}" text-anchor="middle" font-family="{sans}">'
            f'{max_value}</text>\n'
        )

    return (
        f'<svg viewBox="{vb[0]} {vb[1]} {vb[2]} {vb[3]}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="xMidYMid meet">\n'
        f'  <path d="{arc_path}" fill="none" stroke="{track_color}" '
        f'stroke-width="{stroke_w}" stroke-linecap="{linecap}"/>\n'
        f'  <path d="{arc_path}" fill="none" stroke="{fill_color}" '
        f'stroke-width="{stroke_w}" stroke-linecap="{linecap}" '
        f'stroke-dasharray="{dash_pattern}"/>\n'
        f'{endpoint_labels_svg}'
        f'</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Horizon row (page 1)
# ─────────────────────────────────────────────────────────────────────────────


def pick_horizon_shape(
    current_age: int,
    target_retirement_age: int | None,
    is_retired: bool,
) -> str:
    """Pick which of the three horizon shapes applies to a client.

    Three shapes:
      - "preretirement": Still working, current age < target retirement age.
      - "working_past_age": Still working, current age >= target retirement age.
      - "retired": is_retired flag is set.

    Args:
        current_age: Client's current age.
        target_retirement_age: Client's target retirement age, or None if
            unset. If None and not retired, defaults to preretirement.
        is_retired: Whether the client is already in retirement.

    Returns:
        One of "preretirement" / "working_past_age" / "retired".
    """
    if is_retired:
        return "retired"
    if target_retirement_age is None:
        return "preretirement"
    if current_age >= target_retirement_age:
        return "working_past_age"
    return "preretirement"


def compute_horizon_data(
    current_age: int,
    settings: Mapping[str, Any],
    *,
    target_retirement_age: int | None = None,
    is_retired: bool = False,
    retired_since_age: int | None = None,
    planning_horizon_years: int | None = None,
) -> dict[str, Any]:
    """Compute everything the page-1 horizon row needs to render.

    Returns a dict with the resolved status label/sublabel, the headline
    horizon string, the sub-label string, the bar endpoint labels, and the
    bar fill percentage. The advisor app calls this once and feeds the
    result into the template. The rendering code stays free of any math
    or conditional logic.

    Args:
        current_age: Client's current age.
        settings: Output of load_settings().
        target_retirement_age: Target retirement age. Required for
            preretirement and working_past_age shapes.
        is_retired: True if the client has already retired.
        retired_since_age: Age at which the client retired. Required for
            the retired shape.
        planning_horizon_years: For retired clients, years of planning
            horizon. Defaults to the value in horizon_copy._retired_default.

    Returns:
        Dict with keys: shape, status_label, status_sublabel,
        horizon_headline, horizon_sublabel, bar_left_label, bar_right_label,
        bar_fill_pct (None for working_past_age which has no bar), and
        signed years (years_to_retirement or years_retired).

    Example (Cole J: age 40, targeting 65):
        compute_horizon_data(current_age=40, settings=s,
                             target_retirement_age=65)
        # → {
        #     "shape": "preretirement",
        #     "status_label": "Pre-retirement",
        #     "status_sublabel": "Accumulation phase",
        #     "horizon_headline": "25 years",
        #     "horizon_sublabel": "Age 40 → 65",
        #     "bar_left_label": "Working life · 40",
        #     "bar_right_label": "Retirement · 65",
        #     "bar_fill_pct": 61.54,  # 40/65 × 100
        #     "years": 25,
        #   }
    """
    copy = settings["proposal_copy"]["horizon_copy"]
    shape = pick_horizon_shape(current_age, target_retirement_age, is_retired)

    status_label = copy["status_labels"][shape]
    status_sublabel = copy["status_sublabels"][shape]

    result: dict[str, Any] = {
        "shape": shape,
        "status_label": status_label,
        "status_sublabel": status_sublabel,
    }

    if shape == "preretirement":
        assert target_retirement_age is not None
        years = target_retirement_age - current_age
        fill_pct = round(current_age / target_retirement_age * 100, 2)
        result.update({
            "horizon_headline": copy["horizon_headline_templates"][shape].format(years=years),
            "horizon_sublabel": copy["horizon_sublabel_templates"][shape].format(
                current_age=current_age, target_age=target_retirement_age,
            ),
            "bar_left_label":  copy["horizon_bar_labels"][shape][0].format(current_age=current_age),
            "bar_right_label": copy["horizon_bar_labels"][shape][1].format(target_age=target_retirement_age),
            "bar_fill_pct":    fill_pct,
            "years":           years,
        })

    elif shape == "working_past_age":
        assert target_retirement_age is not None
        years_past = current_age - target_retirement_age
        result.update({
            "horizon_headline": copy["horizon_headline_templates"][shape],
            "horizon_sublabel": copy["horizon_sublabel_templates"][shape].format(
                target_age=target_retirement_age,
            ),
            "bar_left_label":  copy["horizon_bar_labels"][shape][0].format(target_age=target_retirement_age),
            "bar_right_label": copy["horizon_bar_labels"][shape][1].format(current_age=current_age),
            "bar_fill_pct":    None,  # No meaningful range; omit the bar
            "years":           years_past,
        })

    else:  # retired
        assert retired_since_age is not None, (
            "retired_since_age is required for the retired shape"
        )
        years_retired = current_age - retired_since_age
        horizon_years = (
            planning_horizon_years
            if planning_horizon_years is not None
            else copy["_retired_default_horizon_years"]
        )
        fill_pct = round(years_retired / horizon_years * 100, 2)
        result.update({
            "horizon_headline": copy["horizon_headline_templates"][shape].format(
                years=horizon_years,
            ),
            "horizon_sublabel": copy["horizon_sublabel_templates"][shape].format(
                retired_since_age=retired_since_age,
            ),
            "bar_left_label":  copy["horizon_bar_labels"][shape][0].format(retired_since_age=retired_since_age),
            "bar_right_label": copy["horizon_bar_labels"][shape][1].format(current_age=current_age),
            "bar_fill_pct":    fill_pct,
            "years":           years_retired,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Proposal copy formatting
# ─────────────────────────────────────────────────────────────────────────────


def format_proposal_copy(
    key_path: str,
    settings: Mapping[str, Any],
    **substitutions: Any,
) -> str:
    """Look up a copy string by dotted key path and format it with substitutions.

    Args:
        key_path: e.g. "cover.title_html" or "footer_template".
            Path is rooted at proposal_copy.
        settings: Output of load_settings().
        **substitutions: Passed to .format() on the resolved string.

    Returns:
        Formatted string.

    Example:
        format_proposal_copy(
            "footer_template",
            settings,
            client_name="Cole J",
            prepared_date="May 12, 2026",
        )
        # → "Confidential · Cole J · Prepared May 12, 2026"
    """
    node: Any = settings["proposal_copy"]
    for segment in key_path.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            raise KeyError(
                f"proposal_copy.{key_path} not found in settings"
            )
        node = node[segment]
    if not isinstance(node, str):
        raise TypeError(
            f"proposal_copy.{key_path} is not a string, got {type(node).__name__}"
        )
    return node.format(**substitutions) if substitutions else node


def get_option_labels(settings: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Return the three option label dicts for the recommendations page.

    Returns:
        {"option_1": {"tag": "Option 1", "title": "Recommended option"},
         "option_2": {"tag": "Option 2", "title": "Slightly more conservative"},
         "option_3": {"tag": "Option 3", "title": "Slightly more aggressive"}}
    """
    options = settings["proposal_copy"]["options"]
    return {
        key: options[key]
        for key in ("option_1", "option_2", "option_3")
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit theme derivation
# ─────────────────────────────────────────────────────────────────────────────


def streamlit_theme_from_settings(
    settings: Mapping[str, Any],
) -> dict[str, str]:
    """Derive a .streamlit/config.toml [theme] block from firm_settings.json.

    Args:
        settings: Output of load_settings().

    Returns:
        Dict of theme keys → values, ready to serialize to TOML.

        Example output:
            {
                "primaryColor": "#b8943f",
                "backgroundColor": "#ffffff",
                "secondaryBackgroundColor": "#fafaf6",
                "textColor": "#1a2030",
                "font": "sans serif",
            }

    Typical usage: call this once at app startup, write to .streamlit/config.toml
    via tomllib/tomli_w, then let Streamlit pick it up on next rerun. Or, simpler,
    use it to call st.set_page_config() with custom theme overrides directly.
    """
    brand = settings["brand"]
    return {
        "primaryColor":             brand["accent"]["gold"],
        "backgroundColor":          brand["surface"]["white"],
        "secondaryBackgroundColor": brand["surface"]["cream"],
        "textColor":                brand["text"]["primary"],
        "font":                     "sans serif",
    }


def write_streamlit_config(
    settings: Mapping[str, Any],
    path: str | Path = ".streamlit/config.toml",
) -> Path:
    """Write the Streamlit theme to a config.toml file. Creates parent dir if needed.

    Args:
        settings: Output of load_settings().
        path: Destination, defaults to .streamlit/config.toml in cwd.

    Returns:
        Path to the written file.
    """
    theme = streamlit_theme_from_settings(settings)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Hand-rolled TOML write — keeps zero runtime dependencies. The theme is
    # always flat strings, so we don't need a real TOML library.
    lines = ["[theme]"]
    for k, v in theme.items():
        lines.append(f'{k} = "{v}"')
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: bulk color resolution for a holdings dict
# ─────────────────────────────────────────────────────────────────────────────


def resolve_holdings_colors(
    holdings: Mapping[str, float],
    settings: Mapping[str, Any],
) -> dict[str, str]:
    """Return {ticker: hex_color} for every ticker in a holdings dict.

    Handy when rendering a holdings list or table — call this once and pass the
    color dict around rather than calling get_ticker_color() in a loop inside
    template code.
    """
    return {t: get_ticker_color(t, settings) for t in holdings}


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (smoke test all helpers on the real settings file)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running self-test on firm_settings.json…\n")

    s = load_settings()
    print(f"✓ Loaded settings v{s['_version']} for {s['firm']['name']}")

    # Ticker color: explicit + fallback
    schd = get_ticker_color("SCHD", s)
    unknown = get_ticker_color("ZZZX", s)
    print(f"✓ Explicit ticker SCHD → {schd}")
    print(f"✓ Fallback ticker ZZZX → {unknown} (deterministic md5 hash)")

    # Alignment tier across the spectrum
    print("\n  Alignment tiers (profile=62):")
    for current in (60, 64, 68, 73, 78, 93, 97):
        status, detail, color_key, delta = pick_alignment_tier(62, current, s)
        hex_color = s["brand"]["semantic"][color_key]
        print(f"    current={current:>3}  Δ={delta:+3d}  →  {status:14s}  "
              f"[{color_key:7s} {hex_color}]")
        print(f"                              {detail}")

    # Donut SVG
    cole_j_option_1 = {
        "SCHD": 15.0, "VOO": 10.0, "SPMO": 10.0, "SGOV": 10.0, "GLDM": 10.0,
        "AVUV": 7.0, "VEA": 6.0, "AVEM": 6.0, "FBTC": 6.0,
        "VGT": 5.0, "UTES": 5.0, "VWEHX": 5.0, "CTA": 3.0, "MNA": 2.0,
    }
    svg = generate_donut_svg(cole_j_option_1, s)
    print(f"\n✓ Donut SVG generated, {len(svg)} chars, "
          f"{svg.count('<circle')} segments")

    # Gauge SVG
    print("\n  Gauge SVGs (single-arc fill via stroke-dasharray):")
    for value in (52, 60, 66, 93):
        gauge = generate_gauge_svg(value, s)
        # Sanity-check: extract the dash length from the SVG and compare
        # against the formula.
        import re
        m = re.search(r'stroke-dasharray="([\d.]+)', gauge)
        dash = float(m.group(1)) if m else -1
        expected = s["gauge_geometry"]["arc_length"] * value / 99
        match = "✓" if abs(dash - expected) < 0.01 else "✗"
        print(f"    value={value:>3}  dash={dash:>7.3f}  "
              f"expected={expected:>7.3f}  {match}")

    # Color key resolver
    print("\n  Dotted-path color resolution:")
    for path in (
        "brand.primary.navy",
        "brand.accent.gold",
        "brand.semantic.warning_bg",
        "gauge_geometry.fill_color_key",
    ):
        try:
            if path.endswith("_key"):
                # Geometry keys point to another path; resolve transitively
                inner = resolve_color_key(path, s)
                resolved = resolve_color_key(inner, s)
                print(f"    {path:42s} → '{inner}' → {resolved}")
            else:
                resolved = resolve_color_key(path, s)
                print(f"    {path:42s} → {resolved}")
        except KeyError as e:
            print(f"    {path:42s} → ERROR: {e}")

    # Horizon shapes
    print("\n  Horizon row scenarios:")
    scenarios = [
        ("Cole J pre-retirement",
         dict(current_age=40, target_retirement_age=65, is_retired=False)),
        ("Working past target",
         dict(current_age=68, target_retirement_age=65, is_retired=False)),
        ("Recently retired",
         dict(current_age=70, target_retirement_age=65, is_retired=True,
              retired_since_age=65)),
        ("Long retirement",
         dict(current_age=78, target_retirement_age=65, is_retired=True,
              retired_since_age=63)),
    ]
    for label, kwargs in scenarios:
        h = compute_horizon_data(settings=s, **kwargs)
        bar = f" bar={h['bar_fill_pct']}%" if h["bar_fill_pct"] is not None else " no bar"
        print(f"    {label:25s} → shape={h['shape']:18s} "
              f"headline={h['horizon_headline']!r:18s}{bar}")

    # Copy formatting
    footer = format_proposal_copy(
        "footer_template", s,
        client_name="Cole J",
        prepared_date="May 12, 2026",
    )
    print(f"✓ Formatted copy: {footer}")

    labels = get_option_labels(s)
    print("✓ Option labels:")
    for k, v in labels.items():
        print(f"    {k}: {v['tag']} · {v['title']}")

    # Streamlit theme
    theme = streamlit_theme_from_settings(s)
    print(f"\n✓ Streamlit theme: {theme}")

    print("\nAll helpers working.\n")
