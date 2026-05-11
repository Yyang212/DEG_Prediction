from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "outputs" / "deg_xgboost_analysis" / "deg_xgboost_payload.json"
OUT = ROOT / "outputs" / "deg_xgboost_analysis" / "prediction_results.png"


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def nice_ticks(lo: float, hi: float, count: int = 5):
    if math.isclose(lo, hi):
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def draw_axes(draw, rect, x_ticks, y_ticks, x_fmt, y_fmt, color="#D1D5DB"):
    left, top, right, bottom = rect
    draw.line((left, top, left, bottom), fill="#374151", width=2)
    draw.line((left, bottom, right, bottom), fill="#374151", width=2)
    for x, label in x_ticks:
        draw.line((x, bottom, x, bottom + 6), fill="#374151", width=1)
        draw.line((x, top, x, bottom), fill=color, width=1)
        draw.text((x - 40, bottom + 8), label, font=x_fmt, fill="#374151")
    for y, label in y_ticks:
        draw.line((left - 6, y, left, y), fill="#374151", width=1)
        draw.line((left, y, right, y), fill=color, width=1)
        tw, th = text_size(draw, label, y_fmt)
        draw.text((left - 12 - tw, y - th / 2), label, font=y_fmt, fill="#374151")


def map_range(v, src_lo, src_hi, dst_lo, dst_hi):
    if math.isclose(src_hi, src_lo):
        return (dst_lo + dst_hi) / 2
    return dst_lo + (v - src_lo) * (dst_hi - dst_lo) / (src_hi - src_lo)


def main():
    data = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    df = pd.DataFrame(data["sheets"]["Test_Predictions"])
    df["sample_time"] = pd.to_datetime(df["sample_time"])
    df = df.sort_values("sample_time").reset_index(drop=True)

    metrics = data["summary"]["test_metrics"]
    n = len(df)
    actual = df["target_DEG_pct"].astype(float).tolist()
    pred = df["predicted_DEG_pct"].astype(float).tolist()
    times = df["sample_time"].tolist()
    residual = (df["predicted_DEG_pct"] - df["target_DEG_pct"]).astype(float).tolist()

    W, H = 1600, 1100
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    title_font = font(34, bold=True)
    subtitle_font = font(20)
    axis_font = font(18)
    small_font = font(16)
    tick_font = font(15)

    draw.text((60, 30), "DEG Prediction Results", font=title_font, fill="#111827")
    subtitle = (
        f"Test set: n={n} | RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
        f"R2={metrics['r2']:.3f} | MAPE={metrics['mape_pct']:.2f}%"
    )
    draw.text((60, 78), subtitle, font=subtitle_font, fill="#374151")

    # Top panel: time series
    left, top, right, bottom = 80, 150, 1520, 610
    draw.rounded_rectangle((40, 130, 1560, 640), radius=18, outline="#E5E7EB", width=2, fill="#FAFAFA")
    draw.text((70, 145), "Actual vs Predicted over Time", font=subtitle_font, fill="#111827")
    plot = (left, top, right, bottom)

    x_min = 0
    x_max = max(n - 1, 1)
    y_lo = min(min(actual), min(pred)) - 0.01
    y_hi = max(max(actual), max(pred)) + 0.01

    xticks = []
    for i in range(min(5, n)):
        idx = round(i * (n - 1) / max(4, 1))
        x = map_range(idx, x_min, x_max, left, right)
        xticks.append((x, times[idx].strftime("%m-%d")))
    yticks = []
    for yv in nice_ticks(y_lo, y_hi, 5):
        y = map_range(yv, y_lo, y_hi, bottom, top)
        yticks.append((y, f"{yv:.3f}"))
    draw_axes(draw, plot, xticks, yticks, tick_font, tick_font)

    for i in range(n):
        x = map_range(i, x_min, x_max, left, right)
        y_a = map_range(actual[i], y_lo, y_hi, bottom, top)
        y_p = map_range(pred[i], y_lo, y_hi, bottom, top)
        if i > 0:
            x0 = map_range(i - 1, x_min, x_max, left, right)
            y0a = map_range(actual[i - 1], y_lo, y_hi, bottom, top)
            y0p = map_range(pred[i - 1], y_lo, y_hi, bottom, top)
            draw.line((x0, y0a, x, y_a), fill="#2563EB", width=3)
            draw.line((x0, y0p, x, y_p), fill="#DC2626", width=3)
        draw.ellipse((x - 4, y_a - 4, x + 4, y_a + 4), fill="#2563EB")
        draw.ellipse((x - 4, y_p - 4, x + 4, y_p + 4), fill="#DC2626")

    legend_x = 1220
    legend_y = 165
    draw.rectangle((legend_x, legend_y, legend_x + 260, legend_y + 80), fill="white", outline="#D1D5DB")
    draw.line((legend_x + 16, legend_y + 22, legend_x + 46, legend_y + 22), fill="#2563EB", width=4)
    draw.text((legend_x + 56, legend_y + 12), "Actual", font=small_font, fill="#111827")
    draw.line((legend_x + 16, legend_y + 52, legend_x + 46, legend_y + 52), fill="#DC2626", width=4)
    draw.text((legend_x + 56, legend_y + 42), "Predicted", font=small_font, fill="#111827")

    # Bottom left: parity scatter
    draw.rounded_rectangle((40, 680, 1040, 1060), radius=18, outline="#E5E7EB", width=2, fill="#FAFAFA")
    draw.text((70, 695), "Parity Plot", font=subtitle_font, fill="#111827")
    l2, t2, r2, b2 = 100, 740, 980, 1020
    x_lo, x_hi = min(actual) - 0.01, max(actual) + 0.01
    y_lo2, y_hi2 = min(pred) - 0.01, max(pred) + 0.01
    parity_ticks_x = []
    parity_ticks_y = []
    for v in nice_ticks(x_lo, x_hi, 5):
        x = map_range(v, x_lo, x_hi, l2, r2)
        parity_ticks_x.append((x, f"{v:.3f}"))
    for v in nice_ticks(y_lo2, y_hi2, 5):
        y = map_range(v, y_lo2, y_hi2, b2, t2)
        parity_ticks_y.append((y, f"{v:.3f}"))
    draw_axes(draw, (l2, t2, r2, b2), parity_ticks_x, parity_ticks_y, tick_font, tick_font)
    draw.line((l2, b2, r2, t2), fill="#111827", width=2)
    for a, p in zip(actual, pred):
        x = map_range(a, x_lo, x_hi, l2, r2)
        y = map_range(p, y_lo2, y_hi2, b2, t2)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#2563EB", outline="white")

    # Bottom right: residual summary
    draw.rounded_rectangle((1080, 680, 1560, 1060), radius=18, outline="#E5E7EB", width=2, fill="#FAFAFA")
    draw.text((1110, 695), "Residuals", font=subtitle_font, fill="#111827")
    draw.text((1110, 735), f"Mean residual: {pd.Series(residual).mean():+.4f}", font=small_font, fill="#374151")
    draw.text((1110, 765), f"Max abs error: {max(abs(x) for x in residual):.4f}", font=small_font, fill="#374151")
    draw.text((1110, 795), f"Bias direction: {'over-predict' if pd.Series(residual).mean() > 0 else 'under-predict'}", font=small_font, fill="#374151")

    bar_left, bar_top, bar_right, bar_bottom = 1120, 860, 1530, 1000
    res_min = min(residual)
    res_max = max(residual)
    axis_zero = map_range(0, res_min, res_max, bar_bottom, bar_top)
    draw.line((bar_left, axis_zero, bar_right, axis_zero), fill="#6B7280", width=2)
    bar_w = (bar_right - bar_left) / n
    for i, res in enumerate(residual):
        x0 = bar_left + i * bar_w + 4
        x1 = bar_left + (i + 1) * bar_w - 4
        y = map_range(res, res_min, res_max, bar_bottom, bar_top)
        if res >= 0:
            draw.rectangle((x0, y, x1, axis_zero), fill="#DC2626")
        else:
            draw.rectangle((x0, axis_zero, x1, y), fill="#2563EB")

    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
