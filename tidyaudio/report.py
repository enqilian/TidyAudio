"""质量报告：把输出文件夹里的记录画成图，生成一个打开就能看的 report.html。

只读取输出文件夹里已有的文件（manifest.csv、dropped.csv、processing_steps.csv、
dataset/summary.csv、config.yaml），所以任何一次的输出都可以重新生成报告。

报告内容：
  1. Overview：输入多少、输出多少、丢掉多少、总时长
  2. Source audio（处理前）：采样率、声道、时长分布
  3. Processed audio（处理后）：时长、电平分布
  4. Classes：类别分布（数据集 CSV 里有类别列才有）
  5. Quality issues：削波、直流偏移、归一化没达到目标、几乎没有声音的输出
  6. Dropped files：按原因统计
  7. Processing steps
"""

import base64
import csv
import html
import io
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # 不弹窗口，只画到图片里
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

# 字体：英文优先，后面的中文字体用来显示文件名里可能出现的中文
plt.rcParams["font.sans-serif"] = [
    "Helvetica Neue", "Arial", "PingFang SC", "Arial Unicode MS", "Microsoft YaHei",
    "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACCENT = "#3a3a3c"   # 图表颜色：深灰，和界面的黑白灰风格一致
TOP_N = 20           # 问题文件列表最多显示多少条
QUIET_DB = -60.0     # 平均电平低于这个值，算"几乎没有声音"
FLOOR_DB = -100.0    # 画图时低于这个值的不画（数字静音会算出 -300 dB 这种极端值，把图压扁）


def write_report(out_dir: str | Path) -> Path:
    """读取输出文件夹，生成 report.html，返回它的路径。"""
    out_dir = Path(out_dir)
    manifest = _read_csv(out_dir / "manifest.csv")
    dropped = _read_csv(out_dir / "dropped.csv")
    steps = _read_csv(out_dir / "processing_steps.csv")
    summary = _read_csv(out_dir / "dataset" / "summary.csv")
    run_info = _run_info(out_dir / "config.yaml")

    # 原始文件：manifest 里一条源文件可能切成了好几段，按 source 去重
    sources = {}
    for r in manifest:
        sources.setdefault(r["source"], r)
    originals = list(sources.values())

    sections = [
        _overview(manifest, dropped, originals, run_info),
        _original_section(originals),
        _output_section(manifest),
        _class_section(summary),
        _problem_section(manifest, originals),
        _dropped_section(dropped),
        _steps_section(steps),
    ]
    page = _PAGE.format(
        title=html.escape(out_dir.name),
        time=html.escape(str(run_info.get("time", "")).replace("T", " ")),
        body="\n".join(s for s in sections if s),
    )
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


# ---------------------------------------------------------------- 各个部分

def _overview(manifest, dropped, originals, run_info) -> str:
    orig_total = sum(_num(r.get("orig_duration_sec")) for r in originals)
    out_total = sum(_num(r.get("duration_sec")) for r in manifest)
    cards = [
        ("Input files", run_info.get("input_files", "—")),
        ("Output files", len(manifest)),
        ("Dropped", len(dropped)),
        ("Source duration", _fmt_duration(orig_total)),
        ("Output duration", _fmt_duration(out_total)),
    ]
    items = "".join(
        f'<div class="stat"><div class="value">{html.escape(str(v))}</div>'
        f'<div class="label">{html.escape(k)}</div></div>' for k, v in cards)
    return f'<section><h2>Overview</h2><div class="stats">{items}</div></section>'


def _original_section(originals) -> str:
    if not originals or "orig_sample_rate" not in originals[0]:
        return ""
    sr = Counter(f'{r["orig_sample_rate"]} Hz' for r in originals)
    ch = Counter(_channel_name(r["orig_channels"]) for r in originals)
    durations = [_num(r["orig_duration_sec"]) for r in originals]
    if len(sr) > 1 or len(ch) > 1:
        hint = "Sample rates or channel counts differ; resampling and channel conversion are usually needed."
    else:
        hint = "All files share the same sample rate and channel count."
    return (
        "<section><h2>Source audio</h2>"
        f"<p class='hint'>Source files that were kept. {hint}</p>"
        '<div class="row">'
        + _count_table("Sample rate", sr) + _count_table("Channels", ch)
        + "</div>"
        + _histogram(durations, "Duration per source file", "Duration (s)")
        + "</section>"
    )


def _output_section(manifest) -> str:
    if not manifest:
        return ""
    durations = [_num(r["duration_sec"]) for r in manifest]
    parts = ["<section><h2>Processed audio</h2>"]
    if max(durations) - min(durations) < 1e-6:
        parts.append(f"<p>Every output file is <b>{durations[0]:g} s</b> long.</p>")
    else:
        parts.append(_histogram(durations, "Duration per output file", "Duration (s)"))
    rms = [_rms(r) for r in manifest]
    shown = [v for v in rms if v >= FLOOR_DB]
    if shown:
        hidden = len(rms) - len(shown)
        note = (f" {hidden} files below {FLOOR_DB:g} dBFS (digital silence) are not shown."
                if hidden else "")
        parts.append("<p class='hint'>A narrow peak means consistent levels; after normalization "
                     f"this should be tight.{note}</p>")
        parts.append(_histogram(shown, "Average level (RMS) per output file", "dBFS"))
    parts.append("</section>")
    return "".join(parts)


def _class_section(summary) -> str:
    rows = [r for r in summary if list(r.values())[0] != "TOTAL"]
    if not rows:
        return ""
    label_col = list(rows[0])[0]
    labels = [r[label_col] or "(empty)" for r in rows]
    counts = [int(r["count"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1))
    ax.barh(labels[::-1], counts[::-1], color=ACCENT, height=0.65)
    for i, c in enumerate(counts[::-1]):
        ax.text(c, i, f"  {c}", va="center", fontsize=9, color="#6e6e73")
    ax.set_xlabel("Files")
    _tidy(ax)
    ratio = max(counts) / max(min(counts), 1)
    hint = (f"The largest class has {ratio:.1f}x as many files as the smallest. Consider class "
            "weights or resampling when training." if ratio >= 3 else "Classes are fairly balanced.")
    return (f"<section><h2>Classes <span class='muted'>by {html.escape(label_col)}</span></h2>"
            f"<p class='hint'>{hint}</p>{_img(fig)}</section>")


def _problem_section(manifest, originals) -> str:
    parts = []
    if originals and "clipping_ratio_pct" not in originals[0]:
        parts.append("<p class='hint'>Clipping Detection was not enabled for this run.</p>")

    if originals and "clipping_ratio_pct" in originals[0]:
        clipped = sorted((r for r in originals if _num(r.get("clipping_ratio_pct")) > 0),
                         key=lambda r: -_num(r["clipping_ratio_pct"]))
        parts.append(_issue_title("Clipping", len(clipped), len(originals), "source files"))
        if clipped:
            parts.append("<p class='hint'>Files with a high ratio and long runs of clipped samples "
                         "usually have audible distortion.</p>")
            parts.append(_table(
                ["Source file", "Clipped (%)", "Longest run (samples)"],
                [[r["source"], r["clipping_ratio_pct"], r.get("longest_clip_run", "")]
                 for r in clipped[:TOP_N]], more=len(clipped) - TOP_N))

    if originals and "dc_offset" in originals[0]:
        dc = sorted((r for r in originals if _num(r.get("dc_offset")) > 0.01),
                    key=lambda r: -_num(r["dc_offset"]))
        parts.append(_issue_title("DC offset above 0.01", len(dc), len(originals), "source files"))
        if dc:
            parts.append(_table(["Source file", "DC offset"],
                                [[r["source"], r["dc_offset"]] for r in dc[:TOP_N]],
                                more=len(dc) - TOP_N))

    # 归一化在切片之前做，所以按源文件统计（同一个文件切出来的段，结果都一样）
    if originals and "normalize_gain_db" in originals[0]:
        limited = [r for r in originals if str(r.get("normalize_limited", "")).lower() == "true"]
        parts.append(_issue_title("Normalization target not reached", len(limited),
                                  len(originals), "source files"))
        if limited:
            parts.append("<p class='hint'>Reaching the target would have clipped these files, so "
                         "less gain was applied. They will sound slightly quieter.</p>")
            parts.append(_table(["Source file", "Applied gain (dB)"],
                                [[r["source"], r["normalize_gain_db"]] for r in limited[:TOP_N]],
                                more=len(limited) - TOP_N))

    if manifest and "rms_dbfs" in manifest[0]:
        quiet = sorted((r for r in manifest if _rms(r) < QUIET_DB), key=_rms)
        parts.append(_issue_title(f"Nearly silent output (RMS below {QUIET_DB:g} dBFS)",
                                  len(quiet), len(manifest), "output files"))
        if quiet:
            parts.append("<p class='hint'>Usually segments cut from silent parts of a recording, or "
                         "short files padded with a lot of silence. Worth a listen before training.</p>")
            parts.append(_table(["Output file", "RMS (dBFS)"],
                                [[r["path"], r["rms_dbfs"]] for r in quiet[:TOP_N]],
                                more=len(quiet) - TOP_N))

    return "<section><h2>Quality issues</h2>" + "".join(parts) + "</section>"


def _dropped_section(dropped) -> str:
    if not dropped:
        return "<section><h2>Dropped files</h2><p>No files were dropped.</p></section>"
    kinds = Counter(_reason_kind(r.get("reason", "")) for r in dropped)
    return ("<section><h2>Dropped files</h2>"
            "<p class='hint'>Per-file reasons are in dropped.csv.</p>"
            + _table(["Reason", "Count"], kinds.most_common()) + "</section>")


def _steps_section(steps) -> str:
    if not steps:
        return ""
    fields = list(steps[0])
    return ("<section><h2>Processing steps</h2>"
            + _table([f.capitalize() for f in fields], [[r[f] for f in fields] for r in steps])
            + "</section>")


# ---------------------------------------------------------------- 工具函数

def _issue_title(name: str, n: int, total: int, unit: str) -> str:
    cls = "badge warn" if n else "badge ok"
    return f"<h3>{html.escape(name)} <span class='{cls}'>{n} / {total} {unit}</span></h3>"


def _histogram(values, title, xlabel) -> str:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.hist(values, bins=min(40, max(5, len(values) // 5)), color=ACCENT, edgecolor="white",
            linewidth=0.5)
    ax.set_title(title, loc="left", fontsize=11, color="#1d1d1f")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Files")
    _tidy(ax)
    return _img(fig)


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d2d2d7")
    ax.tick_params(colors="#6e6e73", labelsize=9)
    ax.xaxis.label.set_color("#6e6e73")
    ax.yaxis.label.set_color("#6e6e73")


def _img(fig) -> str:
    """把图转成 base64 嵌进网页里，这样 report.html 一个文件就能看，不依赖图片文件。"""
    buf = io.BytesIO()
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 缺字体时的警告
        fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _count_table(name, counter: Counter) -> str:
    return _table([name, "Files"], sorted(counter.items(), key=lambda kv: -kv[1]))


def _table(headers, rows, more: int = 0) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    extra = f"<p class='hint'>…and {more} more. See manifest.csv for the full list.</p>" if more > 0 else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{extra}"


def _reason_kind(reason: str) -> str:
    """原因的类别：括号或冒号之前的部分，比如 "Too short (0.30s < 0.5s)" → "Too short"。"""
    for sep in ("(", ":"):
        reason = reason.split(sep)[0]
    return reason.strip() or "(not specified)"


def _channel_name(n: str) -> str:
    return {"1": "Mono", "2": "Stereo"}.get(n, f"{n} channels")


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _rms(row) -> float:
    """平均电平；全静音时 CSV 里是 -inf，这里也返回 -inf。"""
    try:
        return float(row.get("rms_dbfs"))
    except (TypeError, ValueError):
        return float("-inf")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _run_info(path: Path) -> dict:
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("run_info", {})


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quality Report · {title}</title>
<style>
  :root {{ --ink: #1d1d1f; --muted: #6e6e73; --line: #e5e5ea; --bg: #f5f5f7; --card: #ffffff; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial,
         sans-serif; max-width: 960px; margin: 0 auto; padding: 48px 20px 72px;
         color: var(--ink); background: var(--bg); line-height: 1.5; font-size: 14px;
         -webkit-font-smoothing: antialiased; }}
  header {{ margin-bottom: 12px; }}
  .eyebrow {{ color: var(--muted); font-weight: 600; font-size: 13px; }}
  h1 {{ margin: 2px 0 4px; font-size: 32px; font-weight: 700; letter-spacing: -0.02em; }}
  .time, .muted {{ color: var(--muted); font-weight: 400; }}
  section {{ background: var(--card); border-radius: 14px; padding: 22px 26px; margin-top: 18px; }}
  h2 {{ margin: 0 0 8px; font-size: 19px; font-weight: 600; letter-spacing: -0.01em; }}
  h3 {{ font-size: 14px; font-weight: 600; margin: 20px 0 6px; }}
  .hint {{ color: var(--muted); }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
  .stat {{ background: var(--bg); border-radius: 12px; padding: 16px 18px; }}
  .stat .value {{ font-size: 26px; font-weight: 600; letter-spacing: -0.02em; }}
  .stat .label {{ color: var(--muted); font-size: 12px; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 32px; }}
  .badge {{ font-size: 12px; font-weight: 600; border-radius: 999px; padding: 2px 10px;
           margin-left: 6px; }}
  .badge.warn {{ background: #fff3e0; color: #b25000; }}
  .badge.ok {{ background: #e9f7ee; color: #1f7a3a; }}
  table {{ border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 7px 12px 7px 0; text-align: left;
           word-break: break-all; vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  .row table {{ width: auto; }}
  img {{ max-width: 100%; height: auto; margin-top: 8px; }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">TidyAudio · Quality Report</div>
  <h1>{title}</h1>
  <div class="time">{time}</div>
</header>
{body}
</body>
</html>
"""
