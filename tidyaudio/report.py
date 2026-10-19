"""质量报告：把输出文件夹里的记录画成图，生成一个打开就能看的 report.html。

只读取输出文件夹里已有的文件（manifest.csv、dropped.csv、processing_steps.csv、
dataset/summary.csv、config.yaml），所以任何一次的输出都可以重新生成报告。

报告分两层：
  第一层：每次都有的部分
    Summary（结论）、Overview（数量和时长，以及时长为什么变了）、Format（格式前后对比）、
    Duration（时长分布）、Classes（类别分布）、Silent output（几乎没声音的输出）、
    Dropped files（哪一步丢了什么）
  第二层：开了哪个处理才出现对应的部分，按实际执行顺序排列
    重采样、声道转换、位深转换的结果在 Format 里已经能看到，不再单独成段
  最后：Workflow（处理步骤和参数）

每个部分除了网页内容，还会给出几条"发现"（findings），汇总到最上面的 Summary。
"""

import base64
import csv
import html
import io
import math
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")   # 不弹窗口，只画到图片里
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from .steps import REGISTRY  # noqa: E402

# 字体：英文优先，后面的中文字体用来显示文件名里可能出现的中文
plt.rcParams["font.sans-serif"] = [
    "Helvetica Neue", "Arial", "PingFang SC", "Arial Unicode MS", "Microsoft YaHei",
    "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#3a3a3c"       # 图表主色：深灰，和界面的黑白灰风格一致
LIGHT = "#c7c7cc"     # "处理前"用浅灰
TOP_N = 20            # 问题文件列表最多显示多少条
QUIET_DB = -60.0      # 平均电平低于这个值，算"几乎没有声音"
SILENT_DB = -200.0    # 低于这个值就是数字静音（全 0），直接写 "silent"
FLOOR_DB = -100.0     # 画图时低于这个值的不画（数字静音会算出 -300 dB 这种极端值，把图压扁）

# 削波的严重程度。只看比例会误判：合成的警报声、方波经常"碰到顶"很多次，但每次只有 1、2 个点，
# 听不出来；真正的破音是波形顶部被削平一段（连续很多个点）。
NOTICEABLE_RUN = 4        # 连续 4 个点以上被削平：可能听得出毛刺
NOTICEABLE_PCT = 0.1      # 或者削波的点占 0.1% 以上
SEVERE_RUN = 20           # 连续 20 个点以上
SEVERE_PCT = 1.0          # 或者占 1% 以上
BIG_CHANGE_DB = 6.0   # 高通 / 降噪让电平下降超过这么多，值得听一下
NORM_TOLERANCE_DB = 1.0   # 归一化差目标不到 1 dB，听不出区别，不算问题


@dataclass
class Section:
    title: str
    body: str
    findings: list = field(default_factory=list)   # [(级别, 文字)]，级别：warn / info / ok
    subtitle: str = ""


class Run:
    """一次输出的全部记录，以及几个常用的派生数据。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.manifest = _read_csv(out_dir / "manifest.csv")
        self.dropped = _read_csv(out_dir / "dropped.csv")
        self.step_rows = _read_csv(out_dir / "processing_steps.csv")
        self.summary = _read_csv(out_dir / "dataset" / "summary.csv")
        config = _read_yaml(out_dir / "config.yaml")
        self.run_info = config.get("run_info") or {}
        self.settings = config.get("steps") or {}

        # 实际执行的每一步：(key, processing_steps.csv 里的那一行)。脚本的 key 是 "script"
        labels = {cls.label: key for key, cls in REGISTRY.items()}
        self.workflow = []
        for row in self.step_rows[1:]:
            name = row.get("name", "")
            key = "script" if name.startswith("Python script") else labels.get(name, name)
            self.workflow.append((key, row))

        # 原始文件：manifest 里一条源文件可能切成了好几段，按 source 去重
        self.first_path = {}
        for r in self.manifest:
            self.first_path.setdefault(r["source"], r["path"])
        sources = {}
        for r in self.manifest + self.dropped:
            if r.get("orig_sample_rate"):
                sources.setdefault(r["source"], r)
        self.sources = list(sources.values())   # 所有读得出来的源文件（包括后来被丢掉的）

    def enabled(self, key: str) -> bool:
        return any(k == key for k, _ in self.workflow)

    def position(self, key: str) -> int:
        return next((i for i, (k, _) in enumerate(self.workflow) if k == key), -1)

    def params(self, key: str) -> dict:
        return self.settings.get(key) or {}

    def label(self, key: str) -> str:
        return REGISTRY[key].label if key in REGISTRY else key

    def units(self, column: str, key: str) -> list[dict]:
        """某一步处理时的"每一条音频"。

        在切片之前的步骤，一个源文件只算一次（它切出来的每段记录都一样）；
        在切片之后的步骤，每一段各算一次。被丢掉的也算在内（它们在 dropped.csv 里）。
        """
        after_segment = self.enabled("segment") and self.position(key) > self.position("segment")
        seen = {}
        for r in self.manifest + self.dropped:
            if r.get(column) in (None, ""):
                continue
            seen.setdefault((r["source"], r.get("segment_index", "") if after_segment else ""), r)
        return list(seen.values())

    def play_path(self, row: dict) -> str | None:
        """这条记录对应的输出音频（用来在报告里播放）。被丢掉的没有。"""
        return row.get("path") or self.first_path.get(row.get("source"))


def write_report(out_dir: str | Path) -> Path:
    """读取输出文件夹，生成 report.html，返回它的路径。"""
    out_dir = Path(out_dir)
    run = Run(out_dir)

    common = [_overview(run), _format(run), _durations(run), _classes(run),
              _silent_output(run), _dropped(run)]
    details, scripts_done = [], False
    for key, _ in run.workflow:
        if key == "script":
            if not scripts_done:
                details.append(_scripts(run))
                scripts_done = True
        elif key in DETAIL_SECTIONS:
            row = _step_row(run, key)
            if row and row.get("input") not in (None, "") and _int(row["input"]) == 0:
                # 前面的步骤已经把文件全丢了，这一步什么都没处理
                details.append(Section(run.label(key), "<p class='hint'>No files reached this "
                                                       "step; they were all dropped earlier.</p>"))
            else:
                details.append(DETAIL_SECTIONS[key](run))
    common = [s for s in common if s]
    details = [s for s in details if s]

    body = [_render(_summary(common + details), "summary")] + [_render(s) for s in common]
    if details:
        body.append("<h2 class='part'>Processing details</h2>"
                    "<p class='hint part-hint'>One section for each processing step that was "
                    "turned on, in the order they ran.</p>")
        body += [_render(s) for s in details]
    body.append(_render(_workflow(run)))

    page = _PAGE.format(
        title=html.escape(out_dir.name),
        time=html.escape(str(run.run_info.get("time", "")).replace("T", " ")),
        version=html.escape(str(run.run_info.get("tidyaudio_version", ""))),
        body="\n".join(body),
        script=_PLAYER_JS,
    )
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


# ================================================================ 第一层：每次都有

def _summary(sections: list[Section]) -> Section:
    order = {"warn": 0, "info": 1, "ok": 2}
    findings = sorted((f for s in sections for f in s.findings), key=lambda f: order[f[0]])
    warns = sum(1 for level, _ in findings if level == "warn")
    head = (f"<p class='verdict warn'>{_plural(warns, 'thing')} worth checking before training."
            "</p>" if warns else
            "<p class='verdict ok'>No problems found. The dataset is ready to use.</p>")
    items = "".join(f"<li class='{level}'>{html.escape(text)}</li>" for level, text in findings)
    return Section("Summary", head + (f"<ul class='findings'>{items}</ul>" if items else ""))


def _overview(run: Run) -> Section:
    load = run.step_rows[0] if run.step_rows else {}
    n_in = _int(run.run_info.get("input_files", load.get("input", len(run.sources))))
    source_sec = _num(load.get("output_sec")) or sum(_num(r.get("orig_duration_sec"))
                                                     for r in run.sources)
    out_sec = sum(_num(r.get("duration_sec")) for r in run.manifest)
    cards = _cards([
        ("Input files", f"{n_in:,}"),
        ("Output files", f"{len(run.manifest):,}"),
        ("Dropped", f"{len(run.dropped):,}"),
        ("Source duration", _fmt_duration(source_sec)),
        ("Output duration", _fmt_duration(out_sec)),
    ])
    parts, findings = [cards], []
    if not run.manifest:
        findings.append(("warn", "There are no output files: every file was dropped. "
                                 "See Dropped files for the reasons."))

    # 输出比输入多：切片
    seg_sources = len({r["source"] for r in run.manifest if r.get("segment_index", "") != ""})
    if seg_sources and len(run.manifest) > seg_sources:
        parts.append(f"<p>There are more output files than input files because segmentation cut "
                     f"{_plural(seg_sources, 'file')} into "
                     f"{len(run.manifest) / seg_sources:.1f} segments on average.</p>")

    # 时长变了：一步一步说清楚是哪一步变的、为什么
    changes = []
    for key, row in run.workflow:
        delta = _num(row.get("output_sec")) - _num(row.get("input_sec"))
        if abs(delta) < max(1.0, 0.002 * max(source_sec, 1)):
            continue
        changes.append((row["name"], delta, _why(run, key, row, delta)))
    unread = _int(load.get("dropped", 0))
    if changes:
        rows = [[name, _signed_duration(d), why] for name, d, why in changes]
        parts.append("<h3>Why the duration changed</h3>"
                     + _table(["Step", "Change", "Reason"], rows))
        if unread:
            parts.append(f"<p class='hint'>{_plural(unread, 'file')} could not be read and "
                         "are not counted in the source duration.</p>")
        padding = sum(_num(r.get("segment_padded_sec")) for r in run.manifest)
        if padding > 0.05 * max(source_sec, 1):
            findings.append(("info", f"Output is {_fmt_duration(padding)} longer than the "
                                     "source because segments were padded with silence."))
    return Section("Overview", "".join(parts), findings)


def _why(run: Run, key: str, row: dict, delta: float) -> str:
    dropped = _int(row.get("dropped"))
    if key == "segment":
        padding = sum(_num(r.get("segment_padded_sec")) for r in run.manifest)
        bits = []
        if padding:
            padded = sum(1 for r in run.manifest if _num(r.get("segment_padded_sec")) > 0)
            bits.append(f"{_fmt_duration(padding)} of silence added to fill the last segment "
                        f"of {_plural(padded, 'file')}")
        if delta < 0 or run.params("segment").get("last") in ("drop",):
            bits.append("short last segments were dropped" if dropped == 0
                        else f"{_plural(dropped, 'file')} dropped")
        if _num(run.params("segment").get("overlap_sec")) > 0:
            bits.append("overlapping segments repeat some audio")
        return "; ".join(bits) or "segmentation"
    if key == "silence_removal":
        text = "silence trimmed"
        return text + (f"; {_plural(dropped, 'file')} {_verb(dropped, 'were')} all silence" if dropped else "")
    if dropped:
        return f"{_plural(dropped, 'file')} dropped"
    return "audio length changed by this step"


def _format(run: Run) -> Section | None:
    if not run.sources:
        return None
    props = [("Sample rate", "sample rates", "orig_sample_rate", "sample_rate",
              lambda v: f"{_int(v):,} Hz", "resample"),
             ("Channels", "channel counts", "orig_channels", "channels", _channel_name,
              "channels"),
             ("Bit depth", "bit depths", "orig_subtype", "subtype", _bit_depth, "bit_depth")]
    rows, findings = [], []
    for name, plural, before_col, after_col, fmt, fix in props:
        if not run.sources[0].get(before_col):
            continue
        before = Counter(fmt(r[before_col]) for r in run.sources)
        after = Counter(fmt(r[after_col]) for r in run.manifest if r.get(after_col))
        rows.append((name, before, after))
        if len(after) > 1:
            findings.append(("warn", f"Output files still have mixed {plural} ("
                                     + ", ".join(after) + "). Turn on "
                                     + REGISTRY[fix].label + " to unify them."))
        elif len(before) > 1 and after:
            findings.append(("ok", f"Mixed {plural} were unified to {next(iter(after))}."))
    if not rows:
        return None

    def cell(counter):
        if not counter:
            return "<span class='muted'>—</span>"
        if len(counter) == 1:
            return html.escape(next(iter(counter)))
        return "<br>".join(f"{html.escape(k)} <span class='muted'>· {v:,}</span>"
                           for k, v in counter.most_common())

    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{cell(b)}</td><td class='arrow'>→</td>"
        f"<td>{cell(a)}</td></tr>" for name, b, a in rows)
    table = ("<table class='compare'><thead><tr><th></th><th>Source</th><th></th><th>Output</th>"
             f"</tr></thead><tbody>{body}</tbody></table>")
    hint = ("<p class='hint'>Source counts include every file that could be read, also those "
            "dropped later.</p>")
    return Section("Format", table + hint, findings)


def _durations(run: Run) -> Section | None:
    if not run.manifest:
        return None
    before = [_num(r.get("orig_duration_sec")) for r in run.sources]
    after = [_num(r.get("duration_sec")) for r in run.manifest]
    charts = [_hist([(before, "Source files", INK)], "Duration (s)", title="Source files")]
    if max(after) - min(after) < 1e-6:
        side = (f"<div class='callout'><div class='value'>{after[0]:g} s</div>"
                "<div class='label'>Every output file has the same length</div></div>")
    else:
        side = _hist([(after, "Output files", INK)], "Duration (s)", title="Output files")
    stats = _table(["", "Shortest", "Median", "Longest"], [
        ["Source files", *(_fmt_seconds(v) for v in _spread(before))],
        ["Output files", *(_fmt_seconds(v) for v in _spread(after))],
    ])
    return Section("Duration", f"<div class='pair'>{charts[0]}{side}</div>{stats}")


def _classes(run: Run) -> Section | None:
    rows = [r for r in run.summary if list(r.values())[0] != "TOTAL"]
    if not rows:
        return None
    label_col = list(rows[0])[0]
    labels = [r[label_col] or "(empty)" for r in rows]
    counts = [_int(r.get("count")) for r in rows]
    minutes = [_num(r.get("total_duration_s")) / 60 for r in rows]

    height = 0.3 * len(rows) + 1.0
    fig, axes = plt.subplots(1, 2, figsize=(9, height), sharey=True)
    for ax, values, xlabel, fmt in ((axes[0], counts, "Files", "{:,.0f}"),
                                    (axes[1], minutes, "Duration (min)", "{:,.1f}")):
        ax.barh(labels[::-1], values[::-1], color=INK, height=0.65)
        for i, v in enumerate(values[::-1]):
            ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=8.5, color="#6e6e73")
        ax.set_xlabel(xlabel)
        ax.margins(x=0.18)
        _tidy(ax)

    findings = []
    ratio = max(counts) / max(min(counts), 1)
    big, small = labels[counts.index(max(counts))], labels[counts.index(min(counts))]
    if ratio >= 3:
        hint = (f"<b>{html.escape(big)}</b> has {ratio:.1f}× as many files as "
                f"<b>{html.escape(small)}</b>. Consider class weights or balanced sampling when "
                "training.")
        findings.append(("warn", f"Classes are imbalanced: {big} has {ratio:.1f}× as many files "
                                 f"as {small}."))
    else:
        hint = "Classes are fairly balanced."
        findings.append(("ok", f"{_plural(len(rows), 'class', 'classes')}, fairly balanced."))
    if run.enabled("segment"):
        hint += " Counts are output files, so a long recording counts once per segment."
    return Section("Classes", f"<p class='hint'>{hint}</p>{_img(fig)}", findings,
                   subtitle=f"by {label_col}")


def _silent_output(run: Run) -> Section | None:
    if not run.manifest or "rms_dbfs" not in run.manifest[0]:
        return None
    quiet = sorted((r for r in run.manifest if _rms(r) < QUIET_DB), key=_rms)
    if not quiet:
        return Section("Silent output",
                       f"<p>{_badge(0, len(run.manifest), 'output files')} No output file has an "
                       f"average level below {QUIET_DB:g} dBFS.</p>",
                       [("ok", "No silent output files.")])
    digital = sum(1 for r in quiet if _rms(r) < SILENT_DB)
    hint = ("Usually segments cut from silent parts of a recording, or short files padded with "
            "a lot of silence. Worth a listen before training.")
    if digital:
        hint += f" {_plural(digital, 'file')} {_verb(digital, 'contain')} only digital silence (all zeros)."
    table = _table(["Output file", "Average level"],
                   [[r["path"], _fmt_db(_rms(r))] for r in quiet[:TOP_N]],
                   more=len(quiet) - TOP_N, play=[r["path"] for r in quiet[:TOP_N]])
    return Section(
        "Silent output",
        f"<p>{_badge(len(quiet), len(run.manifest), 'output files')} have an average level below "
        f"{QUIET_DB:g} dBFS.</p><p class='hint'>{hint}</p>{table}",
        [("warn", f"{_plural(len(quiet), 'output file')} {_verb(len(quiet), 'are')} nearly silent.")])


def _dropped(run: Run) -> Section:
    if not run.dropped:
        return Section("Dropped files", "<p>No files were dropped.</p>",
                       [("ok", "No files were dropped.")])
    groups = Counter((r.get("step") or "—", _reason_kind(r.get("reason", "")))
                     for r in run.dropped)
    summary = _table(["Step", "Reason", "Files"],
                     [[step, reason, f"{n:,}"] for (step, reason), n in groups.most_common()])
    listing = _table(["File", "Step", "Reason"],
                     [[_dropped_name(r), r.get("step", ""), r.get("reason", "")]
                      for r in run.dropped[:TOP_N]], more=len(run.dropped) - TOP_N,
                     more_file="dropped.csv")
    by_kind = Counter(reason for (_, reason) in groups.elements())
    text = ", ".join(f"{n:,} {k.lower()}" for k, n in by_kind.most_common(4))
    failed = sum(1 for r in run.dropped
                 if "failed" in r.get("reason", "") or r.get("step") == "Read files")
    findings = [("info", f"{_plural(len(run.dropped), 'file')} dropped: {text}.")]
    if failed:
        findings.insert(0, ("warn", f"{_plural(failed, 'file')} could not be read or made a "
                                    "step fail. See Dropped files."))
    return Section("Dropped files", summary + "<h3>Files</h3>" + listing, findings)


# ================================================================ 第二层：开了才有

def _validation(run: Run) -> Section:
    row = _step_row(run, "validate")
    rows = [r for r in run.dropped if r.get("step") == run.label("validate")]
    kinds = Counter(_reason_kind(r.get("reason", "")) for r in rows)
    checked = _int(row.get("input")) if row else 0
    body = _cards([("Checked", f"{checked:,}"), ("Passed", f"{checked - len(rows):,}")]
                  + [(k, f"{n:,}") for k, n in kinds.most_common()])
    if rows:
        body += _table(["File", "Problem"], [[_dropped_name(r), r.get("reason", "")]
                                             for r in rows[:TOP_N]], more=len(rows) - TOP_N,
                       more_file="dropped.csv")
        findings = [("info", f"File validation dropped {_plural(len(rows), 'file')}.")]
    else:
        body += "<p>Every file passed: none were empty, corrupted or silent.</p>"
        findings = [("ok", "Every file passed validation.")]
    return Section(run.label("validate"), body, findings)


def _clipping(run: Run) -> Section:
    units = run.units("clipping_ratio_pct", "clipping_check")
    tiers = defaultdict(list)
    for r in units:
        tiers[_clip_tier(r)].append(r)
    body = _cards([("Clean", f"{len(tiers['clean']):,}"), ("Minor", f"{len(tiers['minor']):,}"),
                   ("Noticeable", f"{len(tiers['noticeable']):,}"),
                   ("Severe", f"{len(tiers['severe']):,}")])
    body += (f"<p class='hint'><b>Minor</b>: the waveform only touched the maximum briefly "
             f"(at most {NOTICEABLE_RUN - 1} samples in a row, under {NOTICEABLE_PCT:g}% of the "
             f"file), usually inaudible. <b>Noticeable</b>: flat tops of {NOTICEABLE_RUN}+ "
             f"samples, or {NOTICEABLE_PCT:g}%+ of the file; may sound slightly harsh. "
             f"<b>Severe</b>: {SEVERE_PCT:g}%+ of the file, flat tops of {SEVERE_RUN}+ samples, "
             f"or both of the above; usually audible distortion. Clipping cannot be undone.</p>")
    shown = (sorted(tiers["severe"], key=lambda r: -_num(r["clipping_ratio_pct"]))
             + sorted(tiers["noticeable"], key=lambda r: -_num(r["longest_clip_run"])))
    if shown:
        body += _table(["File", "Level", "Clipped (%)", "Longest run (samples)"],
                       [[r["source"], _clip_tier(r).capitalize(), _fmt_pct(r["clipping_ratio_pct"]),
                         r.get("longest_clip_run", "")] for r in shown[:TOP_N]],
                       more=len(shown) - TOP_N, play=[run.play_path(r) for r in shown[:TOP_N]])
    dropped = sum(1 for r in run.dropped if r.get("step") == run.label("clipping_check"))
    if dropped:
        body += f"<p>{_plural(dropped, 'file')} above the allowed ratio were dropped.</p>"

    findings = []
    if tiers["severe"]:
        kept = sum(1 for r in tiers["severe"] if r.get("path") or r["source"] in run.first_path)
        findings.append(("warn", f"{_plural(len(tiers['severe']), 'file')} {_verb(len(tiers['severe']), 'are')} severely clipped"
                                 + (f" ({kept} still in the dataset)." if kept else
                                    ", all dropped.")))
    if tiers["noticeable"]:
        findings.append(("info", f"{_plural(len(tiers['noticeable']), 'file')} {_verb(len(tiers['noticeable']), 'have')} noticeable "
                                 "clipping."))
    if not tiers["severe"] and not tiers["noticeable"]:
        findings.append(("ok", "No audible clipping."))
    return Section(run.label("clipping_check"), body, findings)


def _dc_offset(run: Run) -> Section:
    units = run.units("dc_offset", "dc_offset")
    high = sorted((r for r in units if _num(r["dc_offset"]) > 0.01),
                  key=lambda r: -_num(r["dc_offset"]))
    removed = sum(1 for r in units if str(r.get("dc_offset_removed", "")).lower() == "true")
    body = _cards([("Checked", f"{len(units):,}"), ("Offset above 0.01", f"{len(high):,}"),
                   ("Removed", f"{removed:,}"),
                   ("Largest", f"{max((_num(r['dc_offset']) for r in units), default=0):.4f}")])
    body += ("<p class='hint'>The average of the waveform should be close to 0. An offset above "
             "0.01 usually points to a recording-hardware problem.</p>")
    if high:
        body += _table(["File", "Offset", "Removed"],
                       [[r["source"], r["dc_offset"],
                         "Yes" if str(r.get("dc_offset_removed", "")).lower() == "true" else "No"]
                        for r in high[:TOP_N]], more=len(high) - TOP_N)
    left = sum(1 for r in high if str(r.get("dc_offset_removed", "")).lower() != "true")
    findings = ([("warn", f"{_plural(left, 'file')} {_verb(left, 'have')} a DC offset above 0.01 that was not "
                          "removed.")] if left else
                [("ok", "No DC offset left in the output.")])
    return Section(run.label("dc_offset"), body, findings)


def _level_change(run: Run, key: str, column: str, what: str) -> Section:
    units = run.units(column, key)
    changes = [_num(r[column]) for r in units]
    big = sorted((r for r in units if _num(r[column]) < -BIG_CHANGE_DB),
                 key=lambda r: _num(r[column]))
    body = _cards([("Files", f"{len(units):,}"),
                   ("Median change", f"{statistics.median(changes):+.1f} dB" if changes else "—"),
                   ("Largest drop", f"{min(changes):+.1f} dB" if changes else "—"),
                   (f"Dropped more than {BIG_CHANGE_DB:g} dB", f"{len(big):,}")])
    body += (f"<p class='hint'>How much quieter each file became. A small change means only "
             f"{what} was removed; a large drop means a lot of the file was removed, so listen "
             "to check that the sound you care about is still there.</p>")
    if changes:
        body += _hist([(changes, "", INK)], "Level change (dB)")
    worst = sorted(units, key=lambda r: _num(r[column]))[:10]
    if worst and _num(worst[0][column]) < -1:
        body += "<h3>Largest changes</h3>" + _table(
            ["File", "Change (dB)"], [[r["source"], f"{_num(r[column]):+.1f}"] for r in worst],
            play=[run.play_path(r) for r in worst])
    findings = ([("warn", f"{run.label(key)} lowered {_plural(len(big), 'file')} by more than "
                          f"{BIG_CHANGE_DB:g} dB; worth a listen.")] if big else [])
    return Section(run.label(key), body, findings)


def _highpass(run: Run) -> Section:
    cutoff = run.params("highpass").get("cutoff_hz", "")
    return _level_change(run, "highpass", "highpass_change_db",
                         f"low-frequency rumble below {cutoff:g} Hz" if cutoff else "rumble")


def _denoise(run: Run) -> Section:
    return _level_change(run, "denoise", "denoise_change_db", "background noise")


def _silence_removal(run: Run) -> Section:
    units = run.units("silence_removed_sec", "silence_removal")
    removed = [_num(r["silence_removed_sec"]) for r in units]
    # 剪掉的部分占原文件的比例
    share = {id(r): _num(r["silence_removed_sec"]) / max(_num(r.get("orig_duration_sec")), 1e-9)
             for r in units}
    trimmed = [r for r in units if _num(r["silence_removed_sec"]) > 0]
    half = [r for r in units if share[id(r)] > 0.5]
    dropped = sum(1 for r in run.dropped if r.get("step") == run.label("silence_removal"))
    mode = run.params("silence_removal").get("mode", "trim")
    body = _cards([("Files trimmed", f"{len(trimmed):,} / {len(units):,}"),
                   ("Removed in total", _fmt_duration(sum(removed))),
                   ("More than half removed", f"{len(half):,}"),
                   ("All silence, dropped", f"{dropped:,}")])
    body += ("<p class='hint'>" + ("Only silence at the start and end was removed."
                                   if mode == "trim" else
                                   "Silence at the start, the end and long pauses in between "
                                   "was removed.") + "</p>")
    most = sorted(trimmed, key=lambda r: -share[id(r)])[:10]
    if most:
        body += "<h3>Most trimmed</h3>" + _table(
            ["File", "Removed", "Share of the file"],
            [[r["source"], _fmt_seconds(_num(r["silence_removed_sec"])),
              f"{100 * share[id(r)]:.0f}%"] for r in most],
            play=[run.play_path(r) for r in most])
    findings = ([("info", f"Silence removal trimmed {_fmt_duration(sum(removed))} in total.")]
                if sum(removed) > 0 else [])
    if half:
        findings.insert(0, ("warn", f"Silence removal cut more than half of "
                                    f"{_plural(len(half), 'file')}; check the threshold."))
    return Section(run.label("silence_removal"), body, findings)


def _duration_filter(run: Run) -> Section:
    p = run.params("duration_filter")
    lo, hi = _num(p.get("min_sec")), _num(p.get("max_sec"))
    units = run.units("duration_checked_sec", "duration_filter")
    values = [_num(r["duration_checked_sec"]) for r in units]
    reasons = Counter(_reason_kind(r.get("reason", "")) for r in run.dropped
                      if r.get("step") == run.label("duration_filter"))
    short, long = reasons.get("Too short", 0), reasons.get("Too long", 0)
    rule = f"Keep files from {lo:g} s" + (f" to {hi:g} s" if hi > 0 else " (no maximum)")
    body = _cards([("Checked", f"{len(units):,}"), ("Too short", f"{short:,}"),
                   ("Too long", f"{long:,}"), ("Kept", f"{len(units) - short - long:,}")])
    body += f"<p class='hint'>{rule}. The dashed lines mark the limits.</p>"
    if values:
        lines = [(lo, "min")] + ([(hi, "max")] if hi > 0 else [])
        body += _hist([(values, "", INK)], "Duration when checked (s)", lines=lines)
    share = (short + long) / max(len(units), 1)
    findings = []   # 丢了多少已经在 Dropped files 里说了，这里只在丢得太多时提醒
    if share > 0.2:
        findings.append(("warn", f"Duration filtering dropped {100 * share:.0f}% of the files; "
                                 "check the limits."))
    return Section(run.label("duration_filter"), body, findings)


def _normalization(run: Run) -> Section:
    p = run.params("normalize")
    method = p.get("method", "lufs")
    unit = "LUFS" if method == "lufs" else "dBFS"
    target = _num(p.get({"peak": "target_peak_db", "rms": "target_rms_db"}.get(method,
                                                                             "target_lufs")))
    units = run.units("normalize_gain_db", "normalize")
    skipped = run.units("normalize_skipped", "normalize")
    measured = [r for r in units if r.get("normalize_before") not in (None, "")]
    short = sorted((r for r in units if _num(r.get("normalize_shortfall_db")) >= NORM_TOLERANCE_DB),
                   key=lambda r: -_num(r["normalize_shortfall_db"]))
    reached = len(units) - len(short)
    body = _cards([("Method", {"lufs": "Loudness", "rms": "RMS", "peak": "Peak"}[method]),
                   ("Target", f"{target:g} {unit}"),
                   ("Reached target", f"{reached:,} / {len(units) + len(skipped):,}"),
                   ("Below target", f"{len(short):,}"), ("Not measured", f"{len(skipped):,}")])
    if measured:
        before = [_num(r["normalize_before"]) for r in measured]
        after = [_num(r["normalize_after"]) for r in measured]
        body += _hist([(before, "Before", LIGHT), (after, "After", INK)], unit,
                      lines=[(target, "target")])
        body += ("<p class='hint'>A wide spread before and a narrow peak at the target after "
                 "means normalization worked.</p>")
    if short:
        ceiling = _num(p.get("peak_ceiling_db", -1))
        body += (f"<h3>Below target</h3><p class='hint'>These files have sharp peaks compared "
                 f"to their average level (hits, knocks, clicks). Reaching the target would push "
                 f"the peaks above {ceiling:g} dBFS, so less gain was applied and they end up "
                 f"quieter than the rest. Differences under {NORM_TOLERANCE_DB:g} dB are "
                 "ignored.</p>")
        body += _table(["File", "Below target (dB)", "Gain applied (dB)"],
                       [[r["source"], f"{_num(r['normalize_shortfall_db']):.1f}",
                         f"{_num(r['normalize_gain_db']):+.1f}"] for r in short[:TOP_N]],
                       more=len(short) - TOP_N, play=[run.play_path(r) for r in short[:TOP_N]])
    if skipped:
        reasons = Counter(r["normalize_skipped"] for r in skipped)
        body += ("<h3>Not measured</h3><p>Left unchanged: "
                 + ", ".join(f"{n} {html.escape(k)}" for k, n in reasons.items()) + ".</p>")

    findings = []
    if short:
        findings.append(("warn", f"{_plural(len(short), 'file')} ended up more than "
                                 f"{NORM_TOLERANCE_DB:g} dB below the normalization target."))
    if skipped:
        findings.append(("warn", f"{_plural(len(skipped), 'file')} could not be normalized "
                                 "(too short or too quiet to measure) and "
                                 f"{_verb(len(skipped), 'were')} left unchanged."))
    if not findings:
        findings.append(("ok", f"Every file was normalized to {target:g} {unit}."))
    return Section(run.label("normalize"), body, findings)


def _segmentation(run: Run) -> Section:
    p = run.params("segment")
    row = _step_row(run, "segment")
    segs = [r for r in run.manifest if r.get("segment_index", "") != ""]
    sources = {r["source"] for r in segs}
    padded = [r for r in segs if _num(r.get("segment_padded_sec")) > 0]
    padding = sum(_num(r.get("segment_padded_sec")) for r in padded)
    mostly = sorted((r for r in padded
                     if _num(r["segment_padded_sec"]) > 0.5 * _num(r.get("duration_sec"))),
                    key=lambda r: -_num(r["segment_padded_sec"]))
    body = _cards([("Segments", f"{len(segs):,}"), ("From files", f"{len(sources):,}"),
                   ("Per file (average)", f"{len(segs) / max(len(sources), 1):.1f}"),
                   ("Silence padding added", _fmt_duration(padding)),
                   ("Dropped", f"{_int(row.get('dropped')) if row else 0:,}")])
    if p.get("mode", "fixed") == "fixed":
        length = _num(p.get("segment_sec"))
        last = {"pad": "padded with silence to the full length", "keep": "kept at its own length",
                "drop": "dropped"}.get(p.get("last", "pad"), "")
        shorter = sum(1 for r in run.sources if _num(r.get("orig_duration_sec")) < length)
        text = f"Fixed {length:g} s segments"
        if _num(p.get("overlap_sec")) > 0:
            text += f" with {_num(p['overlap_sec']):g} s overlap"
        text += f". The last, shorter segment of each file is {last}."
        if shorter:
            text += f" {_plural(shorter, 'file')} {_verb(shorter, 'were')} shorter than one segment."
        body += f"<p class='hint'>{text}</p>"
    else:
        body += "<p class='hint'>Split at pauses. Segment lengths are shown under Duration.</p>"
    counts = list(Counter(r["source"] for r in segs).values())
    if counts and max(counts) > 1:
        body += _hist([(counts, "", INK)], "Segments per file", integer=True)
    if mostly:
        body += ("<h3>Mostly padding</h3><p class='hint'>More than half of these segments is "
                 "added silence. A model may learn little from them; consider "
                 "<b>keep</b> or <b>drop</b> for the last segment.</p>")
        body += _table(["Output file", "Sound", "Padding"],
                       [[r["path"], _fmt_seconds(_num(r["duration_sec"])
                                                 - _num(r["segment_padded_sec"])),
                         _fmt_seconds(_num(r["segment_padded_sec"]))] for r in mostly[:TOP_N]],
                       more=len(mostly) - TOP_N, play=[r["path"] for r in mostly[:TOP_N]])
    findings = ([("warn", f"{_plural(len(mostly), 'segment')} {_verb(len(mostly), 'are')} more than half silence "
                          "padding.")] if mostly else [])
    return Section(run.label("segment"), body, findings)


def _features(run: Run) -> Section:
    p = run.params("features")
    hop = _num(p.get("hop_ms", 10))
    shapes = []
    if p.get("rms", True):
        shapes.append(("RMS energy", "1"))
    if p.get("spectral", True):
        shapes.append(("Spectral centroid, bandwidth, rolloff, flatness, ZCR", "1 each"))
    if p.get("log_mel"):
        shapes.append(("Log-mel spectrogram", str(_int(p.get("n_mels", 80)))))
    if p.get("mfcc"):
        shapes.append(("MFCC", str(_int(p.get("n_mfcc", 13)))))
    files = sum(1 for r in run.manifest if r.get("features_path"))
    body = _cards([("Feature files", f"{files:,}"), ("Window", f"{_num(p.get('win_ms', 25)):g} ms"),
                   ("Hop", f"{hop:g} ms"),
                   ("Frames per second", f"{1000 / hop:g}" if hop else "—")])
    body += _table(["Feature", "Values per frame"], shapes)
    body += ("<p class='hint'>One .npz file per output file in <code>features/</code>, with "
             "the same folder layout as the dataset. Each array has the shape (values per "
             "frame, frames).</p>")
    return Section(run.label("features"), body)


def _scripts(run: Run) -> Section:
    rows, findings = [], []
    for key, row in run.workflow:
        if key != "script":
            continue
        name = row["name"].split(":", 1)[-1].strip()
        failed = [r for r in run.dropped
                  if r.get("step") == row["name"] and "failed" in r.get("reason", "")]
        delta = _num(row.get("output_sec")) - _num(row.get("input_sec"))
        rows.append([name, row.get("input", ""), row.get("output", ""), row.get("dropped", ""),
                     _signed_duration(delta) if abs(delta) >= 0.5 else "—",
                     f"{len(failed):,}"])
        if failed:
            findings.append(("warn", f"{name} failed on {_plural(len(failed), 'file')}, e.g. "
                                     f"{_short_reason(failed[0].get('reason', ''))}"))
    body = _table(["Script", "In", "Out", "Dropped", "Duration change", "Errors"], rows)
    body += ("<p class='hint'>A copy of every script is in <code>scripts/</code>. Errors are "
             "listed in dropped.csv.</p>")
    return Section("Python scripts", body, findings)


DETAIL_SECTIONS = {
    "validate": _validation,
    "clipping_check": _clipping,
    "dc_offset": _dc_offset,
    "highpass": _highpass,
    "denoise": _denoise,
    "silence_removal": _silence_removal,
    "duration_filter": _duration_filter,
    "normalize": _normalization,
    "segment": _segmentation,
    "features": _features,
    # 重采样、声道转换、位深转换：结果在 Format 里
}


def _workflow(run: Run) -> Section:
    if not run.step_rows:
        return Section("Workflow", "<p>No processing record found.</p>")
    rows = []
    for r in run.step_rows:
        params = r.get("parameters", "")
        if r.get("name", "").startswith("Python script"):
            params = Path(params).name   # 只写文件名，不把电脑上的路径带进报告
        rows.append([r.get("step", ""), r.get("name", ""), params.replace("; ", "\n"),
                     r.get("input", ""), r.get("output", ""), r.get("dropped", "")])
    return Section("Workflow", _table(["#", "Step", "Settings", "In", "Out", "Dropped"], rows,
                                      cls="workflow"))


# ================================================================ 网页和图表

def _render(section: Section, cls: str = "") -> str:
    sub = f" <span class='muted'>{html.escape(section.subtitle)}</span>" if section.subtitle else ""
    return (f"<section class='{cls}'><h2>{html.escape(section.title)}{sub}</h2>"
            f"{section.body}</section>")


def _cards(items) -> str:
    return "<div class='stats'>" + "".join(
        f'<div class="stat"><div class="value">{html.escape(str(v))}</div>'
        f'<div class="label">{html.escape(k)}</div></div>' for k, v in items) + "</div>"


def _badge(n: int, total: int, unit: str) -> str:
    cls = "badge warn" if n else "badge ok"
    return f"<span class='{cls}'>{n:,} / {total:,} {unit}</span>"


def _table(headers, rows, more: int = 0, play=None, cls: str = "",
           more_file: str = "manifest.csv") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    if play is not None:
        head = "<th class='play-col'></th>" + head
    body = []
    for i, row in enumerate(rows):
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        if play is not None:
            src = play[i] if i < len(play) else None
            button = (f"<button class='play' data-src='{html.escape(quote(src))}' "
                      f"title='Play'></button>" if src else "")
            cells = f"<td class='play-col'>{button}</td>" + cells
        body.append(f"<tr>{cells}</tr>")
    extra = (f"<p class='hint'>…and {more:,} more. See {more_file} for the full list.</p>"
             if more > 0 else "")
    return (f"<table class='{cls}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>{extra}")


def _hist(series, xlabel, title=None, lines=(), integer=False) -> str:
    """直方图。series：[(数值, 图例, 颜色)]，多组时画在同一张图上方便对比。"""
    series = [(np.asarray([v for v in values if math.isfinite(v)], dtype=float), label, color)
              for values, label, color in series]
    series = [s for s in series if s[0].size]
    if not series:
        return ""
    allv = np.concatenate([s[0] for s in series] + [np.asarray([x for x, _ in lines], float)])
    lo, hi = float(allv.min()), float(allv.max())
    if integer:
        edges = np.arange(math.floor(lo), math.ceil(hi) + 2) - 0.5
    else:
        n = max(len(s[0]) for s in series)
        edges = np.linspace(lo, hi if hi > lo else lo + 1, min(40, max(6, n // 5)) + 1)
    fig, ax = plt.subplots(figsize=(8, 2.6) if title is None else (4.6, 2.6))
    for values, label, color in series:
        ax.hist(values, bins=edges, color=color, edgecolor="white", linewidth=0.5,
                label=label or None, alpha=0.95)
    for x, label in lines:
        ax.axvline(x, color="#1d1d1f", linestyle="--", linewidth=1)
        ax.annotate(f" {label}", (x, 1), xycoords=("data", "axes fraction"), va="top",
                    fontsize=8.5, color="#1d1d1f")
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color="#1d1d1f")
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Files")
    _tidy(ax)
    return _img(fig)


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d2d2d7")
    ax.tick_params(colors="#6e6e73", labelsize=8.5)
    ax.xaxis.label.set_color("#6e6e73")
    ax.yaxis.label.set_color("#6e6e73")
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _img(fig) -> str:
    """把图转成 base64 嵌进网页里，这样 report.html 一个文件就能看，不依赖图片文件。"""
    buf = io.BytesIO()
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 缺字体时的警告
        fig.savefig(buf, format="png", dpi=144)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="">'


# ================================================================ 小工具

def _step_row(run: Run, key: str) -> dict | None:
    return next((row for k, row in run.workflow if k == key), None)


def _clip_tier(row: dict) -> str:
    ratio, run_len = _num(row.get("clipping_ratio_pct")), _num(row.get("longest_clip_run"))
    if ratio <= 0:
        return "clean"
    if (ratio >= SEVERE_PCT or run_len >= SEVERE_RUN
            or (ratio >= NOTICEABLE_PCT and run_len >= NOTICEABLE_RUN)):
        return "severe"
    return "noticeable" if ratio >= NOTICEABLE_PCT or run_len >= NOTICEABLE_RUN else "minor"


def _reason_kind(reason: str) -> str:
    """原因的类别：括号或冒号之前的部分，比如 "Too short (0.30s < 0.5s)" → "Too short"。"""
    for sep in ("(", ":"):
        reason = reason.split(sep)[0]
    return reason.strip() or "(not specified)"


def _short_reason(reason: str) -> str:
    text = reason.split("failed:", 1)[-1].strip()
    return text if len(text) <= 80 else text[:77] + "…"


def _dropped_name(row: dict) -> str:
    seg = row.get("segment", "")
    return f"{row['source']} ({seg})" if seg else row["source"]


def _channel_name(n) -> str:
    return {"1": "Mono", "2": "Stereo"}.get(str(n), f"{n} channels")


def _bit_depth(subtype: str) -> str:
    return {"PCM_16": "16-bit", "PCM_24": "24-bit", "PCM_32": "32-bit", "PCM_S8": "8-bit",
            "PCM_U8": "8-bit", "FLOAT": "32-bit float", "DOUBLE": "64-bit float",
            "MPEG_LAYER_III": "MP3", "VORBIS": "Vorbis", "OPUS": "Opus"}.get(subtype, subtype)


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


def _verb(n: int, plural_verb: str) -> str:
    """单复数动词：1 file is / 2 files are。"""
    return {"are": "is", "have": "has", "were": "was", "contain": "contains"}[plural_verb] \
        if n == 1 else plural_verb


def _spread(values):
    return (min(values), statistics.median(values), max(values)) if values else (0, 0, 0)


def _fmt_duration(seconds: float) -> str:
    seconds = abs(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _signed_duration(seconds: float) -> str:
    return ("+" if seconds >= 0 else "−") + _fmt_duration(seconds)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def _fmt_db(value: float) -> str:
    return "silent" if value < SILENT_DB else f"{value:.1f} dBFS"


def _fmt_pct(value) -> str:
    v = _num(value)
    return f"{v:.3f}" if v < 0.01 else f"{v:.2f}"


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _int(v) -> int:
    return int(_num(v))


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


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# 点击播放：同一时间只播一个，再点一下停止
_PLAYER_JS = """
let current = null, currentButton = null;
document.addEventListener('click', (e) => {
  const button = e.target.closest('.play');
  if (!button) return;
  const same = button === currentButton;
  if (current) { current.pause(); currentButton.classList.remove('on'); }
  current = null; currentButton = null;
  if (same) return;
  const audio = new Audio(button.dataset.src);
  current = audio; currentButton = button;
  button.classList.add('on');
  audio.onended = () => button.classList.remove('on');
  audio.play().catch(() => { button.classList.remove('on'); button.classList.add('missing');
                             button.title = 'This file could not be played here'; });
});
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quality Report · {title}</title>
<style>
  :root {{ --ink: #1d1d1f; --muted: #6e6e73; --line: #e5e5ea; --bg: #f5f5f7; --card: #ffffff;
           --warn: #b25000; --warn-bg: #fff3e0; --ok: #1f7a3a; --ok-bg: #e9f7ee; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial,
         sans-serif; max-width: 980px; margin: 0 auto; padding: 48px 16px 72px;
         color: var(--ink); background: var(--bg); line-height: 1.5; font-size: 14px;
         -webkit-font-smoothing: antialiased; }}
  header {{ margin-bottom: 12px; }}
  .eyebrow {{ color: var(--muted); font-weight: 600; font-size: 13px; }}
  h1 {{ margin: 2px 0 4px; font-size: 32px; font-weight: 700; letter-spacing: -0.02em;
        overflow-wrap: anywhere; }}
  .time, .muted {{ color: var(--muted); font-weight: 400; }}
  section {{ background: var(--card); border-radius: 14px; padding: 22px 26px; margin-top: 16px; }}
  h2 {{ margin: 0 0 10px; font-size: 19px; font-weight: 600; letter-spacing: -0.01em; }}
  h2.part {{ margin: 40px 4px 0; font-size: 24px; letter-spacing: -0.02em; }}
  .part-hint {{ margin: 2px 4px 0; }}
  h3 {{ font-size: 14px; font-weight: 600; margin: 20px 0 6px; }}
  p {{ margin: 8px 0; }}
  .hint {{ color: var(--muted); }}
  code {{ font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; }}
  .verdict {{ font-size: 16px; font-weight: 600; margin: 0 0 8px; }}
  .verdict.warn {{ color: var(--warn); }}
  .verdict.ok {{ color: var(--ok); }}
  .findings {{ list-style: none; padding: 0; margin: 0; }}
  .findings li {{ padding: 6px 0 6px 20px; position: relative; border-top: 1px solid var(--line); }}
  .findings li:first-child {{ border-top: none; }}
  .findings li::before {{ content: ""; position: absolute; left: 2px; top: 13px; width: 8px;
                         height: 8px; border-radius: 50%; background: #c7c7cc; }}
  .findings li.warn::before {{ background: #ff9500; }}
  .findings li.ok::before {{ background: #34c759; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;
           margin-bottom: 8px; }}
  .stat {{ background: var(--bg); border-radius: 12px; padding: 14px 16px; min-width: 0; }}
  .stat .value {{ font-size: 22px; font-weight: 600; letter-spacing: -0.02em;
                 overflow-wrap: anywhere; }}
  .stat .label {{ color: var(--muted); font-size: 12px; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: center; }}
  .callout {{ background: var(--bg); border-radius: 12px; padding: 24px; text-align: center; }}
  .callout .value {{ font-size: 28px; font-weight: 600; }}
  .callout .label {{ color: var(--muted); font-size: 12px; }}
  .badge {{ font-size: 12px; font-weight: 600; border-radius: 999px; padding: 2px 10px;
           margin-right: 4px; white-space: nowrap; }}
  .badge.warn {{ background: var(--warn-bg); color: var(--warn); }}
  .badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
  table {{ border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 7px 12px 7px 0; text-align: left;
           overflow-wrap: anywhere; vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  table.compare td:first-child {{ color: var(--muted); width: 22%; }}
  table.compare td.arrow {{ color: var(--muted); width: 32px; }}
  table.workflow td:nth-child(3) {{ white-space: pre-line; color: var(--muted); }}
  .play-col {{ width: 30px; padding-right: 6px; }}
  .play {{ width: 22px; height: 22px; border-radius: 50%; border: none; cursor: pointer;
          background: var(--bg); position: relative; vertical-align: middle; }}
  .play:hover {{ background: #e5e5ea; }}
  .play::before {{ content: ""; position: absolute; left: 8px; top: 6px; border-style: solid;
                  border-width: 5px 0 5px 8px; border-color: transparent transparent transparent
                  var(--ink); }}
  .play.on {{ background: var(--ink); }}
  .play.on::before {{ left: 7px; top: 7px; width: 8px; height: 8px; border: none;
                     background: #fff; border-radius: 1px; }}
  .play.missing {{ opacity: 0.35; }}
  img {{ max-width: 100%; height: auto; margin-top: 8px; display: block; }}
  @media (max-width: 640px) {{
    section {{ padding: 18px 16px; }}
    .pair {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">TidyAudio {version} · Quality Report</div>
  <h1>{title}</h1>
  <div class="time">{time}</div>
</header>
{body}
<script>{script}</script>
</body>
</html>
"""
