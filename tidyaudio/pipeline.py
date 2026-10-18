"""第四步：pipeline —— 把所有功能组装起来，真正读文件、处理、输出。

流程：
  1. 扫描输入文件夹
  2. 按"推荐顺序"排好你勾选的功能
  3. 每个文件：读取 → 依次经过每个功能 → 保存
  3.5 自定义脚本：first 在所有内置功能之前跑，last 在内置处理之后、位深和特征提取之前跑
  4. 如果数据集自带 CSV（比如 file_list.csv），输出一份更新后的
  5. 写出 manifest.csv（留下的）、dropped.csv（被丢掉的）、
     processing_steps.csv（做了哪些处理）、config.yaml（这次的设置）

输出文件夹的结构：
  xxx_output/
  ├── dataset/        处理后的数据集本身，可以直接替换原来的文件夹：
  │   ├── alarm/ ...      处理后的音频，保持和输入一样的子文件夹结构
  │   ├── file_list.csv   更新后的数据集 CSV（原数据集有 CSV 才有）
  │   ├── summary.csv     按类别重新统计的数量和总时长（能找到类别列才有）
  │   └── README.txt      这个数据集是怎么处理出来的
  ├── features/       特征文件（勾选了特征提取才有），每条音频一个 .npz
  ├── manifest.csv    每条输出音频一行：路径、时长、采样率、各项检查结果……
  ├── dropped.csv     被丢掉的文件和原因
  ├── processing_steps.csv  做了哪些处理：顺序、参数、每一步处理/丢掉了多少条
  ├── report.html     质量报告，用浏览器打开
  ├── scripts/        用到的自定义脚本的副本（用了才有），以后脚本改了也能找到当时的版本
  └── config.yaml     这次用的全部设置，可以直接加载、重新跑一遍
"""

import csv
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from . import __version__
from .audio_io import Clip, load_audio, save_audio
from .custom_script import CustomScript, make_script
from .dataset_csv import CsvResult, find_dataset_csv, read_dataset_csv, write_updated_csv
from .report import write_report
from .scan import scan_folder
from .steps import REGISTRY, Step
from .steps.checks import to_db
from .steps.level import peak, rms

# 推荐的处理顺序。不管你勾选的顺序如何，都按这个顺序执行。
STEP_ORDER = [
    "validate",          # 先把坏文件挑出来
    "clipping_check",    # 必须在归一化之前：归一化后峰值都被压到 -1 dBFS，就检测不到原始削波了
    "dc_offset",         # 在归一化之前：偏移会让峰值"虚高"
    "resample",
    "channels",
    "highpass",
    "denoise",
    "silence_removal",
    "duration_filter",   # 在去静音之后：判断的是"去掉静音后"的真实时长
    "normalize",         # 在切片之前：整条统一音量后再切，保留一条音频内部的强弱变化
    "segment",
    "bit_depth",         # 只决定保存格式，放哪都行
    "features",          # 最后：提取的是处理完的音频的特征
]
# 位置为 last 的自定义脚本，插在这些功能之前（让特征提取的是脚本处理后的音频）
TAIL_STEPS = {"bit_depth", "features"}
SCRIPT_POSITIONS = ["first", "last"]


def default_order(builtin_keys: list[str], scripts: list[tuple[str, str]]) -> list[str]:
    """推荐顺序。builtin_keys：打开的内置功能；scripts：[(脚本的 key, "first"/"last"), ...]。

    first 的脚本在最前面；内置功能按 STEP_ORDER；last 的脚本在内置处理之后、
    位深转换和特征提取之前（让特征提取的是脚本处理后的音频）。
    """
    rank = {k: i for i, k in enumerate(STEP_ORDER)}
    builtin = sorted(builtin_keys, key=lambda k: (rank.get(k, len(rank)), k))
    first = [k for k, pos in scripts if pos == "first"]
    last = [k for k, pos in scripts if pos != "first"]
    main = [k for k in builtin if k not in TAIL_STEPS]
    tail = [k for k in builtin if k in TAIL_STEPS]
    return first + main + last + tail


def merge_order(default: list[str], custom: list[str]) -> list[str]:
    """用户调整过的顺序 + 后来新加/去掉的功能。

    - custom 里有、现在已经没打开的：去掉
    - 现在打开了、custom 里没有的：插在"推荐顺序里排在它前面的那些步骤"中、
      在当前顺序里最靠后的那一个后面（这样它不会跑到本该在它前面的步骤之前）
    """
    result = [k for k in custom if k in default]
    for i, key in enumerate(default):
        if key in result:
            continue
        before = [result.index(k) for k in default[:i] if k in result]
        result.insert(max(before) + 1 if before else 0, key)
    return result


def order_warnings(keys: list[str], names: dict[str, str] | None = None) -> list[str]:
    """检查顺序有没有常见的问题。只提醒，不阻止：最终由用户决定。

    names：脚本等非内置步骤显示用的名字，比如 {"script_0": "my_script.py"}。
    """
    names = names or {}
    pos = {k: i for i, k in enumerate(keys)}
    label = lambda k: names.get(k) or (REGISTRY[k].label if k in REGISTRY else k)  # noqa: E731
    warnings = []

    def after(a, b):
        return a in pos and b in pos and pos[a] > pos[b]

    if after("clipping_check", "normalize"):
        warnings.append(f"{label('clipping_check')} runs after {label('normalize')}, so clipping "
                        "in the original audio can no longer be detected.")
    if after("dc_offset", "normalize"):
        warnings.append(f"{label('dc_offset')} runs after {label('normalize')}; an offset makes "
                        "peaks look higher, so the normalization gain may be off.")
    if after("silence_removal", "duration_filter"):
        warnings.append(f"{label('duration_filter')} runs before {label('silence_removal')}, so it "
                        "checks lengths that still include silence.")
    if after("normalize", "segment"):
        warnings.append(f"{label('normalize')} runs after {label('segment')}, so every segment is "
                        "normalized on its own instead of the whole file.")
    if "features" in pos:
        later = [k for k in keys[pos["features"] + 1:] if k != "bit_depth"]
        if later:
            warnings.append(f"{label('features')} is not the last step, so the features do not "
                            f"reflect: {', '.join(label(k) for k in later)}.")
    return warnings


@dataclass
class PipelineConfig:
    """一次处理的全部设置。可以存成 / 读取 config.yaml。"""

    input_dir: str
    output_dir: str | None = None        # None：自动用输入文件夹同级的 "xxx_output"
    output_format: str = "wav"           # "wav" / "flac" / "same"（保持原格式）
    recursive: bool = True               # 是否处理子文件夹
    steps: dict[str, dict] = field(default_factory=dict)   # {"resample": {"target_sr": 16000}, ...}
    # 数据集自带的 CSV："auto" 自动在输入文件夹里找；也可以写路径；None 表示不用
    dataset_csv: str | None = "auto"
    # 按哪一列统计 summary.csv："auto" 自动找 class / label 等列名；None 表示不统计
    label_column: str | None = "auto"
    # 自定义脚本：[{"path": "/xx/my_script.py", "position": "last"}, ...]，按列表顺序执行
    scripts: list[dict] = field(default_factory=list)
    # 用哪个 Python 运行自定义脚本：None 表示软件自己的 Python；
    # 也可以是别的环境的 python 路径，比如 "/opt/anaconda3/envs/torch/bin/python"
    script_python: str | None = None
    # 自定义执行顺序：每一步的 key 按顺序排（内置功能用 "resample" 这样的名字，
    # 脚本用 "script_0"、"script_1"……对应 scripts 列表的第几个）。None 表示推荐顺序
    order: list[str] | None = None

    def build_steps(self) -> list[Step]:
        """按推荐顺序，把勾选的功能和自定义脚本变成可以执行的 Step 对象。"""
        unknown = set(self.steps) - set(REGISTRY)
        if unknown:
            raise ValueError(f"Unknown steps: {sorted(unknown)}")
        order = STEP_ORDER + sorted(set(REGISTRY) - set(STEP_ORDER))
        builtin = [REGISTRY[key](**(self.steps[key] or {})) for key in order if key in self.steps]

        first, last = [], []
        try:
            for i, s in enumerate(self.scripts):
                position = s.get("position", "last")
                if position not in SCRIPT_POSITIONS:
                    raise ValueError(
                        f"Script position must be one of {SCRIPT_POSITIONS}, got {position!r}")
                script = make_script(s["path"], index=i, python=self.script_python)
                (first if position == "first" else last).append(script)
        except Exception:
            # 后面的脚本加载失败时，前面已经启动的外部 Python 进程也要关掉
            for script in first + last:
                script.close()
            raise

        steps = {s.key: s for s in builtin + first + last}
        keys = default_order([s.key for s in builtin],
                             [(s.key, "first") for s in first] + [(s.key, "last") for s in last])
        if self.order:
            unknown = [k for k in self.order if k not in steps and k not in REGISTRY
                       and not k.startswith("script_")]
            if unknown:
                for script in first + last:
                    script.close()
                raise ValueError(f"Unknown steps in order: {unknown}")
            keys = merge_order(keys, self.order)
        return [steps[k] for k in keys]

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        data = asdict(self)
        if extra:
            data["run_info"] = extra
        Path(path).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        data.pop("run_info", None)   # 上次运行的记录，不是设置
        return cls(**data)


def default_output_dir(input_dir: str | Path) -> Path:
    """输入文件夹同级的 "xxx_output"。

    如果已经存在且不是空的，就用 "xxx_output_2"、"xxx_output_3"……
    这样永远不会覆盖之前的处理结果。
    """
    input_dir = Path(input_dir).resolve()
    base = input_dir.parent / f"{input_dir.name}_output"
    candidate, n = base, 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate = base.with_name(f"{base.name}_{n}")
        n += 1
    return candidate


@dataclass
class RunSummary:
    """处理结果汇总。"""

    output_dir: Path
    input_count: int = 0       # 输入了多少个文件
    output_count: int = 0      # 输出了多少条音频（切片后可能比输入多）
    dropped_count: int = 0     # 丢掉了多少个文件 / 片段
    seconds: float = 0.0       # 用时
    cancelled: bool = False


# 进度回调：(已处理文件数, 总文件数, 当前文件名)。界面用它来更新进度条。
ProgressCallback = Callable[[int, int, str], None]


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RunSummary:
    """执行整个处理流程。

    progress：每处理完一个文件调用一次，用来更新进度
    should_stop：每个文件开始前检查一次，返回 True 就停止（界面上的"取消"按钮）
    """
    start_time = time.time()
    steps = config.build_steps()          # 先检查设置，有错马上报，不要处理到一半才发现
    try:
        return _run(config, steps, start_time, progress, should_stop)
    finally:
        # 不管正常结束、出错还是取消，都要把外部 Python 进程关掉
        for step in steps:
            if isinstance(step, CustomScript):
                step.close()


def _run(config: PipelineConfig, steps: list[Step], start_time: float,
         progress: ProgressCallback | None,
         should_stop: Callable[[], bool] | None) -> RunSummary:
    files = scan_folder(config.input_dir, recursive=config.recursive).files
    input_dir = Path(config.input_dir).resolve()
    out_dir = Path(config.output_dir).resolve() if config.output_dir else default_output_dir(input_dir)
    if out_dir == input_dir or input_dir in out_dir.parents:
        # 输出放在输入里面的话，下次运行会把上次的输出当成新的输入
        raise ValueError("The output folder cannot be the input folder or inside it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    summary = RunSummary(output_dir=out_dir, input_count=len(files))
    kept_rows: list[dict] = []
    # 每个功能的统计：处理了多少条、留下多少条、丢掉多少条
    step_stats = {s.key: {"in": 0, "out": 0, "dropped": 0} for s in steps}
    load_stats = {"in": 0, "out": 0, "dropped": 0}   # "读取文件"这一步
    dropped_rows: list[dict] = []
    used_paths: set[Path] = set()

    for i, path in enumerate(files):
        if should_stop and should_stop():
            summary.cancelled = True
            break
        rel = path.resolve().relative_to(input_dir).as_posix()

        # 1. 读取。打不开的文件记下来，继续处理下一个，不让整个程序崩掉
        load_stats["in"] += 1
        try:
            clip = load_audio(path)
            clip.rel_path = rel
            load_stats["out"] += 1
            # 记下原始文件的样子，质量报告里做"处理前 vs 处理后"的对比
            clip.info.update(
                orig_sample_rate=clip.sr,
                orig_channels=clip.channels,
                orig_duration_sec=round(clip.duration, 3),
            )
        except Exception as e:
            load_stats["dropped"] += 1
            dropped_rows.append({"source": rel, "reason": f"Could not read: {e}"})
            _report(progress, i + 1, len(files), rel)
            continue

        # 2. 依次经过每个功能
        clips, dropped = _apply_steps(steps, clip, step_stats)
        for c in dropped:
            dropped_rows.append({
                "source": rel,
                "segment": c.name if c.name != path.stem else "",
                "reason": c.info.get("dropped_reason", ""),
            })

        # 3. 保存
        for c in clips:
            audio_path = _unique_path(
                dataset_dir / Path(rel).parent / f"{c.name}.{_extension(config, path)}", used_paths)
            save_audio(c, audio_path)
            row = {
                "path": audio_path.relative_to(out_dir).as_posix(),
                "source": rel,
                "duration_sec": round(c.duration, 3),
                "sample_rate": c.sr,
                "channels": c.channels,
                # 输出音频的电平：峰值和平均能量（dBFS）
                "peak_dbfs": round(to_db(peak(c.audio)), 2),
                "rms_dbfs": round(to_db(rms(c.audio)), 2),
            }
            if c.features:
                feat_path = out_dir / "features" / audio_path.relative_to(dataset_dir)
                feat_path = feat_path.with_suffix(".npz")
                feat_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(feat_path, **c.features)
                row["features_path"] = feat_path.relative_to(out_dir).as_posix()
            row.update(c.info)
            kept_rows.append(row)

        _report(progress, i + 1, len(files), rel)

    summary.output_count = len(kept_rows)
    summary.dropped_count = len(dropped_rows)
    summary.seconds = round(time.time() - start_time, 2)

    # 数据集自带的 CSV：在原来的基础上更新
    csv_result = None
    csv_path = _resolve_dataset_csv(config.dataset_csv, input_dir)
    if csv_path:
        audio_rel_paths = [f.resolve().relative_to(input_dir).as_posix() for f in files]
        csv_table = read_dataset_csv(csv_path)
        csv_result = write_updated_csv(
            csv_table, audio_rel_paths, kept_rows, dataset_dir, config.label_column)

    scripts = [s for s in steps if isinstance(s, CustomScript)]
    _copy_scripts(scripts, out_dir / "scripts")
    step_rows = _step_rows(steps, step_stats, load_stats)
    _write_readme(dataset_dir / "README.txt", input_dir, summary, step_rows,
                  csv_result, csv_table.footer if csv_path else [], scripts)

    _write_csv(out_dir / "manifest.csv", kept_rows)
    _write_csv(out_dir / "dropped.csv", dropped_rows, default_fields=["source", "segment", "reason"])
    _write_csv(out_dir / "processing_steps.csv", step_rows,
               default_fields=STEP_FIELDS)
    # 保存时把默认值也一起写进去：以后就算改了某个默认值，
    # 用这份 config.yaml 重跑，结果也和这次完全一样
    resolved = PipelineConfig(
        input_dir=str(input_dir),
        output_dir=config.output_dir,
        output_format=config.output_format,
        recursive=config.recursive,
        steps={s.key: dict(s.values) for s in steps if not isinstance(s, CustomScript)},
        dataset_csv=config.dataset_csv,
        label_column=config.label_column,
        scripts=[{"path": str(Path(sc["path"]).expanduser().resolve()),
                  "position": sc.get("position", "last")} for sc in config.scripts],
        script_python=config.script_python,
        order=[s.key for s in steps],          # 记下实际执行的顺序
    )
    resolved.save(out_dir / "config.yaml", extra={
        "time": datetime.now().isoformat(timespec="seconds"),
        "tidyaudio_version": __version__,
        "python": platform.python_version(),
        "input_files": summary.input_count,
        "output_clips": summary.output_count,
        "dropped": summary.dropped_count,
        "cancelled": summary.cancelled,
        "seconds": summary.seconds,
    })
    write_report(out_dir)   # 放在最后：报告要读上面写好的这些文件
    return summary


def _apply_steps(steps: list[Step], clip: Clip,
                 stats: dict) -> tuple[list[Clip], list[Clip]]:
    """让一段音频依次经过所有功能。返回 (留下的, 被丢掉的)。"""
    alive, dropped = [clip], []
    for step in steps:
        next_alive = []
        for c in alive:
            try:
                out = step.apply(c)
            except Exception as e:
                c.info["dropped_reason"] = f"{step.label} failed: {type(e).__name__}: {e}"
                out = []
            if not out:
                dropped.append(c)
                stats[step.key]["dropped"] += 1
            next_alive.extend(out)
        stats[step.key]["in"] += len(alive)
        stats[step.key]["out"] += len(next_alive)
        alive = next_alive
        if not alive:
            break
    return alive, dropped


STEP_FIELDS = ["step", "name", "parameters", "input", "output", "dropped"]


def _step_rows(steps: list[Step], stats: dict, load: dict) -> list[dict]:
    """processing_steps.csv 的内容：第 0 行是读取文件，之后每个功能一行。"""
    rows = [{"step": 0, "name": "Read files", "parameters": "",
             "input": load["in"], "output": load["out"], "dropped": load["dropped"]}]
    for i, step in enumerate(steps, start=1):
        st = stats[step.key]
        if isinstance(step, CustomScript):
            params = str(step.path)
        else:
            # 只列出当前生效的参数（比如选了 lufs，就不列"目标峰值"）
            params = "; ".join(f"{p.label} = {step.values[p.name]}"
                               for p in step.params if p.is_active(step.values))
        rows.append({"step": i, "name": step.label, "parameters": params,
                     "input": st["in"], "output": st["out"], "dropped": st["dropped"]})
    return rows


def _resolve_dataset_csv(setting: str | None, input_dir: Path) -> Path | None:
    if not setting:
        return None
    if setting == "auto":
        return find_dataset_csv(input_dir)
    path = Path(setting)
    return path if path.is_absolute() else input_dir / path


def _copy_scripts(scripts: list[CustomScript], folder: Path) -> None:
    """把用到的脚本复制一份到输出文件夹，以后脚本改了也能找到当时用的版本。"""
    for sc in scripts:
        folder.mkdir(exist_ok=True)
        shutil.copy2(sc.path, folder / f"{sc.key}_{sc.path.name}")


def _write_readme(path: Path, input_dir: Path, summary: "RunSummary", step_rows: list[dict],
                  csv_result: CsvResult | None, footer: list[str],
                  scripts: list[CustomScript]) -> None:
    """给人看的说明：这个数据集是从哪来的、做了哪些处理。"""
    lines = [
        f"This dataset was produced by TidyAudio {__version__}",
        f"Processed: {datetime.now().isoformat(sep=' ', timespec='seconds')}",
        f"Source: {input_dir}",
        f"{summary.input_count} input files -> {summary.output_count} output files, "
        f"{summary.dropped_count} dropped (reasons in ../dropped.csv)",
        "",
        "Processing steps (in order):",
    ]
    for r in step_rows[1:]:
        lines.append(f"  {r['step']}. {r['name']}" + (f": {r['parameters']}" if r["parameters"] else ""))
    if len(step_rows) == 1:
        lines.append("  (none selected; files were only copied / converted)")

    if scripts:
        lines += ["", "Python scripts (copies are in ../scripts/):"]
        lines += [f"  {sc.path.name}  sha256 {sc.sha256()}" for sc in scripts]
        lines.append(f"  Run with: {scripts[0].python_description()}")

    if csv_result:
        lines += ["", "Dataset CSV:"]
        if csv_result.file_column is None:
            lines.append(f"  No column in {csv_result.csv_name} matches the audio file names; "
                         "it was not updated.")
        else:
            lines.append(f"  {csv_result.csv_name} was updated from the original "
                         f"(file column: {csv_result.file_column}).")
            lines.append("  Duration, sample rate and channels reflect the processed audio; rows for "
                         "dropped files were removed; all other columns are unchanged.")
            if csv_result.missing_audio:
                lines.append(f"  Note: {len(csv_result.missing_audio)} rows in the original CSV had no "
                             "matching audio file and were left out, e.g.:")
                lines += [f"    {v}" for v in csv_result.missing_audio[:5]]
            if csv_result.not_in_csv:
                lines.append(f"  Note: {len(csv_result.not_in_csv)} audio files were not listed in the "
                             "original CSV; they were added with empty annotation columns, e.g.:")
                lines += [f"    {v}" for v in csv_result.not_in_csv[:5]]
            if csv_result.summary_written:
                lines.append(f"  summary.csv: counts and total duration per {csv_result.label_column}, "
                             "recomputed after processing.")
        if footer:
            lines += ["", f"Text below the table in the original {csv_result.csv_name} "
                          "(before processing; may no longer apply):"]
            lines += [f"  {line}" for line in footer]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extension(config: PipelineConfig, source: Path) -> str:
    if config.output_format == "same":
        return source.suffix.lstrip(".").lower()
    return config.output_format


def _unique_path(path: Path, used: set[Path]) -> Path:
    """同一个文件夹里有 a.wav 和 a.flac 时，输出都会叫 a.wav，
    第二个自动改名为 a_2.wav，避免互相覆盖。"""
    candidate, n = path, 2
    while candidate in used:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    used.add(candidate)
    return candidate


def _report(progress: ProgressCallback | None, done: int, total: int, rel: Path) -> None:
    if progress:
        progress(done, total, str(rel))


def _write_csv(path: Path, rows: list[dict], default_fields: list[str] | None = None) -> None:
    # 不同的行可能有不同的列（比如只有部分文件做了切片），取所有列的并集
    fields = list(default_fields or [])
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    # utf-8-sig：带 BOM，用 Excel 打开非英文字符（比如中文文件名）不会乱码
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
