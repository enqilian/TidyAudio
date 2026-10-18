"""数据集自带的 CSV：读取原来的，输出更新后的。

很多数据集都带一个 CSV（比如 file_list.csv），一行一个文件，写着类别、时长等标注。
处理完之后，输出一个同名的 CSV，原则是：
  - 原来有的列，处理后变了 → 改值（比如 duration_s）
  - 原来没有的列 → 加在后面（比如 sample_rate、channels）
  - 和处理无关的列 → 原样保留（比如 class）
  - 被丢掉的文件 → 删掉这一行
  - 切片：1 行变 N 行，标注复制到每一行，并加上来源文件和起止时间

原 CSV 底部如果还有汇总表、备注等内容，不会混进新 CSV（会导致程序读取出错），
而是：汇总表重新统计后写进 summary.csv；备注原文抄进 README.txt 供参考。
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# 常见的列名（全部小写比较）
DURATION_NAMES = ["duration", "duration_s", "duration_sec", "duration_seconds",
                  "length", "length_s", "dur", "seconds"]
SAMPLE_RATE_NAMES = ["sample_rate", "samplerate", "sampling_rate", "sr", "fs", "rate"]
CHANNEL_NAMES = ["channels", "n_channels", "num_channels", "nchannels", "channel"]
LABEL_NAMES = ["class", "label", "category", "classname", "class_name", "target", "labels"]
# 自动寻找数据集 CSV 时，文件名里带这些词的优先
CSV_HINTS = ["list", "meta", "label", "annotation", "manifest", "index", "info"]


@dataclass
class DatasetTable:
    """原 CSV 里的第一张表。"""

    path: Path
    fields: list[str]
    rows: list[dict]
    delimiter: str = ","
    footer: list[str] = field(default_factory=list)   # 表格下面的其他内容（原文）
    has_bom: bool = False


def find_dataset_csv(input_dir: str | Path) -> Path | None:
    """在输入文件夹的最上层找数据集 CSV。

    只有一个 CSV 就用它；有多个时，选文件名里带 list/meta/label 等词的；
    实在分不出来就返回 None（界面上可以手动选）。
    """
    csvs = sorted(p for p in Path(input_dir).glob("*.csv")
                  if p.is_file() and not p.name.startswith("."))
    if len(csvs) == 1:
        return csvs[0]
    hinted = [p for p in csvs if any(h in p.stem.lower() for h in CSV_HINTS)]
    return hinted[0] if len(hinted) == 1 else None


def read_dataset_csv(path: str | Path) -> DatasetTable:
    """只读第一张表：从表头开始，读到第一个空行为止。空行后面的内容原样保存在 footer。"""
    path = Path(path)
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8-sig").splitlines()

    # 跳过开头的空行
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise ValueError(f"CSV is empty: {path}")

    end = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))
    table_lines, footer = lines[:end], lines[end:]
    while footer and not footer[0].strip():
        footer.pop(0)

    try:
        delimiter = csv.Sniffer().sniff(table_lines[0], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(table_lines, delimiter=delimiter)
    rows = [dict(r) for r in reader]
    return DatasetTable(path=path, fields=list(reader.fieldnames or []), rows=rows,
                        delimiter=delimiter, footer=footer, has_bom=has_bom)


def _norm(value: str) -> str:
    """统一路径写法：反斜杠换成斜杠，去掉开头的 ./"""
    v = (value or "").strip().replace("\\", "/")
    while v.startswith("./"):
        v = v[2:]
    return v


def _find_column(fields: list[str], names: list[str]) -> str | None:
    lower = {f.strip().lower(): f for f in fields}
    return next((lower[n] for n in names if n in lower), None)


def match_rows(table: DatasetTable, audio_rel_paths: list[str]) -> tuple[str | None, dict[str, int]]:
    """找出哪一列是文件路径，并把每个音频文件对应到 CSV 的第几行。

    返回 (文件列名, {音频相对路径: 行号})。
    先按完整相对路径匹配（alarm/x.wav），对不上的再按文件名匹配（x.wav），
    选能对上最多的那一列。
    """
    by_path = {p: p for p in audio_rel_paths}
    by_name: dict[str, list[str]] = {}
    for p in audio_rel_paths:
        by_name.setdefault(PurePosixPath(p).name, []).append(p)

    best_col, best_map = None, {}
    for col in table.fields:
        mapping = {}
        for i, row in enumerate(table.rows):
            v = _norm(row.get(col, ""))
            if v in by_path:
                mapping[by_path[v]] = i
            elif len(by_name.get(PurePosixPath(v).name, [])) == 1:   # 文件名唯一才算
                mapping[by_name[PurePosixPath(v).name][0]] = i
        if len(mapping) > len(best_map):
            best_col, best_map = col, mapping
    return best_col, best_map


@dataclass
class CsvResult:
    """更新 CSV 的结果统计，写进 README.txt。"""

    csv_name: str
    file_column: str | None
    matched: int = 0              # 对上了 CSV 的音频文件数
    not_in_csv: list[str] = field(default_factory=list)      # 有音频、CSV 里没有
    missing_audio: list[str] = field(default_factory=list)   # CSV 里有、找不到音频
    label_column: str | None = None
    summary_written: bool = False


def write_updated_csv(
    table: DatasetTable,
    audio_rel_paths: list[str],
    kept_rows: list[dict],
    dataset_dir: Path,
    label_column: str | None = "auto",
) -> CsvResult:
    """生成更新后的数据集 CSV（以及 summary.csv）。

    audio_rel_paths：输入文件夹里所有音频的相对路径（用来匹配 CSV）
    kept_rows：pipeline 输出的每条音频的信息（manifest 的内容）
    dataset_dir：处理后的数据集文件夹，CSV 写在这里，路径写法和原 CSV 保持一致
    """
    file_col, mapping = match_rows(table, audio_rel_paths)
    result = CsvResult(csv_name=table.path.name, file_column=file_col, matched=len(mapping))
    result.not_in_csv = sorted(set(audio_rel_paths) - set(mapping))
    matched_rows = set(mapping.values())
    if file_col:
        result.missing_audio = [table.rows[i][file_col] for i in range(len(table.rows))
                                if i not in matched_rows]

    if file_col is None:
        # 没有任何一列能对上音频文件名，说明这个 CSV 不是这批音频的文件列表，不处理
        return result

    dur_col = _find_column(table.fields, DURATION_NAMES)
    sr_col = _find_column(table.fields, SAMPLE_RATE_NAMES)
    ch_col = _find_column(table.fields, CHANNEL_NAMES)

    # 输出的列：原来的列在前，新加的列在后
    fields = list(table.fields)
    dur_col = dur_col or _append(fields, "duration_s")
    sr_col = sr_col or _append(fields, "sample_rate")
    ch_col = ch_col or _append(fields, "channels")
    segmented = any("segment_index" in r for r in kept_rows)
    if segmented:
        for col in ("source_file", "segment_start_s", "segment_end_s"):
            _append(fields, col)

    out_rows = []
    for k in kept_rows:
        source = k["source"]
        idx = mapping.get(source)
        # CSV 里没有这个文件：照样加一行，标注列留空
        row = dict(table.rows[idx]) if idx is not None else {f: "" for f in table.fields}
        original_value = _norm(row[file_col]) if idx is not None else source
        new_rel = k["path"].split("/", 1)[1]           # 去掉开头的 "dataset/"
        row[file_col] = _same_style(original_value, source, new_rel)
        row[dur_col] = f"{k['duration_sec']:.3f}".rstrip("0").rstrip(".")
        row[sr_col] = k["sample_rate"]
        row[ch_col] = k["channels"]
        if segmented:
            row["source_file"] = original_value
            row["segment_start_s"] = k.get("segment_start_sec", "")
            row["segment_end_s"] = k.get("segment_end_sec", "")
        out_rows.append(row)

    out_path = dataset_dir / table.path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig" if table.has_bom else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=table.delimiter,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    # 按类别重新统计
    if label_column == "auto":
        label_column = _find_column(table.fields, LABEL_NAMES)
    if label_column and label_column in fields:
        _write_summary(dataset_dir / "summary.csv", out_rows, label_column, dur_col)
        result.label_column = label_column
        result.summary_written = True
    return result


def _append(fields: list[str], name: str) -> str:
    if name not in fields:
        fields.append(name)
    return name


def _same_style(original_value: str, source_rel: str, new_rel: str) -> str:
    """新路径的写法和原 CSV 保持一致：
    原来写的是 "alarm/x.wav" → 新的写 "alarm/x_seg000.wav"
    原来只写了文件名 "x.wav" → 新的也只写文件名 "x_seg000.wav"
    """
    if original_value.endswith(source_rel):
        return original_value[: len(original_value) - len(source_rel)] + new_rel
    return PurePosixPath(new_rel).name


def _write_summary(path: Path, rows: list[dict], label_col: str, dur_col: str) -> None:
    """每个类别多少条、总时长多少秒，最后一行是 TOTAL（和常见数据集的写法一样）。"""
    stats: dict[str, list[float]] = {}
    for r in rows:
        stats.setdefault(str(r.get(label_col, "")), []).append(float(r[dur_col]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([label_col, "count", "total_duration_s"])
        for label in sorted(stats):
            w.writerow([label, len(stats[label]), round(sum(stats[label]), 3)])
        w.writerow(["TOTAL", len(rows), round(sum(sum(v) for v in stats.values()), 3)])
