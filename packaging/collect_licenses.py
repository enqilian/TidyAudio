"""生成第三方库的许可证清单 THIRD_PARTY_LICENSES.txt，打包时放进 App。

App 里带了 PySide6 / Qt、libsndfile、soxr 等库，它们的许可证（比如 LGPL）要求发布时附上
许可证说明。这个脚本读取"当前环境里装的每个库"的名字、版本、许可证类型，
以及库自带的许可证文件全文，写成一个文本文件。

用打包环境的 Python 运行（build_mac.sh 会自动调用）：
    .venv-arm/bin/python packaging/collect_licenses.py <输出文件>
"""

import sys
from importlib import metadata
from pathlib import Path

# 只在打包或测试时用、不会进 App 的库
SKIP = {"pip", "setuptools", "wheel", "pyinstaller", "pyinstaller-hooks-contrib", "altgraph",
        "macholib", "pytest", "pluggy", "iniconfig", "ruff"}
LICENSE_WORDS = ("license", "licence", "copying", "notice", "authors")


def license_type(dist) -> str:
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    text = (meta.get("License") or "").strip()
    if classifiers:
        return ", ".join(classifiers)
    # 有的库把整篇许可证塞在 License 字段里，只取第一行
    return text.splitlines()[0][:120] if text else "see license files below"


def license_files(dist) -> list[tuple[str, str]]:
    found = []
    for f in dist.files or []:
        name = Path(str(f)).name.lower()
        if any(w in name for w in LICENSE_WORDS) and not name.endswith((".py", ".pyc", ".so")):
            path = Path(f.locate())
            if not path.is_file() or path.stat().st_size > 500_000:   # 跳过文件夹和大文件
                continue
            found.append((str(f), path.read_text(encoding="utf-8", errors="replace")))
    return found


def main() -> None:
    out = Path(sys.argv[1])
    dists = {}
    for d in metadata.distributions():
        name = d.metadata["Name"]
        if name and name.lower() not in SKIP:
            dists[name.lower()] = d
    lines = [
        "TidyAudio includes the following third-party software.",
        "Each component is distributed under its own license, reproduced below.",
        "",
        "Summary",
        "=======",
    ]
    ordered = sorted(dists.values(), key=lambda d: d.metadata["Name"].lower())
    for d in ordered:
        lines.append(f"{d.metadata['Name']} {d.version}: {license_type(d)}")
    for d in ordered:
        lines += ["", "", "=" * 78, f"{d.metadata['Name']} {d.version}",
                  f"License: {license_type(d)}"]
        home = d.metadata.get("Home-page") or next(
            (u.split(", ", 1)[-1] for u in d.metadata.get_all("Project-URL") or []), "")
        if home:
            lines.append(f"Project: {home}")
        lines.append("=" * 78)
        for path, text in license_files(d):
            lines += ["", f"--- {path} ---", "", text.strip()]
    # PySide6 / shiboken6 的安装包里没有带许可证全文，只写了 "LGPL-3.0"。
    # LGPL-3.0 要求同时附上 LGPL 和 GPL 第 3 版的全文（LGPL 是在 GPL 基础上加的条款），
    # 还要告诉用户去哪里拿 Qt 的源代码。这两份全文来自 gnu.org，放在 packaging/licenses/。
    here = Path(__file__).parent / "licenses"
    lines += ["", "", "=" * 78,
              "Qt and Qt for Python (PySide6, PySide6_Essentials, PySide6_Addons, shiboken6)",
              "License: GNU Lesser General Public License v3 (LGPL-3.0)",
              "Source code: https://download.qt.io/official_releases/ and https://code.qt.io/",
              "The Qt libraries are included as separate, replaceable dynamic libraries.",
              "=" * 78]
    for name in ("lgpl-3.0.txt", "gpl-3.0.txt"):
        lines += ["", f"--- {name} ---", "", (here / name).read_text(encoding="utf-8").strip()]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Third-party licenses: {len(ordered)} packages → {out}")


if __name__ == "__main__":
    main()
