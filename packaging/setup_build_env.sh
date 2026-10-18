#!/bin/bash
# 准备"打包 App 专用"的 Python 环境（和平时开发用的 .venv 分开，互不影响）。
# 只需要运行一次（或者依赖更新后再运行一次）。
#
#   bash packaging/setup_build_env.sh arm      M 系列芯片版 → .venv-arm
#   bash packaging/setup_build_env.sh intel    Intel 芯片版 → .venv-intel（需要 Rosetta 和 Anaconda/Miniconda）
#
# 为什么要单独的环境：
#   很多库有两种 Mac 安装包：一种支持老系统，一种要求 macOS 14+。
#   pip 会按"这台电脑"的系统挑，挑到要求新系统的那种，别人的老 Mac 就打不开 App。
#   所以这里明确指定"下载 macOS 12 能用的安装包"，版本固定在 constraints-build.txt。
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
    arm)   ARCH=arm64;  ENV=.venv-arm;   RUN=() ;;
    intel) ARCH=x86_64; ENV=.venv-intel; RUN=(arch -x86_64) ;;
    *) echo "Usage: bash packaging/setup_build_env.sh arm|intel"; exit 1 ;;
esac
TARGET="macosx_12_0_${ARCH}"     # 打包出来的 App 最低支持 macOS 12
CONDA=${CONDA:-/opt/anaconda3/bin/conda}

if [ ! -x "$ENV/bin/python" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        # Intel 架构的 Python：从 conda-forge 下载 osx-64 版本，通过 Rosetta 运行
        CONDA_SUBDIR=osx-64 "$CONDA" create -y -q -p "$ENV" -c conda-forge --override-channels python=3.13
    else
        python3 -m venv "$ENV"
    fi
fi
PY=(${RUN[@]+"${RUN[@]}"} "$ENV/bin/python")   # 写成这样是为了兼容 macOS 自带的老版本 bash

WHEELS=$(mktemp -d)
"${PY[@]}" -m pip install -q --upgrade pip
"${PY[@]}" -m pip download -q -d "$WHEELS" --only-binary=:all: \
    --platform "$TARGET" --python-version 3.13 --implementation cp \
    -r requirements.txt -r packaging/requirements-build.txt -c packaging/constraints-build.txt
"${PY[@]}" -m pip install -q --no-index --find-links "$WHEELS" --force-reinstall \
    -r requirements.txt -r packaging/requirements-build.txt -c packaging/constraints-build.txt
rm -rf "$WHEELS"

"${PY[@]}" -c "import platform, numpy, scipy, PySide6; print('Build env ready:', platform.machine(), \
'numpy', numpy.__version__, 'scipy', scipy.__version__, 'PySide6', PySide6.__version__)"
