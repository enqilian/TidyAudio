#!/bin/bash
# 打包 TidyAudio.app，并压缩成可以发给别人的 zip。
#
# 在项目文件夹里运行（第一次要先运行 packaging/setup_build_env.sh 准备打包环境）：
#   bash packaging/build_mac.sh arm      M 系列芯片版（用 .venv-arm）
#   bash packaging/build_mac.sh intel    Intel 芯片版（用 .venv-intel，在 M 芯片的 Mac 上通过 Rosetta 打包）
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
    arm)   PY=(.venv-arm/bin/python) ;;
    intel) PY=(arch -x86_64 .venv-intel/bin/python) ;;
    *) echo "Usage: bash packaging/build_mac.sh arm|intel"; exit 1 ;;
esac
if ! "${PY[@]}" -c "import PyInstaller" 2>/dev/null; then
    echo "Build environment missing. Run first: bash packaging/setup_build_env.sh ${1}"
    exit 1
fi

ARCH=$("${PY[@]}" -c "import platform; print(platform.machine())")
VERSION=$("${PY[@]}" -c "import tidyaudio; print(tidyaudio.__version__)")
OUT="dist/$ARCH"

rm -rf "build/$ARCH" "$OUT"
# 第三方库的许可证清单（放进 App 的 Resources）
export TIDYAUDIO_LICENSES="build/licenses-$ARCH/THIRD_PARTY_LICENSES.txt"
"${PY[@]}" packaging/collect_licenses.py "$TIDYAUDIO_LICENSES"
"${PY[@]}" -m PyInstaller --noconfirm --clean --distpath "$OUT" --workpath "build/$ARCH" \
    packaging/TidyAudio.spec

# 算出 App 里所有程序文件要求的最高 macOS 版本，写进 App 的设置里，
# 这样系统版本太老的 Mac 会直接提示"需要更新系统"，而不是打开后莫名其妙地崩溃
MINOS=$(find "$OUT/TidyAudio.app" -type f \( -name '*.so' -o -name '*.dylib' -o -perm +111 \) \
    -exec sh -c 'file "$1" | grep -q Mach-O && otool -arch '"$ARCH"' -l "$1" 2>/dev/null \
        | grep -A4 LC_BUILD_VERSION | awk "/minos/{print \$2}"' _ {} \; | sort -V | tail -1)
/usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $MINOS" "$OUT/TidyAudio.app/Contents/Info.plist"
# 改了 App 的设置以后要重新签名（"ad-hoc"签名：不需要 Apple 开发者账号，M 芯片的 Mac 要求必须有）
codesign --force --deep --sign - "$OUT/TidyAudio.app" 2>/dev/null

# ditto 是 macOS 推荐的压缩方式，会保留 App 里的符号链接和签名
ZIP="dist/TidyAudio-${VERSION}-macOS-${ARCH}.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$OUT/TidyAudio.app" "$ZIP"

echo
echo "App: $OUT/TidyAudio.app ($(du -sh "$OUT/TidyAudio.app" | cut -f1))"
echo "Zip: $ZIP ($(du -sh "$ZIP" | cut -f1))"
echo "Runs on: $ARCH Macs with macOS $MINOS or newer"
