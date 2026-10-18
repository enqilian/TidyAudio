# PyInstaller 打包配置：把 TidyAudio 打包成 macOS 的 TidyAudio.app。
# 运行：packaging/build_mac.sh（它会调用这个文件）

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

import sys
sys.path.insert(0, str(ROOT))
from tidyaudio import __version__  # noqa: E402

# 代码以外、运行时要读的文件，放到 App 里和代码相同的相对位置
datas = [
    (str(ROOT / "tidyaudio" / "templates"), "tidyaudio/templates"),     # 脚本模板
    (str(ROOT / "tidyaudio" / "gui" / "icons"), "tidyaudio/gui/icons"),  # 界面图标
    # 在别的 Python 环境里运行脚本时，要用到这个文件的"原文件"，不能只打包编译后的版本
    (str(ROOT / "tidyaudio" / "script_runner.py"), "tidyaudio"),
] + collect_data_files("librosa")

# 第三方库许可证清单（build_mac.sh 先生成好，路径通过环境变量传进来），放在 App 的 Resources 里
import os
_licenses = os.environ.get("TIDYAUDIO_LICENSES")
if _licenses:
    datas.append((str(ROOT / _licenses), "."))
datas.append((str(ROOT / "LICENSE.md"), "."))   # TidyAudio 自己的许可证

# 用户脚本在 App 自带的 Python 里运行时，可能用到这些库里软件本身没用到的部分
# （比如 scipy.io.wavfile），所以把它们整个打包进来（测试代码除外）
not_tests = lambda name: ".tests" not in name and ".testing" not in name  # noqa: E731
hiddenimports = []
for package in ("scipy", "librosa", "soundfile", "soxr", "pyloudnorm", "noisereduce"):
    hiddenimports += collect_submodules(package, filter=not_tests)

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "jupyter", "notebook"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="TidyAudio",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="TidyAudio")
app = BUNDLE(
    coll,
    name="TidyAudio.app",
    icon=str(ROOT / "packaging" / "TidyAudio.icns"),
    bundle_identifier="app.tidyaudio",
    info_plist={
        "CFBundleName": "TidyAudio",
        "CFBundleDisplayName": "TidyAudio",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "NSHighResolutionCapable": True,
        # 界面只做了浅色，系统开了深色模式时也保持浅色，避免标题栏和内容颜色不搭
        "NSRequiresAquaSystemAppearance": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
