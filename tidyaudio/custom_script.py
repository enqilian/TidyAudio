"""用户自己的 Python 脚本。

脚本里只需要一个函数 process：

    def process(audio, sr):
        # audio：numpy 数组，形状 (采样点数, 声道数)，float32，数值大约在 -1 到 1
        # sr：采样率
        return audio, sr

如果需要知道"这是哪个文件"（比如按类别删文件），可以多写一个参数 info：

    def process(audio, sr, info):
        # info["file"]   相对路径，比如 "alarm/xxx.wav"
        # info["folder"] 所在的文件夹，比如 "alarm"（很多数据集里就是类别）
        # info["name"]   文件名（不含后缀），切片后是 "xxx_seg000"
        ...

程序会看你的函数有几个参数，自动决定传不传 info，所以两种写法都可以。

运行在哪个 Python 里：
  CustomScript   —— 默认，在软件自己的 Python 里运行，只能用软件里装了的库
  ExternalScript —— 在用户选的另一个 Python 里运行（比如朋友自己的 conda 环境），
                    那个环境装了什么库就能用什么库。通过 script_runner.py 来回传音频。

返回值有三种：
    (audio, sr)                    → 1 进 1 出（最常见）
    None                           → 丢掉这条
    [(audio1, sr1), (audio2, sr2)] → 1 进 N 出（比如自己写的切片逻辑）

返回的 audio 可以是：
    (采样点数, 声道数)  —— 和传进来的一样
    一维数组            —— 当作单声道
    (声道数, 采样点数)  —— librosa 的习惯写法，会自动识别并转过来
"""

import hashlib
import importlib.util
import inspect
import platform
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

from .audio_io import Clip
from .steps.base import Step
from .steps.checks import drop

TEMPLATE = Path(__file__).parent / "templates" / "custom_script_template.py"
RUNNER = Path(__file__).parent / "script_runner.py"


class CustomScript(Step):
    """把用户脚本包装成一个 Step，这样 pipeline 处理它和内置功能完全一样。"""

    params = []

    def __init__(self, path: str | Path, index: int = 0):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Python script not found: {self.path}")
        # 每个脚本一个唯一的 key，统计"每一步处理了多少条"时不会混在一起
        self.key = f"script_{index}"
        self.label = f"Python script: {self.path.name}"
        self.values = {"path": str(self.path)}
        self._start()

    # ---- 下面三个方法决定"在哪运行"，ExternalScript 会换成自己的版本 ----

    def _start(self) -> None:
        """加载脚本。"""
        self.process = _load_process(self.path)
        self.wants_info = _accepts_three_args(self.process)

    def _call(self, audio: np.ndarray, sr: int, info: dict):
        """调用脚本的 process()，返回它的原始返回值。"""
        if self.wants_info:
            return self.process(audio, sr, info)
        return self.process(audio, sr)

    def close(self) -> None:
        """处理全部结束后调用，释放资源。"""

    def python_description(self) -> str:
        return f"built-in Python {platform.python_version()}"

    def sha256(self) -> str:
        """脚本内容的指纹。记进 README，以后能确认用的是不是同一个版本的脚本。"""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def apply(self, clip: Clip) -> list[Clip]:
        rel = clip.rel_path or (clip.source.name if clip.source else clip.name)
        info = {"file": rel, "folder": PurePosixPath(rel).parent.name, "name": clip.name}
        # 传一份拷贝进去：用户脚本就算原地修改了数组，也不会影响别的地方
        result = self._call(clip.audio.copy(), clip.sr, info)

        if result is None:
            return drop(clip, f"Dropped by {self.path.name}")
        if isinstance(result, tuple):
            result = [result]
        if not isinstance(result, list):
            raise TypeError(f"{self.path.name}: process() must return (audio, sr), None, "
                            f"or a list of (audio, sr); got {type(result).__name__}")
        if not result:
            return drop(clip, f"Dropped by {self.path.name}")

        if len(result) == 1:
            clip.audio, clip.sr = _check_output(result[0], self.path.name)
            return [clip]

        # 1 进 N 出：每段一个新 Clip，名字加 _part000、_part001……
        out = []
        for i, item in enumerate(result):
            audio, sr = _check_output(item, self.path.name)
            info = dict(clip.info, script_part=i)
            out.append(Clip(audio=audio, sr=sr, name=f"{clip.name}_part{i:03d}",
                            source=clip.source, info=info, subtype=clip.subtype,
                            rel_path=clip.rel_path))
        return out


class ExternalScript(CustomScript):
    """在另一个 Python 环境里运行的脚本。

    启动一个那个 Python 的进程，运行 script_runner.py；每个文件把音频送过去、拿结果回来。
    整个处理过程只启动一次（不是每个文件启动一次），所以几乎不会变慢。
    """

    def __init__(self, path: str | Path, index: int = 0, python: str | Path = ""):
        self.python = str(Path(python).expanduser())
        self.proc = None
        self.log = None
        super().__init__(path, index)

    def _start(self) -> None:
        self._release()        # 重启时，先把上一个进程的管道和日志文件关掉
        # stderr（脚本里的 print、报错信息）写到一个临时文件，出错时拿来给用户看；
        # 这个文件要一直开着，进程结束时在 _release 里关闭
        self.log = tempfile.TemporaryFile()  # noqa: SIM115
        try:
            self.proc = subprocess.Popen(
                [self.python, str(RUNNER), str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        except OSError as e:
            raise RuntimeError(f"Could not start Python at {self.python}: {e}") from e
        header, _ = self._receive()
        if header.get("status") != "ready":
            self.close()
            raise RuntimeError(f"Could not load {self.path.name} with {self.python}: "
                               f"{header.get('message', 'unknown error')}")
        self.versions = (header["python"], header["numpy"])

    def _call(self, audio: np.ndarray, sr: int, info: dict):
        if self.proc is None or self.proc.poll() is not None:
            self._start()      # 上一个文件把它弄崩了，重新启动，后面的文件继续处理
        _send(self.proc.stdin, {"cmd": "process", "sr": int(sr), "info": info}, [audio])
        header, arrays = self._receive()
        status = header.get("status")
        if status == "drop":
            return None
        if status == "ok":
            # 音频和采样率个数必须一样多；不一样说明消息坏了，直接报错，不要悄悄丢掉一部分
            return list(zip(arrays, header["srs"], strict=True))
        raise RuntimeError(header.get("message", "unknown error"))

    def _receive(self):
        try:
            return _receive(self.proc.stdout)
        except EOFError:
            raise RuntimeError(f"The Python process stopped unexpectedly.{self._log_tail()}") from None

    def _log_tail(self) -> str:
        self.log.seek(0)
        text = self.log.read().decode("utf-8", "replace").strip()
        return f" Last output:\n{text[-1500:]}" if text else ""

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                _send(self.proc.stdin, {"cmd": "quit"})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self._release()

    def _release(self) -> None:
        """关掉进程的管道和日志文件（不关的话，处理很多文件、进程重启很多次时会占满文件句柄）。"""
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            for pipe in (self.proc.stdin, self.proc.stdout):
                if pipe:
                    pipe.close()
        self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def python_description(self) -> str:
        py, npv = self.versions
        return f"{self.python} (Python {py}, numpy {npv})"


def check_python(python: str | Path) -> str:
    """检查一个 Python 能不能用来运行脚本：能运行、并且装了 numpy。

    能用就返回版本说明，比如 "Python 3.11.5 · numpy 1.26.4"；不能用就抛出异常，说明原因。
    """
    python = str(Path(python).expanduser())
    code = "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"
    try:
        result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Could not run {python}: {e}") from e
    if result.returncode != 0:
        if "No module named 'numpy'" in result.stderr:
            raise RuntimeError(f"numpy is not installed in this environment. "
                               f"Install it with: {python} -m pip install numpy")
        raise RuntimeError(f"{python} did not run correctly:\n{result.stderr.strip()[-500:]}")
    py, npv = result.stdout.split()
    return f"Python {py} · numpy {npv}"


# 项目文件夹里常见的虚拟环境文件夹名，按这个顺序找
ENV_FOLDER_NAMES = [".venv", "venv", "env", ".env"]


# 在软件里新建的脚本默认存在这里（"文稿"里，用户容易找到）
SCRIPTS_DIR = Path.home() / "Documents" / "TidyAudio" / "Scripts"


def check_script_source(code: str) -> None:
    """保存前检查脚本：语法对不对、有没有 process 函数。有问题就抛出异常，说明是哪一行。

    只检查、不运行，所以脚本里 import 的库有没有装，要到真正处理时才知道。
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Line {e.lineno}: {e.msg}") from None
    if not any(isinstance(node, ast.FunctionDef) and node.name == "process" for node in tree.body):
        raise ValueError("The script must define a function: def process(audio, sr)")


def new_script_path(name: str, folder: Path | None = None) -> Path:
    """新脚本的文件路径。名字重复就在后面加 _2、_3……，不会覆盖已有的脚本。"""
    folder = folder or SCRIPTS_DIR
    stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip() or "script"
    path = folder / f"{stem}.py"
    n = 2
    while path.exists():
        path = folder / f"{stem}_{n}.py"
        n += 1
    return path


def find_python_in(folder: str | Path) -> Path:
    """在用户选的文件夹里找 Python。

    支持三种选法：
      - 选了环境文件夹本身（.venv、conda 环境等）→ 用里面的 bin/python
      - 选了项目文件夹，里面有 .venv / venv / env → 用那个环境里的 python
      - Windows 上是 Scripts/python.exe（venv）或 python.exe（conda）
    找不到就抛出异常，说明找过哪些地方。
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")

    def python_of(env: Path) -> Path | None:
        for rel in ("bin/python", "bin/python3", "Scripts/python.exe", "python.exe"):
            candidate = env / rel
            if candidate.is_file():
                return candidate
        return None

    found = python_of(folder)
    if found:
        return found
    for name in ENV_FOLDER_NAMES:
        if (folder / name).is_dir():
            found = python_of(folder / name)
            if found:
                return found
    raise FileNotFoundError(
        f"No Python environment found in {folder}. Choose a project folder that contains a "
        f"{' / '.join(ENV_FOLDER_NAMES[:3])} folder, or an environment folder itself "
        f"(for conda: the folder inside .../envs/).")


def resolve_python(path: str | Path) -> Path:
    """用户给的可能是 python 文件，也可能是文件夹：文件就直接用，文件夹就去里面找。"""
    path = Path(str(path).strip().strip("'\"")).expanduser()   # 粘贴的路径可能带引号或空格
    if path.is_file():
        return path
    if path.is_dir():
        return find_python_in(path)
    raise FileNotFoundError(f"Path not found: {path}")


def describe_env(python: str | Path) -> str:
    """界面上显示用的名字：环境所在的文件夹，比如 "~/my_project/.venv"。"""
    env = Path(python).expanduser().parent
    if env.name in ("bin", "Scripts"):
        env = env.parent
    try:
        return "~/" + env.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(env)


def make_script(path: str | Path, index: int = 0, python: str | None = None) -> CustomScript:
    """按设置创建脚本：没指定 Python、或者指定的就是软件自己的 Python，就在软件里直接运行。

    注意比较路径时不能"顺着快捷方式找到真实文件"（resolve）：虚拟环境里的 python
    通常是指向别处（比如 Anaconda）的快捷方式，真实文件相同，但装的库完全不同。
    """
    if python and not _same_path(python, sys.executable):
        return ExternalScript(path, index, python)
    return CustomScript(path, index)


def _same_path(a: str | Path, b: str | Path) -> bool:
    import os
    return os.path.abspath(os.path.expanduser(str(a))) == os.path.abspath(str(b))


# ---- 和 script_runner.py 一样的消息格式 ----

def _send(stream, header: dict, arrays=()) -> None:
    import json
    import struct
    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
    header = dict(header, arrays=[list(a.shape) for a in arrays])
    data = json.dumps(header).encode("utf-8")
    stream.write(struct.pack(">I", len(data)) + data)
    for a in arrays:
        stream.write(a.tobytes())
    stream.flush()


def _receive(stream):
    import json
    import struct
    (n,) = struct.unpack(">I", _read_exact(stream, 4))
    header = json.loads(_read_exact(stream, n).decode("utf-8"))
    arrays = []
    for shape in header.get("arrays", []):
        count = int(np.prod(shape)) if shape else 1
        buf = _read_exact(stream, count * 4)
        arrays.append(np.frombuffer(buf, dtype=np.float32).reshape(shape).copy())
    return header, arrays


def _read_exact(stream, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def _load_process(path: Path):
    """读取脚本文件，取出里面的 process 函数。"""
    spec = importlib.util.spec_from_file_location(f"user_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Could not load {path.name}: {type(e).__name__}: {e}") from e
    process = getattr(module, "process", None)
    if not callable(process):
        raise AttributeError(f"{path.name} must define a function: def process(audio, sr)")
    return process


def _accepts_three_args(func) -> bool:
    """process 能不能接收第三个参数 info。"""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3


def _check_output(item, name: str) -> tuple[np.ndarray, int]:
    """检查脚本返回的一段音频，统一成 (采样点数, 声道数) 的 float32。"""
    if not (isinstance(item, (tuple, list)) and len(item) == 2):
        raise TypeError(f"{name}: each result must be (audio, sr)")
    audio, sr = item
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2:
        # 第一维很小、第二维很大，说明是 librosa 习惯的 (声道数, 采样点数)，转过来
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
    else:
        raise ValueError(f"{name}: audio must be 1-D or 2-D, got shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{name}: returned audio contains NaN or Inf")
    sr = int(sr)
    if sr <= 0:
        raise ValueError(f"{name}: sample rate must be positive, got {sr}")
    return np.ascontiguousarray(audio), sr
