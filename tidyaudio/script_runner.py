"""在"另一个 Python 环境"里运行用户脚本的小程序（跑腿的）。

软件用用户选的那个 Python 启动它：
    /path/to/other/python script_runner.py my_script.py
然后通过 stdin / stdout 来回传消息：软件把音频送过来，它调用脚本的 process()，再把结果送回去。

注意：这个文件是在别人的环境里运行的，所以
  - 不能 import tidyaudio 的任何东西
  - 只能依赖 numpy（交换音频数据需要它）
  - 不要用太新的 Python 语法，兼容 Python 3.8+

消息格式（两个方向都一样）：
    4 个字节：后面 JSON 的长度
    JSON：说明这条消息是什么，"arrays" 里写着后面跟了几个数组、每个的形状
    数组的原始字节：float32，按 JSON 里的形状依次排列
不用 pickle：两边的 numpy 版本可能不同（比如 1.x 和 2.x），pickle 的数组会读不出来。
"""

import importlib.util
import inspect
import json
import struct
import sys
import traceback


def read_exact(stream, n):
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError("connection closed")
        data += chunk
    return data


def send(stream, header, arrays=()):
    import numpy as np
    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
    header = dict(header, arrays=[list(a.shape) for a in arrays])
    data = json.dumps(header).encode("utf-8")
    stream.write(struct.pack(">I", len(data)) + data)
    for a in arrays:
        stream.write(a.tobytes())
    stream.flush()


def receive(stream):
    import numpy as np
    (n,) = struct.unpack(">I", read_exact(stream, 4))
    header = json.loads(read_exact(stream, n).decode("utf-8"))
    arrays = []
    for shape in header.get("arrays", []):
        count = 1
        for d in shape:
            count *= d
        buf = read_exact(stream, count * 4)
        arrays.append(np.frombuffer(buf, dtype=np.float32).reshape(shape).copy())
    return header, arrays


def send_error_without_numpy(stream, message):
    """numpy 都没有的时候，只能发一条不带数组的 JSON。"""
    data = json.dumps({"status": "error", "message": message, "arrays": []}).encode("utf-8")
    stream.write(struct.pack(">I", len(data)) + data)
    stream.flush()


def accepts_three_args(func):
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3


def main():
    stdin = sys.stdin.buffer
    out = sys.stdout.buffer
    # 用户脚本里的 print 会写到 stdout，会把消息格式打乱，所以把它改到 stderr
    sys.stdout = sys.stderr

    try:
        import numpy as np
    except ImportError:
        send_error_without_numpy(
            out, "numpy is not installed in this Python environment ({}). "
                 "Install it with: {} -m pip install numpy".format(sys.executable, sys.executable))
        return

    # 1. 加载用户脚本，告诉软件"准备好了"
    path = sys.argv[1]
    try:
        spec = importlib.util.spec_from_file_location("user_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        process = getattr(module, "process", None)
        if not callable(process):
            raise AttributeError("the script must define a function: def process(audio, sr)")
    except Exception as e:
        send(out, {"status": "error", "message": "{}: {}".format(type(e).__name__, e)})
        return
    wants_info = accepts_three_args(process)
    send(out, {"status": "ready", "python": sys.version.split()[0], "numpy": np.__version__})

    # 2. 一直等软件送音频过来，直到收到 quit
    while True:
        try:
            header, arrays = receive(stdin)
        except EOFError:
            return
        if header.get("cmd") == "quit":
            return
        try:
            audio, sr = arrays[0], header["sr"]
            if wants_info:
                result = process(audio, sr, header.get("info", {}))
            else:
                result = process(audio, sr)

            if result is None:
                send(out, {"status": "drop"})
                continue
            if isinstance(result, tuple):
                result = [result]
            if not isinstance(result, list):
                raise TypeError("process() must return (audio, sr), None, or a list of "
                                "(audio, sr); got {}".format(type(result).__name__))
            if not result:
                send(out, {"status": "drop"})
                continue
            items = []
            for item in result:
                if not (isinstance(item, (tuple, list)) and len(item) == 2):
                    raise TypeError("each result must be (audio, sr)")
                items.append((np.asarray(item[0], dtype=np.float32), int(item[1])))
            send(out, {"status": "ok", "srs": [s for _, s in items]}, [a for a, _ in items])
        except Exception as e:
            send(out, {"status": "error", "message": "{}: {}".format(type(e).__name__, e),
                       "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
