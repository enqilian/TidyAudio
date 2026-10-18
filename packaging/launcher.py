"""打包成 App 时的入口。

双击 App：打开图形界面。
带参数运行（比如 TidyAudio.app/Contents/MacOS/TidyAudio run config.yaml）：走命令行，
方便在服务器上用，也方便测试打包后的 App 能不能正常处理。
"""

import multiprocessing
import sys

CLI_COMMANDS = {"run", "scan", "report"}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in CLI_COMMANDS:
        from tidyaudio.__main__ import main as cli_main
        return cli_main(sys.argv[1:])
    from tidyaudio.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()   # 打包后的程序需要这一行，防止某些库启动子进程时出问题
    sys.exit(main())
