"""命令行入口。没有图形界面时（比如在服务器上）用这个。

用法：
  python -m tidyaudio gui                         打开图形界面
  python -m tidyaudio scan  <输入文件夹>          只数一下有多少条音频
  python -m tidyaudio run   <config.yaml>         按配置文件处理
  python -m tidyaudio run   <config.yaml> -i <新的输入文件夹>
                                                        换一批数据，用同样的设置再跑一遍
  python -m tidyaudio report <输出文件夹>         重新生成质量报告
"""

import argparse
import sys

from .pipeline import PipelineConfig, run_pipeline
from .report import write_report
from .scan import scan_folder


def main(argv: list[str] | None = None) -> int:
    """出错时打印一行清楚的错误信息，而不是一大段 Python 报错。"""
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tidyaudio", description="Audio dataset preprocessing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="count the audio files in a folder")
    p_scan.add_argument("folder")

    p_run = sub.add_parser("run", help="process using a config file")
    p_run.add_argument("config", help="path to config.yaml")
    p_run.add_argument("-i", "--input", help="use a different input folder (overrides the config)")
    p_run.add_argument("-o", "--output", help="use a different output folder (overrides the config)")

    sub.add_parser("gui", help="open the graphical interface")

    p_report = sub.add_parser("report", help="regenerate the quality report")
    p_report.add_argument("output_dir")

    args = parser.parse_args(argv)

    if args.command == "gui":
        from .gui import main as gui_main   # 用到时才加载界面库，命令行模式更快
        return gui_main()

    if args.command == "report":
        print(f"Report written: {write_report(args.output_dir)}")
        return 0

    if args.command == "scan":
        result = scan_folder(args.folder)
        print(f"{result.count} audio files")
        for ext, n in sorted(result.by_extension.items()):
            print(f"  {ext}: {n}")
        return 0

    config = PipelineConfig.load(args.config)
    if args.input:
        config.input_dir = args.input
        config.output_dir = None    # 换了输入，输出也跟着换成新的默认位置
    if args.output:
        config.output_dir = args.output

    def show(done, total, name):
        print(f"\r[{done}/{total}] {name[:60]:<60}", end="", flush=True)

    summary = run_pipeline(config, progress=show)
    print()
    print(f"{summary.input_count} input files -> {summary.output_count} output files, "
          f"{summary.dropped_count} dropped, {summary.seconds} s")
    print(f"Output: {summary.output_dir}")
    print(f"Report: {summary.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
