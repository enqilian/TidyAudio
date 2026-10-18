"""在后台线程里跑 pipeline。

如果直接在界面线程里处理，处理期间整个窗口会卡住、进度条也不会动。
所以把处理放到另一个线程，通过"信号"把进度传回界面。
"""

from PySide6.QtCore import QThread, Signal

from ..pipeline import PipelineConfig, run_pipeline


class PipelineWorker(QThread):
    progress = Signal(int, int, str)   # 已处理数、总数、当前文件名
    finished_ok = Signal(object)       # RunSummary
    failed = Signal(str)               # 出错信息

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop = False

    def cancel(self) -> None:
        """界面上点"取消"时调用。当前文件处理完后就停下。"""
        self._stop = True

    def run(self) -> None:   # QThread 启动后会在新线程里执行这个函数
        try:
            summary = run_pipeline(
                self.config,
                progress=lambda done, total, name: self.progress.emit(done, total, name),
                should_stop=lambda: self._stop,
            )
        except Exception as e:
            self.failed.emit(str(e))
        else:
            self.finished_ok.emit(summary)
