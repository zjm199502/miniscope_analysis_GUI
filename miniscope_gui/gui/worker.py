"""Run long computations off the GUI thread."""

from PySide6.QtCore import QObject, QThread, Signal


class Worker(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._stop = False

    def stop(self):
        self._stop = True

    def should_stop(self):
        return self._stop

    def run(self):
        try:
            res = self.fn(*self.args, progress=self.progress.emit,
                          should_stop=self.should_stop, **self.kwargs)
            self.finished.emit(res)
        except Exception as e:  # report every error to the user
            self.failed.emit(str(e))


def start_worker(parent, fn, *args, **kwargs):
    """Create a worker in its own thread; caller connects signals then calls thread.start()."""
    thread = QThread(parent)
    worker = Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    return thread, worker
