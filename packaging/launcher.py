"""Entry point for the packaged program (Windows and macOS)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.platform == "darwin" and sys.argv[1:] == ["--check-macos-runtime"]:
        from macos_runtime_check import main as check_macos_runtime
        check_macos_runtime()
        raise SystemExit(0)
    if sys.platform == "win32" and sys.argv[1:2] == ["--check-windows-cuda"]:
        from windows_cuda_check import main as check_windows_cuda
        raise SystemExit(check_windows_cuda(sys.argv[2:]))
    # in-app updates: use a newer miniscope_gui from the user's data folder if one is installed
    from miniscope_gui import patching   # stdlib-only module
    patched = None
    if getattr(sys, "frozen", False):
        for name in [m for m in sys.modules if m == "miniscope_gui" or m.startswith("miniscope_gui.")]:
            del sys.modules[name]          # re-import from the patch, if any
        patched = patching.activate()
    try:
        from miniscope_gui.gui.app import main
        main()
    except SystemExit:
        raise
    except BaseException:
        if patched:
            patching.mark_bad(patched)     # next start falls back to the bundled version
        raise
