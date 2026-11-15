"""Entry point for the packaged Windows program."""
import multiprocessing

from miniscope_gui.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
