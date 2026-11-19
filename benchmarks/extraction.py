"""Run from the repository root: python benchmarks/extraction.py --out <directory>."""
import argparse
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from fake_data import make_fake_v4
from synth import make_movie, workspace_from_movie
from miniscope_gui.core.background import BackgroundParams, run_background
from miniscope_gui.core.extraction import ExtractionParams, load_cells, run_extraction
from miniscope_gui.core.motion import MotionParams, SHIFTS_FILE, run_motion_correction
from miniscope_gui.core.preprocess import PreprocessParams, run_preprocess
from miniscope_gui.core.workspace import Workspace
from miniscope_gui.io.ucla_v4 import MiniscopeSession

CASES = ("background", "strong", "weak", "neighbors", "motion", "gui_default", "gui_relaxed", "gui_identity")
RADIUS = 3.0


def match(truth, candidates):
    """Maximize the number of within-radius pairs, then minimize their distance."""
    distance = cdist(truth, candidates)
    cost = np.where(distance <= RADIUS, distance, (1 + min(distance.shape)) * RADIUS)
    rows, cols = linear_sum_assignment(cost)
    valid = distance[rows, cols] <= RADIUS
    pairs = dict(zip(cols[valid], rows[valid]))
    near = (distance <= RADIUS).any(axis=0)
    labels = ["hit" if i in pairs else "duplicate" if flag else "off_target"
              for i, flag in enumerate(near)]
    missed = sorted(set(range(len(truth))) - set(pairs.values()))
    return labels, pairs, missed


def prepare(folder, case, seed):
    shifts = np.zeros((600, 2))
    if case.startswith("gui_"):
        raw = folder / "raw"
        make_fake_v4(raw, n_frames=600, frames_per_file=200, size=(96, 96), seed=seed)
        ws = Workspace.create_or_open(folder / "work", raw)
        run_preprocess(MiniscopeSession.open(raw), ws, PreprocessParams(spatial_bin=1))
        centres = np.array([[20, 20], [40, 55], [15, 60]])
    else:
        centres = np.array([[24, 22], [24, 59], [60, 22], [60, 59]])
        if case == "background":
            centres = np.empty((0, 2))
        if case == "neighbors":
            centres[1] = [24, 29]
        if case == "motion":
            shifts[100:160] = [4, -3]
            shifts[320:400] = [-5, 2]
        movie, _ = make_movie(T=600, n_cells=len(centres), fixed_centres=centres,
                               amplitude=7 if case == "weak" else 40,
                               shifts=shifts if case == "motion" else None, seed=seed)
        ws = workspace_from_movie(folder / "work", movie)
    if case == "gui_identity":
        run_background(None, ws, BackgroundParams(cell_diameter=8, source="preprocess"))
        return ws, centres
    run_motion_correction(None, ws, MotionParams(max_shift=20))
    # Convert source-space coordinates into the corrected movie's reference frame.
    centres = centres + np.median(shifts + np.load(ws.path(SHIFTS_FILE)), axis=0)[::-1]
    run_background(None, ws, BackgroundParams(cell_diameter=8))
    return ws, centres


def real_recording(folder, data_dir):
    import h5py
    from scipy.io import loadmat
    from scipy.ndimage import center_of_mass
    name = "c25_123_348"
    with h5py.File(data_dir / f"{name}.h5") as h5:
        movie = h5["mov"][:].astype(np.float32)
    movie -= movie.min()
    movie *= 255 / movie.max()
    # HDF5 video axes are reversed relative to MATLAB h5read.
    masks = loadmat(data_dir / "GT Masks" / f"FinalMasks_{name}.mat")["FinalMasks"].T.astype(bool)
    assert masks.shape[1:] == movie.shape[1:]
    truth = np.array([center_of_mass(mask) for mask in masks])
    ws = workspace_from_movie(folder / "work", movie, fps=20)
    run_background(None, ws, BackgroundParams(source="preprocess"))
    return ws, truth, masks


def benchmark(case, seed, folder, data_dir=None):
    masks = None
    if case == "TUnCaT":
        ws, truth, masks = real_recording(folder, data_dir)
    else:
        ws, truth = prepare(folder, case, seed)
    params = ExtractionParams()
    if case in ("gui_relaxed", "gui_identity"):
        params.min_corr, params.min_pnr = 0.6, 4
    try:
        run_extraction(None, ws, params)
    except ValueError as exc:
        if not str(exc).startswith("没有找到符合条件的细胞"):
            raise
        centres, scores, sources = np.empty((0, 2)), np.empty(0), np.empty(0)
    else:
        cells = load_cells(ws)
        centres, scores, sources = cells["centres"], cells["quality"], cells["source"]
    iou = None
    if masks is not None and len(centres):
        footprints = cells["A"].toarray()
        binary = footprints > 0.2 * footprints.max(axis=0)
        ref = masks.reshape(len(masks), -1).astype(int)
        overlap = ref @ binary
        iou = overlap / (ref.sum(axis=1)[:, None] + binary.sum(axis=0) - overlap)
    shifts_path = ws.path(SHIFTS_FILE)
    shift = np.linalg.norm(np.load(shifts_path), axis=1) if shifts_path.exists() else np.array([0.])
    rows, points = [], []
    for cutoff in (0, 60):
        ids = np.flatnonzero(scores * 100 >= cutoff)
        labels, pairs, missed = match(truth, centres[ids])
        hits = len(pairs)
        duplicates = labels.count("duplicate")
        rows.append(dict(case=case, seed=seed, quality_min=cutoff, truth=len(truth),
                         candidates=len(ids), hits=hits, missed=len(missed), duplicates=duplicates,
                         off_target=labels.count("off_target"),
                         precision=hits/len(ids) if len(ids) else None,
                         recall=hits/len(truth) if len(truth) else None,
                         duplicate_fraction=duplicates/len(ids) if len(ids) else None))
        rows[-1].update(shift_median_px=float(np.median(shift)), shift_max_px=float(shift.max()),
                        iou50_hits=None)
        if masks is not None:
            valid = np.zeros((len(truth), len(ids)), bool) if iou is None else iou[:, ids] >= 0.5
            a, b = linear_sum_assignment(~valid)
            rows[-1]["iou50_hits"] = int(valid[a, b].sum())
        for j, i in enumerate(ids):
            y, x = centres[i]
            points.append(dict(case=case, seed=seed, quality_min=cutoff, kind=labels[j],
                               candidate=i, truth=pairs.get(j), y=y, x=x,
                               quality=100*scores[i], source=int(sources[i])))
        for i in range(len(truth)):
            y, x = truth[i]
            points.append(dict(case=case, seed=seed, quality_min=cutoff,
                               kind="missed" if i in missed else "matched_truth",
                               candidate=None, truth=i, y=y, x=x, quality=None, source=None))
    return rows, points, np.load(ws.path("summary_cn.npy"))


def figure(path, panels, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles = {"hit": ("o", "#58e1b6"), "duplicate": ("s", "#ffb547"),
              "off_target": ("x", "#ec766c"), "missed": ("+", "#66b7ff")}
    fig, axs = plt.subplots(2, len(panels), figsize=(2.4 * len(panels), 6.2), squeeze=False)
    for j, case in enumerate(panels):
        rows, points, bg = panels[case]
        for ax, row in zip(axs[:, j], rows):
            ax.imshow(bg, cmap="gray", vmin=np.percentile(bg, 2), vmax=np.percentile(bg, 99))
            for label, (marker, color) in styles.items():
                subset = [p for p in points if p["kind"] == label and p["quality_min"] == row["quality_min"]]
                ax.plot([p["x"] for p in subset], [p["y"] for p in subset], linestyle="none",
                        marker=marker, color=color, markerfacecolor="none", markersize=5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(case.replace("_", " ") if row["quality_min"] == 0 else "")
            ax.set_xlabel(f"{row['hits']}/{row['truth']} matched · {row['candidates']} candidates", fontsize=8)
    axs[0, 0].set_ylabel("All candidates"); axs[1, 0].set_ylabel("Quality ≥ 60")
    handles = [Line2D([], [], marker=m, color=c, markerfacecolor="none", linestyle="none",
                      label=k.replace("_", " ")) for k,(m,c) in styles.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle(f"Cell extraction  |  synthetic examples: seed {seeds[0]}  |  3-pixel matching radius")
    fig.tight_layout(rect=(0, .08, 1, .93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tuncat-dir", type=Path, help="Extracted TUnCaT movie and GT Masks directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results, points, panels = [], [], {}
    for seed in args.seeds:
        for case in CASES:
            with TemporaryDirectory() as folder:
                rows, detail, bg = benchmark(case, seed, Path(folder))
            results.extend(rows); points.extend(detail)
            if seed == args.seeds[0]:
                panels[case] = rows, detail, bg
            print(seed, case, rows[0]["hits"], rows[0]["candidates"], flush=True)
    if args.tuncat_dir:
        with TemporaryDirectory() as folder:
            rows, detail, bg = benchmark("TUnCaT", None, Path(folder), args.tuncat_dir)
        results.extend(rows)
        points.extend(detail)
        panels["TUnCaT"] = rows, detail, bg
    for name, data in (("metrics", results), ("locations", points)):
        with (args.out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    figure(args.out / "extraction.png", panels, args.seeds)


if __name__ == "__main__":
    main()
