"""Behavioural transitions and spatial tuning on synchronized analysis bins."""

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .behavior import BehaviorConfig
from .modeling.common import check_stop, load_dataset
from .registration import load_registration
from .workspace import Workspace


def samples(ds):
    if not all(ds.has(k) for k in ("x", "y", "speed")):
        raise ValueError("请先完成位置追踪和行为同步")
    xy = np.column_stack((ds.beh["x"], ds.beh["y"]))
    speed = ds.beh["speed"]
    valid = ds.valid & np.isfinite(xy).all(axis=1) & np.isfinite(speed)
    valid &= np.isfinite(ds.dff).all(axis=0)
    return xy, speed, valid


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2)


def transitions(ds, stop_speed, window, stop=None):
    xy, speed, valid = samples(ds)
    if not ds.has("freezing"):
        raise ValueError("行为数据缺少僵直标注")
    freeze = ds.beh["freezing"]
    w = max(1, round(window * ds.fps))
    offsets = np.arange(-w, w + 1)

    def bouts(mask):
        out = []
        for a, b in runs(mask):
            if a <= w or b + w >= len(valid) or not valid[a - w : b + w + 1].all():
                continue
            if np.max(np.diff(ds.times[a - w : b + w + 1])) > 1.6 / ds.fps:
                continue
            feat = [
                *xy[a:b].mean(axis=0),
                speed[a:b].mean(),
                speed[a - w : a].mean() - speed[a:b].mean(),
            ]
            out.append((a, b, feat))
        return out

    pos = bouts(valid & freeze)
    neg = [
        r
        for r in bouts(valid & ~freeze & (speed <= stop_speed))
        if not freeze[r[0] - w : r[1] + w + 1].any()
    ]
    if not pos or not neg:
        raise ValueError(
            "没有同时满足完整窗口的僵直和普通停顿，请检查速度阈值或缩短窗口"
        )
    x, y = (np.asarray([r[2] for r in group]) for group in (pos, neg))
    scale = np.vstack((x, y)).std(axis=0)
    cost = cdist(x / np.maximum(scale, 1e-8), y / np.maximum(scale, 1e-8))
    eligible = cost <= 2
    i, j = linear_sum_assignment(np.where(eligible, cost, 3 * (1 + min(cost.shape))))
    # Disjoint full windows also prevent leakage between successive bouts.
    used = np.zeros(len(valid), bool)
    events = []
    for a, b in sorted(zip(i, j), key=lambda pair: cost[pair]):
        intervals = [pos[a][:2], neg[b][:2]]
        (s1, e1), (s2, e2) = intervals
        if not eligible[a, b] or max(s1, s2) - w <= min(e1, e2) + w:
            continue
        if any(used[s - w : e + w + 1].any() for s, e in intervals):
            continue
        for s, e in intervals:
            used[s - w : e + w + 1] = True
        events.append((pos[a], neg[b], cost[a, b]))
    if not events:
        raise ValueError("未找到窗口独立且速度、位置、减速幅度相近的匹配停顿")
    result = {"offset_s": offsets / ds.fps, "cells": ds.ids, "events": []}
    for k, pair in enumerate(events):
        for label, (a, b, f) in zip(("僵直", "普通停顿"), pair[:2]):
            result["events"].append(
                [k + 1, label, ds.times[a], ds.times[b], *f, pair[2]]
            )
    for phase, col in (("on", 0), ("off", 1)):
        data = np.array(
            [[ds.dff[:, bout[col] + offsets] for bout in pair[:2]] for pair in events]
        )
        result[phase + "_heat"] = data.mean(axis=2)
        means = data.mean(axis=0)
        result[phase + "_cells"] = means
        trajectory = means.transpose(0, 2, 1).reshape(-1, ds.n_cells)
        centered = trajectory - trajectory.mean(0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        result[phase + "_trajectory"] = (centered @ components[:2].T).reshape(
            2, len(offsets), 2
        )
        # Each time point predicts both held-out events using all other matched pairs.
        acc = np.full(len(offsets), np.nan)
        if len(events) >= 3:
            correct = []
            for k in range(len(events)):
                check_stop(stop)
                train = np.delete(data, k, axis=0)
                sd = train.std(axis=(0, 1))
                d = (
                    (data[k, :, None] - train.mean(0)[None]) / np.maximum(sd, 1e-6)
                ) ** 2
                dist = d.mean(axis=2)
                correct.append(
                    (
                        (dist[0, 0] < dist[0, 1]).astype(float)
                        + (dist[1, 1] < dist[1, 0])
                        + (dist[0, 0] == dist[0, 1]) * 0.5
                        + (dist[1, 1] == dist[1, 0]) * 0.5
                    )
                    / 2
                )
            acc = np.mean(correct, axis=0)
        result[phase + "_accuracy"] = acc
    return result


def bounds(cfg):
    shape = cfg.arena_shape() if cfg else None
    if shape is None:
        raise ValueError("请先标定场地边界")
    d = np.array(shape.data, float)
    if shape.kind == "rect":
        lo, span = d[:2], d[2:]
    elif shape.kind == "circle":
        lo, span = d[:2] - d[2], np.repeat(2 * d[2], 2)
    else:
        p = d.reshape(-1, 2)
        lo, span = p.min(0), np.ptp(p, axis=0)
    if np.any(span <= 0):
        raise ValueError("场地尺寸无效")
    return shape, lo, span


def tuning(x, bins, mask, n):
    counts = np.bincount(bins[mask], minlength=n)
    out = np.full((len(x), n), np.nan)
    for b in np.flatnonzero(counts):
        out[:, b] = x[:, mask & (bins == b)].mean(axis=1)
    return out, counts


def digitize(values, edges):
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def arena_grid(xy, cfg, n):
    shape, lo, span = bounds(cfg)
    norm = (xy - lo) / span
    edges = np.linspace(0, 1, n + 1)
    bins = digitize(norm[:, 1], edges) * n + digitize(norm[:, 0], edges)
    return norm, bins, shape.contains(*xy.T)


def boundary(ds, cfg, n):
    xy, speed, valid = samples(ds)
    shape, _, _ = bounds(cfg)
    valid &= shape.contains(*xy.T)
    d = np.asarray(shape.data, float)
    if shape.kind == "circle":
        dist = d[2] - np.linalg.norm(xy - d[:2], axis=1)
    elif shape.kind == "rect":
        dist = np.minimum(xy - d[:2], d[:2] + d[2:] - xy).min(axis=1)
    else:
        poly = d.reshape(-1, 2).astype(np.float32)
        dist = np.array(
            [
                cv2.pointPolygonTest(poly, tuple(point), True) if ok else np.nan
                for point, ok in zip(xy, valid)
            ]
        )
    scale = cfg.px_per_cm or 1
    dist /= scale
    if valid.sum() < 3 or np.max(dist[valid]) <= 0:
        raise ValueError("场地内的有效移动轨迹不足")
    edges = np.linspace(0, np.max(dist[valid]), n + 1)
    bins = digitize(dist, edges)
    # Ratio of radial displacement to path length, computed from the same two samples.
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / scale
    ratio = np.r_[np.nan, np.diff(dist) / np.maximum(step, 1e-12)]
    moving = (
        valid
        & np.r_[False, valid[:-1] & (step > 1e-4) & (np.diff(ds.times) < 1.6 / ds.fps)]
    )
    out = {
        "distance": (edges[1:] + edges[:-1]) / 2,
        "edges": edges,
        "cells": ds.ids,
        "unit": ds.beh.get("unit", "cm" if cfg.px_per_cm else "px"),
    }
    for label, mask in (
        ("all", valid),
        ("toward", moving & (ratio < -0.5)),
        ("away", moving & (ratio > 0.5)),
        ("along", moving & (abs(ratio) <= 0.5)),
    ):
        out[label], counts = tuning(ds.dff, bins, mask, n)
        out[label + "_seconds"] = counts / ds.fps
    return out


def target(ds, cfg, schedule, n):
    xy, speed, valid = samples(ds)
    _, world, inside = arena_grid(xy, cfg, n)
    valid &= inside
    schedule = np.asarray(schedule, float)
    if (
        schedule.ndim != 2
        or schedule.shape[1] != 3
        or not len(schedule)
        or not np.isfinite(schedule).all()
        or (np.diff(schedule[:, 0]) <= 0).any()
    ):
        raise ValueError("目标表每行需要开始秒、X、Y，时间必须严格递增")
    epoch = np.searchsorted(schedule[:, 0], ds.times, side="right") - 1
    valid &= epoch >= 0
    if not valid.any():
        raise ValueError("目标时间段与有效记录没有交集")
    dist = np.linalg.norm(xy - schedule[epoch, 1:], axis=1) / (cfg.px_per_cm or 1)
    edges = np.linspace(0, max(dist[valid].max(), 1e-8), n + 1)
    dbin = digitize(dist, edges)
    _, origin, span = bounds(cfg)
    out = {
        "markers": (schedule[:, 1:] - origin) / span,
        "aspect": span[0] / span[1],
        "cells": ds.ids,
        "schedule": schedule,
        "edges": edges,
        "unit": ds.beh.get("unit", "cm" if cfg.px_per_cm else "px"),
    }
    for k in range(len(schedule)):
        mask = valid & (epoch == k)
        for label, keys, size in (("place", world, n * n), ("distance", dbin, n)):
            out[f"{label}_{k}"], occ = tuning(ds.dff, keys, mask, size)
            out[f"{label}_seconds_{k}"] = occ / ds.fps
    # Cross-condition prediction after regressing out speed on training samples only.
    scores = []
    for k in range(len(schedule)):
        train = valid & (epoch != k)
        test = valid & (epoch == k)
        moved = np.linalg.norm(schedule[:, 1:] - schedule[k, 1:], axis=1) > 1e-6
        train &= moved[np.maximum(epoch, 0)]
        if train.sum() < 3 or not test.any():
            continue
        design = np.column_stack((np.ones(len(speed)), speed))
        beta = np.linalg.lstsq(design[train], ds.dff[:, train].T, rcond=None)[0]
        resid = ds.dff - (design @ beta).T
        pm, po = tuning(resid, world, train, n * n)
        dm, do = tuning(resid, dbin, train, n)
        common = test & (po[world] >= 2) & (do[dbin] >= 2)
        if common.any():
            for i, cell in enumerate(ds.ids):
                err = resid[i, common]
                scores.append(
                    [
                        k,
                        int(cell),
                        int(common.sum()),
                        np.mean((err - pm[i, world[common]]) ** 2),
                        np.mean((err - dm[i, dbin[common]]) ** 2),
                    ]
                )
    out["prediction"] = scores
    return out


def crossday(folder, n, stop=None):
    reg = load_registration(folder)
    sessions = reg["summary"]["sessions"]
    if len(sessions) < 2:
        raise ValueError("需要至少两天已配准的数据")
    days = []
    for path in sessions:
        check_stop(stop)
        if not (Path(path) / "project.json").is_file():
            raise ValueError(f"找不到配准项目 {path}")
        ws = Workspace.create_or_open(path)
        cfg = BehaviorConfig.load(ws)
        ds = load_dataset(ws, cfg)
        xy, _, valid = samples(ds)
        norm, bins, inside = arena_grid(xy, cfg, n)
        days.append((ds, norm, bins, valid & inside))
    out = {
        "names": reg["summary"]["names"],
        "errors": np.full((len(days), len(days)), np.nan),
        "pairs": [],
    }
    for a, (src, _, sb, sv) in enumerate(days):
        for b, (dst, xy, tb, tv) in enumerate(days):
            check_stop(stop)
            shared = np.isin(reg["cell_ids"][:, a], src.ids) & np.isin(
                reg["cell_ids"][:, b], dst.ids
            )
            ia = np.searchsorted(src.ids, reg["cell_ids"][shared, a])
            ib = np.searchsorted(dst.ids, reg["cell_ids"][shared, b])
            if len(ia) < 2:
                continue
            maps = []
            for ds, indices, bins, ok in ((src, ia, sb, sv), (dst, ib, tb, tv)):
                rate, occ = tuning(ds.dff[indices], bins, ok, n * n)
                rate[:, occ < 2] = np.nan
                maps.append(rate)
            key = f"{a}_{b}"
            combined = np.stack(maps)
            combined[:, :, ~np.isfinite(combined[:, 0]).all(axis=0)] = np.nan
            out["maps_" + key] = combined
            out["ids_" + key] = np.flatnonzero(shared) + 1
            errors = [[], []]
            by_bin = [[] for _ in range(n * n)]
            for fold in (0, 1):
                check_stop(stop)
                # Compare against a target-day baseline on exactly the same held-out samples.
                ts = sv & ((np.arange(len(sv)) >= len(sv) // 2) == bool(fold))
                td = tv & ((np.arange(len(tv)) >= len(tv) // 2) == bool(fold))
                test = tv & ~td
                sm, so = tuning(src.dff[ia], sb, ts, n * n)
                dm, do = tuning(dst.dff[ib], tb, td, n * n)
                supported = np.flatnonzero((so >= 2) & (do >= 2))
                test &= np.isin(tb, supported)
                if len(supported) < 2 or not test.any():
                    continue
                for j, (m, train, features) in enumerate(
                    ((sm, ts, src.dff[ia]), (dm, td, dst.dff[ib]))
                ):
                    sd = np.maximum(features[:, train].std(axis=1), 1e-6)
                    nearest = cdist(
                        dst.dff[ib][:, test].T / sd, m[:, supported].T / sd
                    ).argmin(axis=1)
                    loc = supported[nearest]
                    pred = (np.column_stack((loc % n, loc // n)) + 0.5) / n
                    err = np.linalg.norm(pred - xy[test], axis=1)
                    errors[j].extend(err)
                    if j == 0:
                        for k in np.unique(tb[test]):
                            by_bin[k].extend(err[tb[test] == k])
            if errors[0]:
                med, baseline = map(np.median, errors)
                out["errors"][a, b] = med
                out["pairs"].append([a, b, len(ia), len(errors[0]), med, baseline])
                out["spatial_" + key] = [np.median(e) if e else np.nan for e in by_bin]
    if not any(a != b for a, b, *_ in out["pairs"]):
        raise ValueError("两天之间缺少共同细胞或共同探索位置，请检查配准、场地和网格数")
    return out
