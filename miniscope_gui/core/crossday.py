"""Analyses that use cells registered across sessions (see registration.py).

Reactivation: assemblies found in a template session (e.g. conditioning) are
tracked in a target session (e.g. recall) through the registered cell
identities. The activation strength in the target uses the template weights
(Peyrache et al., 2009; van de Ven et al., 2016). Significance: the cell
identities in the target are shuffled, which keeps each cell's activity but
breaks which cell is which.

Stability: event rates of tracked cells in every session, population-vector
correlation between sessions, and how much the freezing-related cells of two
sessions overlap (hypergeometric test).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .modeling.common import _bin_mean, _plain, zscore
from .registration import load_registration
from .workspace import Workspace


@dataclass
class ReactivationParams:
    template: int = 0              # session index (in the registration)
    target: int = 1
    bin_s: float = 0.5
    method: str = "ica"
    signal: str = "dff"
    n_assemblies: int = 0
    member_sd: float = 2.0
    n_shuffle: int = 500
    seed: int = 0


@dataclass
class StabilityParams:
    n_shuffle: int = 500
    seed: int = 0


def _ranks(x):
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / cnt[inv]


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def hypergeom_sf(k, n, na, nb) -> float:
    """P(overlap >= k) when na and nb cells are drawn at random from n."""
    if k <= 0:
        return 1.0
    tot = math.comb(n, nb)
    return float(sum(math.comb(na, i) * math.comb(n - na, nb - i)
                     for i in range(k, min(na, nb) + 1)) / tot)


def _check(should_stop):
    if should_stop is not None and should_stop():
        raise InterruptedError("已取消")


def session_activity(folder, cell_ids, bin_s: float, signal: str = "dff"):
    """Activity (len(cell_ids), T) of the given app cell numbers, binned; plus times and freezing."""
    from .signals import load_signals
    ws = Workspace.create_or_open(folder)
    if not ws.step("signals"):
        raise ValueError(f"这个 session 还没有完成第 6 步「信号与 spike」：\n{folder}")
    sig = load_signals(ws)
    t = np.asarray(sig["times"], float)
    fps = 1.0 / np.median(np.diff(t))
    n = max(1, int(round(bin_s * fps))) if bin_s and bin_s > 0 else 1
    src = sig["dff"] if signal == "dff" else sig["spikes"]
    X = _bin_mean(np.asarray(src, float)[np.asarray(cell_ids, int)], n)
    if signal != "dff":
        X = X * n
    out = {"X": X, "times": _bin_mean(t, n), "fps": fps / n, "freezing": None, "valid": None}
    fz = session_freezing(ws)
    if fz is not None:
        valid = _bin_mean(fz[1].astype(float), n) > 0.999
        out["freezing"] = (_bin_mean(fz[0].astype(float), n) >= 0.5) & valid
        out["valid"] = valid
    return out


def session_freezing(ws: Workspace):
    """(freezing, valid) on the calcium frames, or None when the session has no synced behaviour."""
    if not (ws.step("behavior") and ws.step("sync")):
        return None
    from .behavior import BehaviorConfig
    from .sync import behavior_on_calcium_frames
    cfg = BehaviorConfig.load(ws)
    if cfg is None:
        return None
    try:
        b = behavior_on_calcium_frames(ws, cfg)
    except Exception:
        return None
    if "freezing" not in b:
        return None
    return np.asarray(b["freezing"], bool), np.asarray(b["valid"], bool)


def _save(folder, key, arrays, summary):
    folder = Path(folder)
    np.savez_compressed(folder / f"{key}.npz", **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
    with open(folder / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(_plain(summary), f, ensure_ascii=False, indent=2)


def load_crossday(folder, key) -> dict:
    folder = Path(folder)
    z = np.load(folder / f"{key}.npz", allow_pickle=False)
    out = {k: z[k] for k in z.files}
    with open(folder / f"{key}.json", encoding="utf-8") as f:
        out["summary"] = json.load(f)
    return out


def has_result(folder, key) -> bool:
    return folder is not None and (Path(folder) / f"{key}.json").exists()


# ------------------------------------------------------------ reactivation
def run_reactivation(reg_folder, p: ReactivationParams, progress=None, should_stop=None) -> dict:
    from .modeling.ensembles import EnsembleParams, activation_strength, detect
    from .sync import state_modulation
    progress = progress or (lambda f, m: None)
    reg = load_registration(reg_folder)
    s = reg["summary"]
    names = s["names"]
    a, b = int(p.template), int(p.target)
    if a == b:
        raise ValueError("模板 session 和目标 session 不能相同")
    ids = reg["cell_ids"]
    shared = np.where((ids[:, a] >= 0) & (ids[:, b] >= 0))[0]
    if len(shared) < 5:
        raise ValueError(f"「{names[a]}」和「{names[b]}」之间配准上的细胞太少（{len(shared)} 个）")
    progress(0.05, "读取两个 session 的活动…")
    A = session_activity(s["sessions"][a], ids[shared, a], p.bin_s, p.signal)
    B = session_activity(s["sessions"][b], ids[shared, b], p.bin_s, p.signal)
    progress(0.15, f"在「{names[a]}」里找细胞组…")
    ep = EnsembleParams(bin_s=p.bin_s, method=p.method, signal=p.signal, n_assemblies=p.n_assemblies,
                        member_sd=p.member_sd, seed=p.seed, n_shuffle=100)
    res = detect(A["X"], ep, lambda f, m: progress(0.15 + 0.35 * f, m), should_stop)
    K = len(res["patterns"])
    summary = {"template": names[a], "target": names[b], "n_shared": int(len(shared)),
               "n_assemblies": K, "method": p.method}
    arrays = dict(global_ids=shared, ids_template=ids[shared, a], ids_target=ids[shared, b],
                  patterns=res["patterns"], members=res["members"], eig=res["eig"],
                  times_template=A["times"], times_target=B["times"])
    if K == 0:
        _save(reg_folder, "crossday_reactivation", arrays, summary)
        progress(1.0, "完成")
        return summary
    pat = res["patterns"]
    Za, Zb = zscore(A["X"], 1), zscore(B["X"], 1)
    Ra = activation_strength(Za, pat)
    Rb = activation_strength(Zb, pat)
    obs = Rb.mean(1)
    rng = np.random.default_rng(p.seed)
    null = np.empty((p.n_shuffle, K))
    for i in range(p.n_shuffle):
        if i % 20 == 0:
            _check(should_stop)
            progress(0.5 + 0.4 * i / p.n_shuffle, "打乱细胞身份检验…")
        null[i] = activation_strength(Zb[rng.permutation(len(Zb))], pat).mean(1)
    pval = (1 + (null >= obs).sum(0)) / (p.n_shuffle + 1)
    sd = null.std(0)
    z = (obs - null.mean(0)) / np.where(sd > 0, sd, 1)
    arrays.update(act_template=Ra, act_target=Rb, mean_template=Ra.mean(1), mean_target=obs,
                  null_mean=null.mean(0), null_lo=np.percentile(null, 2.5, 0),
                  null_hi=np.percentile(null, 97.5, 0), p=pval, z=z)
    summary.update(reactivated=[int(k + 1) for k in np.where(pval < 0.05)[0]],
                   p=pval.tolist(), z=z.tolist(), mean_template=Ra.mean(1).tolist(), mean_target=obs.tolist())
    for tag, X, R in (("template", A, Ra), ("target", B, Rb)):
        fz = X["freezing"]
        if fz is not None and 0 < fz[X["valid"]].mean() < 1:
            m = state_modulation(R, fz, X["valid"], n_shuffle=500, seed=p.seed)
            arrays[f"freezing_{tag}"] = fz
            arrays[f"fz_in_{tag}"], arrays[f"fz_out_{tag}"] = m["in_state"], m["out_state"]
            arrays[f"fz_p_{tag}"] = m["p_up"]
            summary[f"freezing_up_{tag}"] = [int(k + 1) for k in np.where(m["p_up"] < 0.025)[0]]
    _save(reg_folder, "crossday_reactivation", arrays, summary)
    progress(1.0, "跨天重激活分析完成")
    return summary


# --------------------------------------------------------------- stability
def run_stability(reg_folder, p: StabilityParams, progress=None, should_stop=None) -> dict:
    from .sync import state_modulation
    progress = progress or (lambda f, m: None)
    reg = load_registration(reg_folder)
    s = reg["summary"]
    names = s["names"]
    ids = reg["cell_ids"]
    K, S = ids.shape
    rate = np.full((K, S), np.nan)
    fz_up = np.zeros((K, S), bool)
    has_sig = np.zeros(S, bool)
    has_fz = np.zeros(S, bool)
    for k in range(S):
        _check(should_stop)
        progress(0.05 + 0.8 * k / S, f"读取 {names[k]}…")
        ws = Workspace.create_or_open(s["sessions"][k])
        if not ws.step("signals"):
            continue
        rows = np.where(ids[:, k] >= 0)[0]
        act = session_activity(s["sessions"][k], ids[rows, k], 0, "events")
        dur = len(act["times"]) / act["fps"]
        rate[rows, k] = act["X"].sum(1) / max(dur, 1e-9)
        has_sig[k] = True
        fz = act["freezing"]
        if fz is not None and 0 < fz[act["valid"]].mean() < 1:
            dff = session_activity(s["sessions"][k], ids[rows, k], 0, "dff")["X"]
            m = state_modulation(dff, fz, act["valid"], n_shuffle=p.n_shuffle, seed=p.seed)
            fz_up[rows, k] = m["p_up"] < 0.025
            has_fz[k] = True
    if has_sig.sum() < 2:
        raise ValueError("至少需要 2 个 session 完成第 6 步「信号与 spike」")
    pv = np.full((S, S), np.nan)
    overlap = np.full((S, S), np.nan)
    fz_overlap = []
    present = ids >= 0
    for a in range(S):
        for b in range(S):
            both = present[:, a] & present[:, b]
            overlap[a, b] = both.sum() / max(present[:, a].sum(), 1)
            if has_sig[a] and has_sig[b] and both.sum() >= 3:
                pv[a, b] = spearman(rate[both, a], rate[both, b]) if a != b else 1.0
            if a < b and has_fz[a] and has_fz[b] and both.sum() >= 3:
                n = int(both.sum())
                na, nb = int(fz_up[both, a].sum()), int(fz_up[both, b].sum())
                k_ = int((fz_up[both, a] & fz_up[both, b]).sum())
                pval = hypergeom_sf(k_, n, na, nb) if na and nb else 1.0
                fz_overlap.append({"a": a, "b": b, "n_tracked": n, "n_a": na, "n_b": nb, "both": k_,
                                   "expected": na * nb / n if n else 0.0, "p": pval})
    summary = {"names": names, "n_tracked_all": int(present.all(1).sum()),
               "sessions_with_signals": [names[k] for k in np.where(has_sig)[0]],
               "sessions_with_freezing": [names[k] for k in np.where(has_fz)[0]],
               "freezing_overlap": fz_overlap,
               "median_rate": [float(np.nanmedian(rate[:, k])) if has_sig[k] else None for k in range(S)]}
    _save(reg_folder, "crossday_stability", dict(rate=rate, fz_up=fz_up, pv=pv, overlap=overlap,
                                                  has_sig=has_sig, has_fz=has_fz), summary)
    progress(1.0, "跨天稳定性分析完成")
    return summary


def export_crossday(folder):
    """CSV tables of whatever cross-day analyses exist in the registration folder."""
    from .modeling.common import write_csv
    folder = Path(folder)
    done = []
    reg = load_registration(folder)
    names = reg["summary"]["names"]
    if has_result(folder, "crossday_stability"):
        r = load_crossday(folder, "crossday_stability")
        cols = {"global_id": np.arange(1, len(r["rate"]) + 1)}
        for k, n in enumerate(names):
            cols[f"{n}_cell"] = reg["cell_ids"][:, k]
            cols[f"{n}_events_per_s"] = r["rate"][:, k]
            cols[f"{n}_freezing_up"] = r["fz_up"][:, k].astype(int)
        done.append(write_csv(folder / "crossday_rates.csv", cols))
    if has_result(folder, "crossday_reactivation"):
        r = load_crossday(folder, "crossday_reactivation")
        K = len(r["patterns"])
        cols = {"global_id": r["global_ids"] + 1, "template_cell": r["ids_template"], "target_cell": r["ids_target"]}
        for k in range(K):
            cols[f"weight_{k + 1}"] = r["patterns"][k]
            cols[f"member_{k + 1}"] = r["members"][k].astype(int)
        done.append(write_csv(folder / "crossday_assemblies.csv", cols))
        if K:
            cols = {"assembly": np.arange(1, K + 1), "mean_template": r["mean_template"],
                    "mean_target": r["mean_target"], "null_mean": r["null_mean"], "z": r["z"], "p": r["p"]}
            done.append(write_csv(folder / "crossday_reactivation.csv", cols))
    return done


def describe_reactivation(s: dict) -> str:
    if s["n_assemblies"] == 0:
        return (f"「{s['template']}」里（{s['n_shared']} 个两天都配准上的细胞）没有找到显著的细胞组，"
                "无法比较重激活。可以试着增大时间分箱或使用 spike 事件。")
    K = s["n_assemblies"]
    lines = [f"模板「{s['template']}」→ 目标「{s['target']}」：两天共有 {s['n_shared']} 个细胞，找到 {K} 个细胞组。",
             f"在目标 session 中显著重激活（打乱细胞身份，p < 0.05）：" +
             (", ".join(f"#{k}" for k in s["reactivated"]) if s["reactivated"] else "无")]
    for k in range(K):
        lines.append(f"  细胞组 {k + 1}：模板 {s['mean_template'][k]:.2f}，目标 {s['mean_target'][k]:.2f}，"
                     f"z = {s['z'][k]:.1f}，p = {s['p'][k]:.3f}")
    for tag, label in (("template", s["template"]), ("target", s["target"])):
        if f"freezing_up_{tag}" in s:
            v = s[f"freezing_up_{tag}"]
            lines.append(f"「{label}」里僵直时更强的细胞组：" + (", ".join(f"#{k}" for k in v) if v else "无"))
    return "\n".join(lines)


def describe_stability(s: dict) -> str:
    names = s["names"]
    lines = [f"在全部 {len(names)} 个 session 都追踪到的细胞：{s['n_tracked_all']} 个。",
             "各 session 事件率中位数（次/秒）：" +
             "，".join(f"{n} {r:.3f}" for n, r in zip(names, s["median_rate"]) if r is not None)]
    for o in s["freezing_overlap"]:
        lines.append(f"{names[o['a']]} ↔ {names[o['b']]}：两天都追踪到的 {o['n_tracked']} 个细胞中，僵直相关细胞 "
                     f"{o['n_a']} / {o['n_b']} 个，两天都是的 {o['both']} 个（随机期望 {o['expected']:.1f}，"
                     f"超几何检验 p = {o['p']:.3g}）")
    if not s["sessions_with_freezing"]:
        lines.append("（没有 session 带同步好的僵直数据，所以没有比较僵直细胞的重叠。）")
    return "\n".join(lines)
