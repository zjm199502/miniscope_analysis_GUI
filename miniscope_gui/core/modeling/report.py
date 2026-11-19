"""Registry of the modelling analyses and plain-language result summaries."""

from __future__ import annotations

from pathlib import Path

from ..workspace import Workspace
from .common import load_result

# key -> (name, needs behaviour, what it does / how to read it)
ANALYSES = {
    "glm": ("GLM 编码模型", True,
            "用行为变量（僵直、速度、位置、区域、你填写的事件）去预测每个细胞的 ΔF/F，看每个细胞编码了什么。\n"
            "读法：交叉验证 R² 越高，说明行为越能解释这个细胞；「独立贡献」是去掉某个变量后 R² 下降了多少。"),
    "decoder": ("行为解码器", True,
                "反过来用整个群体的活动去预测动物当下的行为（是否僵直、在哪个区域、速度、位置）。\n"
                "读法：僵直/区域看平衡准确率（0.5 = 瞎猜，1 = 全对）；速度/位置看预测值和真实值的相关系数。"
                "灰色分布是打乱时间后的结果，真实结果要明显高于它。"),
    "ensembles": ("细胞组（ensemble）检测", False,
                  "找出经常一起活动的细胞组。先判断有几个细胞组（特征值超过打乱数据的水平），再用 ICA 或 NMF 得到每组的成员。\n"
                  "读法：每个细胞组有一条「激活强度」曲线，可以看它在僵直时是否更强。"),
    "embedding": ("CEBRA 式低维嵌入", False,
                  "用对比学习把每个时刻的群体活动压缩成 3 维空间里的一个点：行为相近（或时间相近）的时刻彼此靠近。\n"
                  "读法：如果不同行为状态在图上分开，说明群体活动里有这个行为的信息。"
                  "留出检验比较「真实标签」和「打乱标签」训练出的嵌入，真实的应明显更高。\n"
                  "注意：这是内置的轻量实现，不是官方 CEBRA 软件包；需要官方结果时可用「导出给官方 CEBRA」。"),
    "hmm": ("隐状态模型（HMM / AR-HMM / rSLDS）", False,
            "假设群体活动在几个看不见的「状态」之间切换，自动找出这些状态和切换时刻，不需要行为数据。\n"
            "HMM：每个状态有自己的活动模式。AR-HMM：每个状态有自己的动力学（活动如何随时间演变）。\n"
            "循环 AR-HMM：切换概率还取决于当前活动所在的位置，相当于 rSLDS 的简化版（连续潜变量直接取 PCA）。\n"
            "读法：看每个状态占多少时间、持续多久，以及哪个状态和僵直重合。"),
    "network": ("功能连接网络与图论", False,
                "把每个细胞当作节点，活动显著相关的两个细胞之间连一条边，再用图论描述这张网络。\n"
                "读法：度 = 一个细胞连了多少个细胞；枢纽细胞 = 连接特别多的细胞；"
                "模块度高说明网络分成了几个内部联系紧密的小团体；小世界指数 > 1 说明既有局部成团又有捷径。"),
    "info": ("互信息与空间信息", True,
             "每个细胞的活动里包含多少关于位置（空间信息，bits/事件）或行为状态（互信息，bits）的信息，"
             "并用打乱时间的方法检验是否显著。\n"
             "读法：p < 0.05 的细胞对该变量有显著选择性；空间信息显著的细胞即位置细胞候选。"),
    "kinetics": ("钙事件动力学聚类（细胞分型）", False,
                 "检测每个细胞的每一个钙事件，测量上升时间、衰减时间常数 τ、半高宽、振幅、面积、间隔等动力学参数，"
                 "再做无监督聚类（高斯混合模型，自动选择类别数），分别在「事件」和「细胞」两个层面看是否分成几类。\n"
                 "动力学直接从平滑后的 ΔF/F 上测量，不用第 6 步 AR 模型的衰减值。"
                 "每个结果都附带稳定性检验（重抽样）和与信噪比的混杂检查。\n"
                 "注意：单光子 GCaMP 的动力学受指示剂表达量、信噪比和背景影响，得到的类别只是「可能的细胞类型」假设，"
                 "不能直接当作已确认的细胞类型。"),
}

VIEWS = {
    "glm": (("glm_map", "主要编码变量（细胞图）"), ("glm_r2", "解释度 R²"), ("glm_kernel", "事件反应核")),
    "decoder": (("dec_time", "解码结果随时间"), ("dec_null", "与打乱对比"), ("dec_weights", "细胞权重图"), ("dec_weights_y", "位置 Y 权重")),
    "ensembles": (("ens_map", "细胞组成员（细胞图）"), ("ens_act", "激活强度"), ("ens_weights", "权重与特征值")),
    "embedding": (("emb_state", "嵌入（按行为着色）"), ("emb_time", "嵌入（按时间着色）")),
    "hmm": (("hmm_states", "状态随时间"), ("hmm_trans", "转移矩阵与轨迹"), ("hmm_cells", "各状态的细胞活动")),
    "network": (("net_graph", "网络图"), ("net_matrix", "相关矩阵"), ("net_degree", "度分布")),
    "kinetics": (("kin_ev_scatter", "事件聚类散点"), ("kin_ev_wave", "各类事件平均波形"),
                 ("kin_cell_scatter", "细胞聚类散点"), ("kin_map", "细胞图（按类别着色）"),
                 ("kin_check", "信噪比混杂检查")),
    "info": (("info_si", "空间信息（细胞图）"), ("info_fields", "位置野"), ("info_mi", "互信息（细胞图）")),
}


def _ids(v, n=20):
    v = list(v)
    return ", ".join("#" + str(i) for i in v[:n]) + (" …" if len(v) > n else "")


def describe(key: str, s: dict) -> str:
    """One short paragraph for the result box."""
    if key == "glm":
        lines = [f"{s['n_cells']} 个细胞中，{s['n_sig']} 个的活动能被行为显著解释（p < 0.05；FDR 校正后 {s['n_sig_fdr']} 个）。",
                 f"交叉验证 R² 中位数 {s['median_r2']:.3f}（显著细胞 {s['median_r2_sig']:.3f}）。"]
        if len(s["groups"]) > 1:
            lines.append("独立贡献显著的细胞数：" + "，".join(f"{g} {n}" for g, n in zip(s["groups"], s["n_sig_group"])) + "。")
        return "\n".join(lines)
    if key == "decoder":
        if s["binary"]:
            t = (f"解码「{s['target']}」：平衡准确率 {s['metric']:.3f}，AUC {s['auc']:.3f}；"
                 f"打乱后平均 {s['null_mean']:.3f}（95% 上限 {s['null_95']:.3f}），p = {s['p']:.3f}。")
        else:
            t = (f"解码「{s['target']}」：预测与真实的相关 r = {s['metric']:.3f}；"
                 f"打乱后平均 {s['null_mean']:.3f}（95% 上限 {s['null_95']:.3f}），p = {s['p']:.3f}。")
            if "median_error" in s:
                t += f" 位置误差中位数 {s['median_error']:.1f} {s.get('unit', '')}。"
        return t + ("\n群体活动里有这个行为的信息。" if s["p"] < 0.05 else "\n没有超过打乱水平，不能说明群体活动包含这个行为的信息。")
    if key == "ensembles":
        if not s["n_assemblies"]:
            return "没有检测到超过随机水平的细胞组。可以尝试加大时间分箱，或确认细胞数量足够。"
        t = (f"检测到 {s['n_assemblies']} 个细胞组，成员数：{', '.join(str(n) for n in s['sizes'])}；"
             f"{s['n_cells']} 个细胞中有 {s['n_cells_in_any']} 个属于至少一个细胞组。")
        if "freezing_up" in s:
            t += (f"\n僵直时激活更强的细胞组：{', '.join(map(str, s['freezing_up'])) or '无'}；"
                  f"更弱的：{', '.join(map(str, s['freezing_down'])) or '无'}（双侧 p < 0.05）。")
        return t
    if key == "embedding":
        t = (f"已得到 {s['out_dim']} 维嵌入（{'行为模式：' + '、'.join(s['labels']) if s['mode'] == 'behavior' else '时间模式'}），"
             f"损失从 {s['first_loss']:.3f} 降到 {s['final_loss']:.3f}。")
        names = {"freezing_bacc": "僵直（平衡准确率）", "continuous_r": "位置/速度（相关 r）"}
        d, c = s.get("decode") or {}, s.get("decode_control") or {}
        for k, v in d.items():
            t += f"\n留出数据上用嵌入解码{names[k]}：{v:.3f}"
            if k in c:
                t += f"，打乱标签对照：{c[k]:.3f}"
            t += "。"
        return t
    if key == "hmm":
        kind = {"hmm": "HMM", "arhmm": "AR-HMM", "rarhmm": "循环 AR-HMM"}[s["kind"]]
        t = (f"{kind}，{s['n_states']} 个状态（{s['n_latent']} 个主成分，解释 {100 * s['explained']:.0f}% 方差）。\n"
             "各状态占比：" + "，".join(f"状态{i + 1} {100 * o:.0f}%" for i, o in enumerate(s["occupancy"])) + "。\n"
             "平均持续：" + "，".join(f"状态{i + 1} {d:.1f} 秒" for i, d in enumerate(s["dwell_s"])) + "。")
        if "freeze_frac" in s:
            t += ("\n各状态中僵直的比例（整体 " + f"{100 * s['freeze_overall']:.0f}%）："
                  + "，".join(f"状态{i + 1} {100 * f:.0f}%" for i, f in enumerate(s["freeze_frac"])) + "。")
            if s["freeze_states"]:
                t += f" 状态 {', '.join(map(str, s['freeze_states']))} 与僵直显著重合。"
        if "best_k" in s:
            t += f"\n留出数据似然最高的状态数：K = {s['best_k']}。"
        return t
    if key == "network":
        t = (f"{s['n_nodes']} 个细胞，{s['n_edges']} 条边（相关 > {s['threshold']:.3f}），密度 {100 * s['density']:.1f}%，"
             f"平均度 {s['mean_degree']:.1f}。\n"
             f"聚类系数 {s['clustering']:.3f}，全局效率 {s['efficiency']:.3f}，模块度 {s['modularity']:.3f}"
             f"（{s['n_communities']} 个模块），最大连通部分占 {100 * s['giant_component']:.0f}%")
        if "small_world" in s:
            t += f"，小世界指数 {s['small_world']:.2f}"
        t += f"。\n枢纽细胞 {s['n_hubs']} 个：{_ids(s['hubs']) or '无'}。相关与距离的关系 r = {s['dist_corr_r']:.2f}。"
        if "by_state" in s:
            a, b = s["by_state"]["freezing"], s["by_state"]["moving"]
            t += (f"\n僵直时：平均相关 {a['mean_corr']:.3f}，密度 {100 * a['density']:.1f}%，模块度 {a['modularity']:.3f}；"
                  f"非僵直时：平均相关 {b['mean_corr']:.3f}，密度 {100 * b['density']:.1f}%，模块度 {b['modularity']:.3f}（同一阈值）。")
        return t
    if key == "info":
        lines = []
        if "n_place" in s:
            lines.append(f"空间信息：{s['n_cells']} 个细胞中 {s['n_place']} 个显著（p < 0.05 且至少 5 个事件），"
                         f"中位数 {s['median_si']:.2f} bits/事件，使用了 {s['used_s']:.0f} 秒的数据。")
        for v in s["variables"]:
            lines.append(f"与「{v['name']}」的互信息：{v['n_sig']} 个细胞显著（FDR 校正后 {v['n_sig_fdr']} 个），"
                         f"中位数 {v['median_mi']:.4f} bits。")
        return "\n".join(lines)
    if key == "kinetics":
        return describe_kinetics(s)
    return ""


def _verdict(sl):
    K = sl["K"]
    if K == 1:
        return "数据更支持「只有一类」，没有找到分得开的亚群。"
    ari = sl.get("ari")
    sil = sl.get("silhouette")
    jac = [j for j in sl.get("jaccard", []) if j is not None]
    ari = float("nan") if ari is None else ari
    sil = float("nan") if sil is None else sil
    stable = ari >= 0.8 and (min(jac) if jac else 0) >= 0.6
    tail = f"（重抽样一致性 ARI {ari:.2f}，轮廓系数 {sil:.2f}）"
    if stable and sil >= 0.25:
        return f"这 {K} 类可重复，也分得比较开{tail}。"
    if stable:
        return f"分类可重复，但各类之间重叠较多{tail}，更像连续变化，而不是截然不同的类型。"
    return f"分类不稳定{tail}，换一批数据结果就会变，不建议当作不同类型解读。"


def describe_kinetics(s: dict) -> str:
    m = s["median"]
    f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    lines = [f"{s['n_cells']} 个细胞共检测到 {s['n_events']} 个钙事件（孤立事件 {s['n_isolated']} 个），"
             f"{s['n_used_events']} 个用于测量动力学。",
             f"中位数：上升 {f(m['rise_s'])} 秒，衰减 τ {f(m['decay_tau_s'])} 秒，半衰减 {f(m['t_half_s'])} 秒，"
             f"半高宽 {f(m['fwhm_s'])} 秒，振幅 {f(m['amp_dff'])} ΔF/F。"]
    crit = lambda sl: "ICL" if sl.get("criterion", "icl") == "icl" else "BIC"
    ev = s["event"]
    if "skipped" in ev:
        lines.append(f"【事件层面】未聚类：{ev['skipped']}。")
    else:
        how = "固定" if ev.get("forced") else f"按 {crit(ev)} 自动选择"
        lines.append(f"【事件层面】{ev['used']} 个事件分成 {ev['K']} 类（{how}），各类事件数 "
                     + "、".join(map(str, ev["sizes"])) + "。" + _verdict(ev))
        if ev["K"] > 1 and ev.get("purity") is not None and ev.get("purity_null") is not None:
            pn = ev["purity_null"]
            eff = (ev["purity"] - pn) / (1 - pn) if pn < 1 else 0.0
            t = (f"同一个细胞的事件有 {100 * ev['purity']:.0f}% 属于它最主要的那一类（随机打乱时 {100 * pn:.0f}%，"
                 f"p = {ev['purity_p']:.3f}）。")
            if ev["purity_p"] < 0.05 and eff >= 0.3:
                t += "事件类型明显随细胞而定，支持「不同细胞有不同的事件类型」。"
            elif ev["purity_p"] < 0.05:
                t += "事件类型和细胞只有微弱关联：多数细胞同时有几类事件，事件类别更可能反映事件本身的差异（例如叠加事件、信噪比高低）。"
            else:
                t += "事件类型与细胞无关，事件类别不能用来区分细胞。"
            lines.append(t)
        icc = ev.get("icc") or {}
        if icc:
            names = {"decay_tau_s": "衰减 τ", "rise_s": "上升时间", "fwhm_s": "半高宽", "amp_dff": "振幅",
                     "area": "面积", "t_half_s": "半衰减时间"}
            lines.append("参数差异中「细胞之间」所占比例（ICC，越接近 1 越像细胞固有属性）："
                         + "，".join(f"{names.get(k, k)} {f(v)}" for k, v in icc.items()) + "。")
        if ev["K"] > 1 and ev.get("p_snr") is not None and ev["p_snr"] < 0.05 and (ev.get("eta2_snr") or 0) >= 0.1:
            lines.append(f"⚠ 各类事件的信噪比差别较大（解释 {100 * ev['eta2_snr']:.0f}% 的方差），部分类别可能只是信号强弱不同。")
    c = s["cell"]
    if "skipped" in c:
        lines.append(f"【细胞层面】未聚类：{c['skipped']}。")
    else:
        how = "固定" if c.get("forced") else f"按 {crit(c)} 自动选择"
        lines.append(f"【细胞层面】{c['used']} 个细胞分成 {c['K']} 类（{how}），各类细胞数 "
                     + "、".join(map(str, c["sizes"])) + "。" + _verdict(c))
        if c["K"] > 1 and c.get("p_snr") is not None and c["p_snr"] < 0.05 and (c.get("eta2_snr") or 0) >= 0.1:
            lines.append(f"⚠ 各类细胞的信噪比明显不同（解释 {100 * c['eta2_snr']:.0f}% 的方差，p = {c['p_snr']:.3f}），"
                         "类别可能主要反映表达量或成像质量，而不是细胞类型。")
        if c["K"] > 1 and c.get("p_area") is not None and c["p_area"] < 0.05 and (c.get("eta2_area") or 0) >= 0.1:
            lines.append(f"⚠ 各类细胞的面积明显不同（解释 {100 * c['eta2_area']:.0f}% 的方差），也可能与离焦或细胞重叠有关。")
    r = c.get("r_tau_snr")
    if r is not None and abs(r) >= 0.3:
        lines.append(f"⚠ 细胞的衰减 τ 与信噪比相关（r = {r:.2f}），测得的动力学受信号强弱影响，解读时要考虑。")
    lines.append("提示：单光子 GCaMP 的动力学受表达量、信噪比和背景影响，以上类别只是「可能的细胞类型」假设，"
                 "需要用形态、标记物或光遗传标记等独立方法验证。")
    return "\n".join(lines)


def export_all(ws: Workspace, folder=None) -> list[str]:
    """Write CSV files for every finished analysis. Returns the analyses exported."""
    from . import decoder, embedding, ensembles, glm, hmm, info, kinetics, network
    fns = {"kinetics": kinetics.export_kinetics, "glm": glm.export_glm, "decoder": decoder.export_decoder, "ensembles": ensembles.export_ensembles,
           "embedding": embedding.export_embedding, "hmm": hmm.export_hmm, "network": network.export_network,
           "info": info.export_info}
    out = Path(folder) if folder else ws.path("export") / "modeling"
    out.mkdir(parents=True, exist_ok=True)
    done = []
    lines = []
    for key, fn in fns.items():
        if ws.step(f"model_{key}"):
            res = load_result(ws, key)
            fn(ws, res, out)
            done.append(key)
            lines.append(f"【{ANALYSES[key][0]}】\n{describe(key, res['summary'])}\n")
    if lines:
        (out / "modeling_summary.txt").write_text("\n".join(lines), encoding="utf-8-sig")
    return done
