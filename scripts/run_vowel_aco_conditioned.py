from __future__ import annotations

import heapq
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_vowel_aco_small import (
    CORE_NODES,
    DATA_CLEAN,
    EDGES as BASE_EDGES,
    EXCLUDED_CHARS,
    FIGS_DIR,
    INPUT_PATH,
    NO_LOW_SLOT_ONSETS,
    SLOT_ORDER,
    SLOT_START,
    VALUE_DIR,
    Edge,
    classify_phonetic,
    set_chinese_font,
    split_phonetic,
)


GROUP_SUMMARY_OUTPUT = VALUE_DIR / "aco_conditioned_group_summary.csv"
PATH_OUTPUT = VALUE_DIR / "aco_conditioned_path_summary.csv"
EDGE_OUTPUT = VALUE_DIR / "aco_conditioned_edge_pheromones.csv"
PRESSURE_OUTPUT = VALUE_DIR / "aco_conditioned_linguistic_pressure.csv"

ONSET_FIG = FIGS_DIR / "aco_conditioned_onset_forces.png"
SUBBRANCH_FIG = FIGS_DIR / "aco_conditioned_subbranch_forces.png"
S3_O_HEATMAP_FIG = FIGS_DIR / "aco_conditioned_s3_o_heatmap.png"

MAIN_CHAIN_KEYS = [("a", "ɑ"), ("ɑ", "ɔ"), ("ɔ", "o"), ("o", "ɷ"), ("ɷ", "u")]
CORE_SPLIT_KEYS = [
    ("u", "əu"),
    ("o", "ou"),
    ("ə", "əɯ"),
    ("ɤ", "ɤɯ"),
    ("ɯ", "ɤɯ"),
    ("ʌ", "ʌɯ"),
    ("ʌ", "ʌɤ"),
]
LOWERING_KEYS = [("u", "ɷ"), ("ɷ", "o"), ("u", "o")]

LOWERING_EDGES = BASE_EDGES + [
    Edge("u", "ɷ", "lowering_branch", 0.46, 0.42, -0.02),
    Edge("ɷ", "o", "lowering_branch", 0.42, 0.36, -0.02),
    Edge("u", "o", "lowering_shortcut", 0.92, 0.78, -0.08),
]

MODEL_EDGES = {
    "base_main_chain": BASE_EDGES,
    "with_s3_lowering": LOWERING_EDGES,
}


def edge_costs(edges: list[Edge], sigma: float) -> dict[tuple[str, str], float]:
    return {edge.key: max(edge.cost(sigma), 0.03) for edge in edges}


def adjacency(edges: list[Edge]) -> dict[str, list[Edge]]:
    graph: dict[str, list[Edge]] = {}
    for edge in edges:
        graph.setdefault(edge.source, []).append(edge)
    return graph


def shortest_cost_to_target(
    target: str, edges: list[Edge], costs: dict[tuple[str, str], float]
) -> dict[str, float]:
    reverse: dict[str, list[tuple[str, float]]] = {}
    nodes = set(CORE_NODES)
    for edge in edges:
        reverse.setdefault(edge.target, []).append((edge.source, costs[edge.key]))
        nodes.add(edge.source)
        nodes.add(edge.target)
    distances = {node: math.inf for node in nodes}
    distances[target] = 0.0
    heap = [(0.0, target)]
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > distances[node]:
            continue
        for prev, cost in reverse.get(node, []):
            new_dist = current_dist + cost
            if new_dist < distances[prev]:
                distances[prev] = new_dist
                heapq.heappush(heap, (new_dist, prev))
    return distances


def choose_next(
    rng: random.Random,
    current: str,
    graph: dict[str, list[Edge]],
    pheromone: dict[tuple[str, str], float],
    costs: dict[tuple[str, str], float],
    remaining: dict[str, float],
    visited: set[str],
    alpha: float,
    beta: float,
) -> Edge | None:
    candidates = [
        edge
        for edge in graph.get(current, [])
        if edge.target not in visited and math.isfinite(remaining.get(edge.target, math.inf))
    ]
    if not candidates:
        return None
    weights = []
    for edge in candidates:
        tau = pheromone[edge.key] ** alpha
        eta = 1.0 / (costs[edge.key] + remaining[edge.target] + 0.08)
        weights.append(tau * (eta**beta))
    total = sum(weights)
    if total <= 0:
        return rng.choice(candidates)
    threshold = rng.random() * total
    running = 0.0
    for edge, weight in zip(candidates, weights):
        running += weight
        if running >= threshold:
            return edge
    return candidates[-1]


def walk_ant(
    rng: random.Random,
    start: str,
    target: str,
    graph: dict[str, list[Edge]],
    pheromone: dict[tuple[str, str], float],
    costs: dict[tuple[str, str], float],
    remaining: dict[str, float],
    alpha: float,
    beta: float,
    max_steps: int = 9,
) -> tuple[list[Edge], float, bool]:
    if start == target:
        return [], 0.0, True
    current = start
    visited = {start}
    path: list[Edge] = []
    total_cost = 0.0
    for _ in range(max_steps):
        edge = choose_next(rng, current, graph, pheromone, costs, remaining, visited, alpha, beta)
        if edge is None:
            return path, total_cost, False
        path.append(edge)
        total_cost += costs[edge.key]
        current = edge.target
        if current == target:
            return path, total_cost, True
        visited.add(current)
    return path, total_cost, current == target


def greedy_best_path(
    start: str,
    target: str,
    edges: list[Edge],
    costs: dict[tuple[str, str], float],
) -> tuple[list[str], float, bool]:
    if start == target:
        return [start], 0.0, True
    graph = adjacency(edges)
    heap = [(0.0, start, [start])]
    best = {start: 0.0}
    while heap:
        score, node, path = heapq.heappop(heap)
        if node == target:
            return path, score, True
        if score > best.get(node, math.inf):
            continue
        for edge in graph.get(node, []):
            if edge.target in path:
                continue
            new_score = score + costs[edge.key]
            if new_score < best.get(edge.target, math.inf):
                best[edge.target] = new_score
                heapq.heappush(heap, (new_score, edge.target, path + [edge.target]))
    return [start], math.inf, False


def load_conditioned_observations() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    required = {
        "point_id",
        "point_name",
        "subbranch",
        "rhyme_modern",
        "chain_slot",
        "onset_class",
        "char",
        "phonetic",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing)}")

    work = df[list(required)].copy()
    for column in required:
        work[column] = work[column].astype("string").str.strip()
    work = work[
        work["point_id"].notna()
        & (work["point_id"] != "")
        & work["chain_slot"].isin(SLOT_ORDER)
        & ~work["char"].isin(EXCLUDED_CHARS)
    ].copy()
    work["onset_class"] = work["onset_class"].replace({"TS*": "TS"})
    work = work[
        ~(
            work["onset_class"].isin(NO_LOW_SLOT_ONSETS)
            & work["chain_slot"].isin(["S0", "S1"])
        )
    ].copy()
    work["phonetic"] = work["phonetic"].apply(split_phonetic)
    work = work.explode("phonetic").reset_index(drop=True)
    work = work[work["phonetic"].notna() & (work["phonetic"] != "")].copy()
    work["category"] = work["phonetic"].apply(classify_phonetic)
    work = work[
        work["category"].isin({"monophthong", "clear_split"})
        & work["phonetic"].isin(CORE_NODES)
    ].copy()
    work["start_node"] = work["chain_slot"].map(SLOT_START)
    return work.reset_index(drop=True)


def build_tasks(work: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    task_cols = group_cols + ["chain_slot", "start_node", "phonetic", "category"]
    for keys, group in work.groupby(task_cols, dropna=False):
        key_map = dict(zip(task_cols, keys))
        rows.append(
            {
                **{column: key_map[column] for column in group_cols},
                "chain_slot": key_map["chain_slot"],
                "start_node": key_map["start_node"],
                "target_node": key_map["phonetic"],
                "target_category": key_map["category"],
                "token_count": len(group),
                "point_count": group["point_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def run_aco(
    tasks: pd.DataFrame,
    edges: list[Edge],
    sigma: float = 1.0,
    iterations: int = 150,
    ants_per_task: int = 3,
    alpha: float = 1.0,
    beta: float = 2.4,
    rho: float = 0.12,
    seed: int = 20260415,
) -> dict[str, pd.DataFrame | dict]:
    rng = random.Random(seed)
    graph = adjacency(edges)
    costs = edge_costs(edges, sigma)
    pheromone = {edge.key: 1.0 for edge in edges}
    deposits = {edge.key: 0.0 for edge in edges}
    total_weight = float(tasks["token_count"].sum())
    weighted_cost_total = 0.0
    weighted_success_total = 0.0
    ant_successes = 0
    ant_failures = 0
    remaining_cache = {
        target: shortest_cost_to_target(target, edges, costs)
        for target in sorted(tasks["target_node"].unique())
    }

    for _ in range(iterations):
        for key in pheromone:
            pheromone[key] *= 1.0 - rho
        iteration_deposits = {edge.key: 0.0 for edge in edges}
        for task in tasks.itertuples(index=False):
            if task.start_node == task.target_node:
                continue
            remaining = remaining_cache[task.target_node]
            if not math.isfinite(remaining.get(task.start_node, math.inf)):
                ant_failures += ants_per_task
                continue
            for _ant in range(ants_per_task):
                path, path_cost, reached = walk_ant(
                    rng,
                    task.start_node,
                    task.target_node,
                    graph,
                    pheromone,
                    costs,
                    remaining,
                    alpha,
                    beta,
                )
                if not reached:
                    ant_failures += 1
                    continue
                ant_successes += 1
                task_weight = float(task.token_count) / total_weight
                deposit = task_weight / (path_cost + 0.18)
                for edge in path:
                    iteration_deposits[edge.key] += deposit
                weighted_cost_total += path_cost * float(task.token_count)
                weighted_success_total += float(task.token_count)
        for key, value in iteration_deposits.items():
            pheromone[key] += value
            deposits[key] += value

    max_pheromone = max(pheromone.values()) if pheromone else 1.0
    max_deposit = max(deposits.values()) if max(deposits.values()) > 0 else 1.0
    edge_rows = []
    for edge in edges:
        edge_rows.append(
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "edge_cost": costs[edge.key],
                "pheromone": pheromone[edge.key],
                "normalized_pheromone": pheromone[edge.key] / max_pheromone,
                "total_deposit": deposits[edge.key],
                "normalized_deposit": deposits[edge.key] / max_deposit,
            }
        )
    edge_df = pd.DataFrame(edge_rows).sort_values("pheromone", ascending=False)

    path_rows = []
    for task in tasks.itertuples(index=False):
        path, raw_cost, reached = greedy_best_path(task.start_node, task.target_node, edges, costs)
        if task.start_node == task.target_node:
            reached = True
            raw_cost = 0.0
        path_rows.append(
            {
                "chain_slot": task.chain_slot,
                "start_node": task.start_node,
                "target_node": task.target_node,
                "target_category": task.target_category,
                "token_count": task.token_count,
                "point_count": task.point_count,
                "reached": reached,
                "best_path": " > ".join(path) if reached else "",
                "path_cost": raw_cost if reached else np.nan,
            }
        )
    path_df = pd.DataFrame(path_rows).sort_values(
        ["token_count", "chain_slot", "target_node"], ascending=[False, True, True]
    )

    summary = {
        "ant_successes": ant_successes,
        "ant_failures": ant_failures,
        "weighted_mean_success_path_cost": weighted_cost_total / weighted_success_total
        if weighted_success_total
        else np.nan,
    }
    return {"edges": edge_df, "paths": path_df, "summary": summary}


def sum_deposit(edge_df: pd.DataFrame, keys: list[tuple[str, str]]) -> float:
    total = 0.0
    indexed = edge_df.set_index(["source", "target"])
    for key in keys:
        if key in indexed.index:
            total += float(indexed.loc[key, "total_deposit"])
    return total


def normalized_edge(edge_df: pd.DataFrame, source: str, target: str) -> float:
    match = edge_df[(edge_df["source"] == source) & (edge_df["target"] == target)]
    if match.empty:
        return 0.0
    return float(match["normalized_pheromone"].iloc[0])


def top_edges(edge_df: pd.DataFrame, limit: int = 5) -> str:
    rows = []
    for row in edge_df.sort_values("total_deposit", ascending=False).head(limit).itertuples(index=False):
        rows.append(f"{row.source}>{row.target}:{row.normalized_deposit:.3f}")
    return " | ".join(rows)


def run_group_model(
    tasks: pd.DataFrame,
    point_count: int,
    group_type: str,
    group_value: str,
    model_name: str,
    edges: list[Edge],
    seed_offset: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    result = run_aco(tasks, edges, seed=20260415 + seed_offset)
    edge_df = result["edges"].copy()
    path_df = result["paths"].copy()
    total_tokens = int(tasks["token_count"].sum())
    path_total = int(path_df["token_count"].sum())
    failure_tokens = int(path_df.loc[~path_df["reached"], "token_count"].sum())
    success_tokens = path_total - failure_tokens
    total_deposit = float(edge_df["total_deposit"].sum()) or 1.0
    main_deposit = sum_deposit(edge_df, MAIN_CHAIN_KEYS)
    split_deposit = sum_deposit(edge_df, CORE_SPLIT_KEYS)
    lowering_deposit = sum_deposit(edge_df, LOWERING_KEYS)
    s3_o_tokens = int(
        path_df.loc[
            (path_df["chain_slot"] == "S3") & (path_df["target_node"] == "o"),
            "token_count",
        ].sum()
    )
    s3_o_reached_tokens = int(
        path_df.loc[
            (path_df["chain_slot"] == "S3")
            & (path_df["target_node"] == "o")
            & path_df["reached"],
            "token_count",
        ].sum()
    )
    summary = {
        "group_type": group_type,
        "group_value": group_value,
        "model": model_name,
        "token_count": total_tokens,
        "task_count": len(tasks),
        "point_count": point_count,
        "success_token_count": success_tokens,
        "failure_token_count": failure_tokens,
        "success_token_share": success_tokens / path_total if path_total else np.nan,
        "weighted_mean_success_path_cost": result["summary"]["weighted_mean_success_path_cost"],
        "main_chain_deposit_share": main_deposit / total_deposit,
        "core_split_deposit_share": split_deposit / total_deposit,
        "lowering_deposit_share": lowering_deposit / total_deposit,
        "u_to_əu_normalized": normalized_edge(edge_df, "u", "əu"),
        "u_to_o_normalized": normalized_edge(edge_df, "u", "o"),
        "u_to_ɷ_normalized": normalized_edge(edge_df, "u", "ɷ"),
        "ɷ_to_o_normalized": normalized_edge(edge_df, "ɷ", "o"),
        "s3_o_token_count": s3_o_tokens,
        "s3_o_reached_token_count": s3_o_reached_tokens,
        "s3_o_reached_share": s3_o_reached_tokens / s3_o_tokens if s3_o_tokens else np.nan,
        "top_edges": top_edges(edge_df),
    }
    edge_df.insert(0, "model", model_name)
    edge_df.insert(0, "group_value", group_value)
    edge_df.insert(0, "group_type", group_type)
    path_df.insert(0, "model", model_name)
    path_df.insert(0, "group_value", group_value)
    path_df.insert(0, "group_type", group_type)
    return summary, edge_df, path_df


def group_specs(work: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    specs: list[tuple[str, str, pd.DataFrame]] = [("global", "ALL", work)]
    for onset, group in work.groupby("onset_class", dropna=False):
        specs.append(("onset_class", str(onset), group))
    for subbranch, group in work.groupby("subbranch", dropna=False):
        if len(group) >= 80:
            specs.append(("subbranch", str(subbranch), group))
    for keys, group in work.groupby(["subbranch", "onset_class"], dropna=False):
        if len(group) >= 45:
            specs.append(("subbranch_onset", f"{keys[0]}__{keys[1]}", group))
    return specs


def pressure_rows(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_type, group_value, group in group_specs(work):
        total = len(group)
        s3 = group[group["chain_slot"] == "S3"]
        s2 = group[group["chain_slot"] == "S2"]
        clear = group[group["category"] == "clear_split"]
        s3_o = s3[s3["phonetic"] == "o"]
        s3_clear = s3[s3["category"] == "clear_split"]
        s2_clear = s2[s2["category"] == "clear_split"]
        rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "token_count": total,
                "point_count": group["point_id"].nunique(),
                "clear_split_tokens": len(clear),
                "clear_split_rate": len(clear) / total if total else np.nan,
                "s2_clear_split_tokens": len(s2_clear),
                "s2_clear_split_rate": len(s2_clear) / len(s2) if len(s2) else np.nan,
                "s3_tokens": len(s3),
                "s3_o_tokens": len(s3_o),
                "s3_o_rate": len(s3_o) / len(s3) if len(s3) else np.nan,
                "s3_clear_split_tokens": len(s3_clear),
                "s3_clear_split_rate": len(s3_clear) / len(s3) if len(s3) else np.nan,
                "s3_o_point_count": s3_o["point_id"].nunique(),
                "s3_o_onset_counts": " | ".join(
                    f"{key}:{value}" for key, value in s3_o["onset_class"].value_counts().items()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_onset_forces(summary: pd.DataFrame) -> None:
    set_chinese_font()
    plot_df = summary[
        (summary["group_type"] == "onset_class") & (summary["model"] == "with_s3_lowering")
    ].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("token_count", ascending=False)
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, 6.3), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    ax.bar(x - 0.24, plot_df["main_chain_deposit_share"], width=0.24, color="#00796b", label="主链")
    ax.bar(x, plot_df["core_split_deposit_share"], width=0.24, color="#d55e00", label="裂化")
    ax.bar(x + 0.24, plot_df["lowering_deposit_share"], width=0.24, color="#4c78a8", label="S3降低")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["group_value"])
    ax.set_ylabel("信息素沉积份额")
    ax.set_title("按声母条件分组的 ACO 推动力", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="y", color="#dedbd2", linewidth=1, alpha=0.8)
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(ONSET_FIG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_subbranch_forces(summary: pd.DataFrame) -> None:
    set_chinese_font()
    plot_df = summary[
        (summary["group_type"] == "subbranch") & (summary["model"] == "with_s3_lowering")
    ].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("lowering_deposit_share")
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, 7.2), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    ax.barh(y - 0.16, plot_df["core_split_deposit_share"], height=0.32, color="#d55e00", label="裂化")
    ax.barh(y + 0.16, plot_df["lowering_deposit_share"], height=0.32, color="#4c78a8", label="S3降低")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["group_value"])
    ax.set_xlabel("信息素沉积份额")
    ax.set_title("按小片分组的裂化与 S3 降低", fontsize=20, weight="bold", pad=14)
    ax.grid(axis="x", color="#dedbd2", linewidth=1, alpha=0.8)
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(SUBBRANCH_FIG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_s3_o_heatmap(pressure: pd.DataFrame) -> None:
    set_chinese_font()
    combo = pressure[pressure["group_type"] == "subbranch_onset"].copy()
    if combo.empty:
        return
    split = combo["group_value"].str.split("__", n=1, expand=True)
    combo["subbranch"] = split[0]
    combo["onset_class"] = split[1]
    pivot = combo.pivot_table(
        index="subbranch",
        columns="onset_class",
        values="s3_o_rate",
        aggfunc="mean",
    ).fillna(0.0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.46 * len(pivot))), facecolor="#f7f7f2")
    ax.set_facecolor("#f7f7f2")
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(0.3, pivot.max().max()))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("S3: u > o 在小片-声母条件中的比例", fontsize=19, weight="bold", pad=14)
    ax.set_xlabel("声母类别")
    ax.set_ylabel("小片")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("S3 中 o 的比例")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(S3_O_HEATMAP_FIG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    VALUE_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    work = load_conditioned_observations()
    summaries = []
    edge_frames = []
    path_frames = []
    seed_offset = 0
    for group_type, group_value, group in group_specs(work):
        group_cols = [] if group_type == "global" else ["condition_value"]
        group = group.copy()
        if group_cols:
            group["condition_value"] = group_value
        tasks = build_tasks(group, group_cols)
        if tasks.empty or tasks["token_count"].sum() < 20:
            continue
        if "condition_value" in tasks.columns:
            tasks = tasks.drop(columns=["condition_value"])
        for model_name, edges in MODEL_EDGES.items():
            seed_offset += 37
            summary, edge_df, path_df = run_group_model(
                tasks,
                int(group["point_id"].nunique()),
                group_type,
                group_value,
                model_name,
                edges,
                seed_offset,
            )
            summaries.append(summary)
            edge_frames.append(edge_df)
            path_frames.append(path_df)

    summary_df = pd.DataFrame(summaries)
    path_df = pd.concat(path_frames, ignore_index=True)
    edge_df = pd.concat(edge_frames, ignore_index=True)
    pressure_df = pressure_rows(work)

    summary_df.to_csv(GROUP_SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    path_df.to_csv(PATH_OUTPUT, index=False, encoding="utf-8-sig")
    edge_df.to_csv(EDGE_OUTPUT, index=False, encoding="utf-8-sig")
    pressure_df.to_csv(PRESSURE_OUTPUT, index=False, encoding="utf-8-sig")

    plot_onset_forces(summary_df)
    plot_subbranch_forces(summary_df)
    plot_s3_o_heatmap(pressure_df)

    print(f"已生成条件化 ACO 分组总表：{GROUP_SUMMARY_OUTPUT}")
    print(f"已生成条件化 ACO 路径表：{PATH_OUTPUT}")
    print(f"已生成条件化 ACO 边信息素表：{EDGE_OUTPUT}")
    print(f"已生成条件化语言压力表：{PRESSURE_OUTPUT}")
    print(f"已生成声母条件图：{ONSET_FIG}")
    print(f"已生成小片条件图：{SUBBRANCH_FIG}")
    print(f"已生成 S3 u>o 热图：{S3_O_HEATMAP_FIG}")


if __name__ == "__main__":
    run()
