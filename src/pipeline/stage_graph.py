"""
ステージ `graph` = ノートブック Cell 3。

vLLM を起動する **前に**、事前計算済みのシンボルグラフと埋め込みが実際に mount されていることを
確認する安い検査。あわせて graph 系 tool が何を受けるかも校准する: 入力は
**dotted symbol path**（pkg.module.Class.method）で、文章を渡ると精度が落ちる lexical fallback に
落ちる（HARNESS_README 6.3）。

ここでの失敗は warning にして落とさない: 選んだ task 群に graph ファイルが無いことはあり得る。
"""

from __future__ import annotations

from typing import Any

from pipeline.common import log, require_module


def run(ctx: Any) -> None:
    require_module("swegemma", why="needed for swegemma.graph")
    from swegemma import graph as sg

    task = ctx.require_tasks()[0]
    repo = getattr(task, "repo", None)
    base_commit = getattr(task, "base_commit", None)
    graph_dir, embeddings_dir = str(ctx.paths.graphs_dir), str(ctx.paths.embeddings_dir)

    try:
        repo_graph = sg.get_graph(repo_name=repo, graph_dir=graph_dir,
                                 embeddings_dir=embeddings_dir, base_commit=base_commit)
    except Exception as exc:  # noqa: BLE001
        log("graph", f"[warn] get_graph 失敗 {repo}@{str(base_commit)[:8]}: "
                     f"{type(exc).__name__}: {str(exc)[:300]}")
        return

    nodes = list(repo_graph.nodes())
    log("graph", f"{task.instance_id} {repo}: nodes={repo_graph.number_of_nodes()} "
                 f"edges={repo_graph.number_of_edges()}")
    if not nodes:
        log("graph", "[warn] graph が空。graph 系 tool は何も返さない")
        return

    query = next((n for n, degree in repo_graph.degree() if degree >= 5), nodes[0])
    log("graph", f"query symbol: {query!r}")
    neighbors = sg.get_neighbor(node=query, graph=repo_graph, max_neighbors=10) or []
    log("graph", "neighbors: " + ", ".join(repr(n) for n in neighbors[:5]))

    try:
        similar = sg.get_similar_nodes(node=query, repo_name=repo, k=5, graph=repo_graph,
                                       graph_dir=graph_dir, embeddings_dir=embeddings_dir,
                                       base_commit=base_commit) or []
        log("graph", "similar: " + ", ".join(
            f"{item.get('node_name')}({item.get('similarity', 0):.4f})" for item in similar[:5]))
    except Exception as exc:  # noqa: BLE001
        log("graph", f"[warn] get_similar_nodes 失敗（embeddings は mount されている？）: "
                     f"{type(exc).__name__}: {str(exc)[:200]}")

    subgraph = sg.get_induced_subgraph(repo_graph, [query, *neighbors[:4]])
    log("graph", f"induced subgraph: {subgraph.number_of_nodes()} nodes / "
                 f"{subgraph.number_of_edges()} edges")
    sample = next((n for n in nodes if "." in str(n)), str(nodes[0]))
    log("graph", f"注意: tool が受けるのは dotted symbol e.g. {sample!r}")
    ctx.notes["graph_nodes"] = repo_graph.number_of_nodes()
    ctx.notes["graph_edges"] = repo_graph.number_of_edges()
