"""
段階実行パイプライン（公式 Getting Started ノートブックの py への切り出し）。

公式ノートブック notebooks/getting-started-gemma-4-developer-agent.ipynb は 6 つの code cell
だけで構成されている。この package は同じ手順を「1 ステージ = 1 モジュール」に分解する。
狙いは **同じコードを 2 か所で動かす** こと:

  * このマシン（GPU なし / swegemma 未インストール）での preflight
  * Kaggle kernel での GPU 実行（wheelhouse dataset が swegemma / adk_submission / vllm を入れる）

ステージ -> ノートブック cell の対応:

  setup    Cell 1  環境変数、ログ静默、wheelhouse インストール
  data     Cell 2  tasks.jsonl の読込と task 選別（ids / shard / limit）
  agent    Cell 2  提出物を staged dir にコピーし、variant を適用し、契約検査する
  graph    Cell 3  事前計算済みのシンボルグラフ / 埋め込みの smoke test
  server   Cell 4  単一モデル検査、adapter 発見、vLLM 起動、ADK model registry
  eval     Cell 5  EvalConfig + Evaluator（Phase 1: agent loop、Phase 2: patch 検証）
  package  Cell 6  validate + submission.zip 作成
  patch    --      patch の静的検査（実行前: gold patch、実行後: agent patch）
  analyze  --      results/<run> を失敗 taxonomy に変える

import 規則: この package は src/ の下に置く。兄弟ステージ package（package/ evaluate/ analyze/）は
トップレベル名として import し、それは common が src/ を sys.path に足すことで成立する
（src/evaluate/evaluate_harness.py と同じ流儀）。
"""

from __future__ import annotations

from pipeline.common import PRESETS, STAGE_ORDER, resolve_stages  # noqa: F401

__all__ = ["STAGE_ORDER", "PRESETS", "resolve_stages"]
