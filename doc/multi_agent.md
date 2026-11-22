# 複数エージェント（multi-agent）の使い方

`adk-submission` は宣言的 YAML だけを通す（競合者の Python は実行されない）。
使えるエージェント型は 4 つ（HARNESS_README §2.3）。

| agent_class | 意味 | このコンペでの実用度 |
|---|---|---|
| `LlmAgent` | 通常。`tools` / `sub_agents` / `agent_tool` で連携 | ★★★（主役） |
| `SequentialAgent` | sub_agents を順に実行 | ★★（固定 pipeline に限定） |
| `ParallelAgent` | sub_agents を並列（isolated branches） | ★（編集を並列にすると競合） |
| `LoopAgent` | `max_iterations`（既定 500、上限 500）まで反復 | ★（ループ防止が目的なら逆効果になりやすい） |

制約（違反すると提出エラー）:
- **全エージェントで base model は 1 種類だけ**（`validate_single_declared_model`）。
  異なる **adapter** は agent ごとに使い分け可（例: 親 `main_lora` / 子 `tool_lora`）。
- sub_agent 参照は `{config_path: sub_agents/xxx.yaml}`、`agent_tool` 参照は
  `agent_tool: {config_path, skip_summarization}`。
- 深さ ≤50、agent 総数 ≤500。

---

## 1. `sub_agents`（transfer）と `agent_tool`（呼び出し）の違い

| | `sub_agents` | `agent_tool: {config_path, skip_summarization: true}` |
|---|---|---|
| 動き | 親がモデル判断で処理を **transfer**（会話の流れごと移動） | 親が **tool として 1 回呼ぶ**（戻りは 1 つの text） |
| 履歴 | transfer 後は子が発言（親の履歴に子の中間 tool 出力が残りやすい） | 中間の `read_file` 出力等を親の context に残さない（summarization skip） |
| 制御性 | 低い（ping-pong transfer の风险） | 高い（どこで呼ぶか決めるのは親） |
| 推奨 | ✗ このコンペでは基本使わない | ✓ 探索・検証の隔離 |

**32k context** が本当のボトルネック。`read_file` 1 回 = 最大 150 行 × 数回で探索履歴が埋まり、
compaction（`token_threshold=14336`）で要約が混ざる → 修正精度が落ちる。
だから **探索は子に投げて親の履歴を汚さない** のが最も効く。
→ `submission/sub_agents/code_analyzer.yaml` を `agent_tool` として根に接続済み。

⚠ **tool budget は階層全体で共有**。`@budget_gated` は session 側のカウンタなので、
子の `read_file` も `max_tool_calls` を減らす。子を呼ぶたびに 1 親 call + 子 N calls 消費。

⚠ **宣言していない tool は初期 prompt に出ない**（`_extract_declared_tools` は階層全体を見る）。
graph 系 tool を **子だけ**に宣言すると、prompt には説明が出るのに親は呼べない →
親が幻覚 tool call を出して `ToolErrorPlugin` が error JSON を返す（budget を無駄に消費）。
→ **親にも tool 名は宣言しておく**（使わなくても cost ゼロ）。

---

## 2. 推奨トポロジー（この repo の現在形）

```
agent.yaml  (LlmAgent, main_lora 相当)
 ├─ tools: run_command, get_status, submit_patch, read_file, edit_file, write_file,
 │         get_code_neighbors, search_similar_code, get_code_subgraph   ← 全部宣言
 ├─ agent_tool: sub_agents/code_analyzer.yaml (skip_summarization: true)
 │      tools: read_file, run_command, search_similar_code, get_code_neighbors, get_code_subgraph
 │            ← 編集系（edit_file/write_file/submit_patch）を持たせない
 └─ agent_tool: sub_agents/verifier.yaml (skip_summarization: true)
        tools: run_command, read_file, get_status   ← 検査専用。宣言済みだが未接続
        （`adapters/verifier_lora` が出来るまで agent.yaml 側はコメントアウト。接続すると
          `build_submission.py` が walk して model 種類の検査も行う）
```

verifier の学習データは `python3 src/collect/collect_trajectories.py --role verifier` で
`data/trajectories_verifier.jsonl` に合成する（`--role analyzer` は analyzer 用の read-only 局所化）。
verifier に `edit_file` / `submit_patch` を呼ぶデータが混ざると幻覚 tool call の教師になるので、
collector は `check_role_calls()` でビルド時に弾く。

残り 1 枚（`doc/tasks.md` T6）:

1. **critic（patch 衛生チェック）**
   - tools: `run_command`（`git diff`, `git status`）+ `read_file` のみ。
   - 役: 提出前に `/workspace` に scratch が残っていないか、保護ファイルを触っていないか、
     hunks が妥当かを確認して YES/NO を返す。NO なら親が掃除してから `submit_patch`。

## 3. やってはいけない構成

| 構成 | 壊れ方 |
|---|---|
| `ParallelAgent` で複数 agent が同時に `edit_file` | 同じファイルを Last-write-wins で上書き。patch が壊れ apply 失敗 |
| `LoopAgent` を「解決するまで反復」 | nudge 上限 3 回／turn cap と衝突。tool budget を空にするだけ |
| 親と子両方に `submit_patch` | 子が早く submit して session 即終了 → 検証前に提出 |
| transfer 形式の `sub_agents` 連鎖 | 誰が担当か分からない ping-pong → 最終 text だけ返って NO_PATCH |
| analyzer に `write_file` を許可 | scratch が `/workspace` に増えて patch に混入 |
