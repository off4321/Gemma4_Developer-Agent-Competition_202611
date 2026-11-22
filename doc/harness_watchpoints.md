# ハーネス側の観測点（スコアが伸びないとき見る場所）

採点は 2 段。どちらで壊れているかで打ち手が完全に変わる。

```
Phase 1  agent loop → /workspace を編集 → submit_patch → git diff = agent_patch
Phase 2  新コンテナに agent_patch を 4-pass git apply → test_patch 適用 → pytest
         resolved=True iff exit_code==0 かつ JUnit XML に FAIL_TO_PASS/PASS_TO_PASS の成功
Score    Resolution Rate = resolved / total
```

**Phase 1 で patch を出す力** と **Phase 2 で patch が適用される力** は別物。
`analyze_results.py` はこれを分離して表示する。

---

## 1. 失敗分類（まずここを見る）

```bash
make out-eval RUN=run01
make analyze RUN=run01
```

| 分類 | 兆候（analyze 出力） | 打ち手 |
|---|---|---|
| **NO_PATCH** | `empty / NO_PATCH submissions: N` | prompt が「submit まで行く」/ turn・tool budget が小さい / agent が途中で文章だけ → `max_turns`↑、prompt に「毎 turn必ず tool call」「最後の手段でも submit」明記（system.md 7 項） |
| **apply 失敗** | `patch rejected by git apply: N`、`test_outputs/<id>.log` に `Failed to apply agent patch` | 全ファイル書き換え（`write_file` 乱用）、hunk offset ずれ、CRLF、`old_string` が曖昧。→ 差分は必ず `edit_file` で作る／大きい変更は分割 |
| **保護ファイル污染** | `patch touched protected test/runner files: N` | 追加した再現テスト・`conftest.py`/`pytest.ini` 編集は **strip されて消える**（Phase 2 では reset もされる）。scratch は `/tmp` |
| **テスト失敗** | 上 3 つが 0 なのに resolved=False | 解決できていない（別ファイルが原因 / 仕様の文字列不一致 / 回帰）。trace を読む |
| **tool 誤用** | `tool / log error types: FileEditError xN, TimeoutExceeded xM` | `edit_file` の `old_string` 一意性、`pytest` 全体実行（300s kill）、パス前置きミス |
| **ループ枯渇** | `tool_calls` が上限に一致 / trace に nudge 文字列 | 下記 §2 |

---

## 2. ループ防止（ハーネスの実装仕様）

- 外側の loop は `runner.run_async` の turn を回す。`submit_patch()` が呼ばれず turn が終わると **nudge** が入る。
- nudge は「連続 3 回」で打ち切り（**有効な tool call が 1 つでもあればカウンタは 0 にリセット**）。
- 打ち切り文面は 3 種:
  1. `<\|tool_call` が閉じずに切れた → 「思考を繰り返すな、即 tool call、edit は分割」
  2. `finish_reason` が `MAX_TOKENS`/`LENGTH` → 「思考を数文で tool call を出せ」
  3. 通常の停止で submit 無し → 「continue か submit」
- tool budget: `max_tool_calls` を **20 以上**使っていて **残り 10 以下**になると、全 tool 応答に `budget_warning` が入る。

### 私側でできる打ち手（実装済みの場所）

| 場所 | 設定 |
|---|---|
| `submission/configs/sampling.yaml` | `max_output_tokens: 16384`（4096 だと tool_call が途中で切れる典型）、`thinking_budget: 4096`（思考暴走の上限） |
| `submission/eval_config.yaml` | `max_turns: 150`, `max_tool_calls: 80`（20 だと nudge 込みで途中終了） |
| `submission/prompts/system.md` | 「毎 turn 必ず tool call で終わる」「同一 tool call を繰り返さない」「`FileEditError` なら同じ引数を再送しない」 |
| trace 解析 | `nudge_mentions`, `max_tokens_hits`, `error_types`（`src/analyze/analyze_results.py`） |

**判定手順**: trace に nudge が多い → ①`thinking_budget` を下げる ②`max_output_tokens` を上げる
③prompt で「思考は数文」を強調 ④それでもなら LoRA（`doc/lora_plan.md`）。

---

## 3. tool_call を誤りなく通す

ハーネスの tool 仕様（逸脱すると error JSON が返り、error は **budget を消費する**）:

| tool | 主な制約 |
|---|---|
| `run_command` | 300s 上限（超えると `TimeoutExceeded`、session は続く）/ stdout・stderr 5000 字で cut |
| `read_file` | **150 行** と **10000 字** の両上限、`is_truncated` 付与。`start_line/end_line` は 1-indexed inclusive |
| `edit_file` | `old_string` が **1 回だけ**一致必要（複数→`FileEditError`）。一致は exact → whitespace flexible → token regex の 3 段。空ファイル/空 old_string は error |
| `write_file` | 親ディレクトリ自動作成。ただし丸ごと書き換えは Phase 2 apply の失敗源 |
| `search_similar_code` / `get_code_neighbors` / `get_code_subgraph` | 入力は **dotted symbol path**（`pkg.module.Class.method`）。文章を渡ると lexical fallback に落ちる精度になる |
| `submit_patch` / `get_status` | **tool budget を消費しない**。`submit_patch` は turn 終了扱い（=最後に呼ぶ） |

追加の実務ルール（すべて system.md に入れている）:

1. scratch は `/tmp`（`git add -N .` で `/workspace` の未追跡ファイルは patch に混入する）
2. `pytest.ini` / `conftest.py` は baseline commit 済み。**触ると diff に混入し、Phase 2 で reset される**
3. 再現テストを新規作成しない（`tests/` 配下は strip 対象）
4. `pip install` 系は全て失敗（network none）
5. `edit_file` の `old_string` は 2〜6 行の実在コンテキスト、`new_string` は 60 行未満

---

## 4. patch の運命（Phase 2 で何が起きるか）

```bash
git add -N . ; git diff --binary _swegemma_baseline   # 抽出
# → strip: test_*.py / *_test.py / tests/** / conftest.py / pytest.ini / pyproject.toml / setup.cfg / tox.ini / *.pth
# → apply 4 pass: -p1 → -p1 -3 → --ignore-space-change/--ignore-whitespace → --recount → -p0 → GNU patch
# → 保護ファイル＋test_patch 参照ファイルを eval_baseline に強制 reset
# → test_patch 適用 → pytest -p no:anyio -o timeout=0 -o python_classes="Test* *Test" -q --junitxml
```

- **patch が空**（＝agent が何も編集しなかった、または編集が全部保護対象）→ resolved=False 一直線。
- **全ファイル書き換え** は 4 pass 全部で落ちやすい。差分を作るのは必ず `edit_file`。
- **pytest の `python_classes` が `Test*` / `*Test`** に絞られている。クラスベースのテストを持つ task では
  agent が「テストが通った」と誤認しやすい（成功したように見えるが Phase 2 では別 collection）。

---

## 5. artifacts の読みどころ

```bash
# Phase 1 の思考・tool 呼び順
results/run01/traces/trace_fastapi_15588.json     # thoughts / tool_calls / token 数
results/run01/logs/fastapi_15588.log              # human readable transcript
# Phase 2 の apply / pytest 結果
results/run01/test_outputs/fastapi_15588.log      # "Failed to apply agent patch: ..." 等
results/run01/patches/fastapi_15588.patch         # 実際に提出された diff
```

`patches/*.patch` を自分の目で読むのが一番速い。
「agent は正しく考えているのに patch が apply されない」case は、だいたい `write_file` の全書き換え。
