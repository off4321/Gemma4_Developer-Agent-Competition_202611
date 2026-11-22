A ?= results/official
B ?= results/mine
NAME ?= run
RESULTS ?= results
HARNESS ?= official-pin
ENV ?= colab-l4

# Gemma 4 Developer Agent — local + Kaggle driver
# Run `make help` for the short version. Full background: doc/kaggle_cli.md

KAGGLE      ?= kaggle
USERNAME    ?= <username>
EVAL_KERNEL  = $(USERNAME)/gemma4-e4b-agent-eval
TRAIN_KERNEL = $(USERNAME)/gemma4-e4b-lora-train
CODE_DATASET = $(USERNAME)/gemma4-dev-agent-code
SUB_DATASET  = $(USERNAME)/gemma4-dev-agent-submission

PIPELINE     = python3 src/pipeline/run_pipeline.py
VARIANT     ?= baseline
STAGES      ?= gpu

.PHONY: help validate zip collect analyze quota push-code push-submission push-eval push-train \
        measure-notebook push-measure-notebook colab-notebook colab-notebook-hf parity failures scoreboard scoreboard-official scoreboard-table \
        status logs logs-follow out-eval out-train submit submissions clean all \
        pipeline-variants pipeline-plan pipeline-local pipeline-post pipeline-eval pipeline-eval-save \
        pipeline-run pipeline-run-save set-variant strip-pycache kernel-ipynb kernel-ipynb-probe probe probe-out logs-dump

help:
	@echo "local:"
	@echo "  make validate          validate submission/ against the competition contract"
	@echo "  make zip               build submission.zip (validated)"
	@echo "  make collect           build data/trajectories.jsonl from data/tasks.jsonl"
	@echo "  make analyze RUN=run01 analyse a results dir pulled from Kaggle"
	@echo "  make patch-check       static patch analysis (gold patches or results/patches)"
	@echo "pipeline (src/pipeline: notebook cells -> stages, see doc/pipeline.md):"
	@echo "  make pipeline-variants    list variants defined in src/pipeline/variants.yaml"
	@echo "  make pipeline-plan        VARIANT=x STAGES=gpu  show the resolved plan, run nothing"
	@echo "  make pipeline-local       VARIANT=x  no-GPU preflight: contract + variant staging"
	@echo "  make pipeline-post        VARIANT=x  patch + analyze stages over results/<run>"
	@echo "  make pipeline-eval        push code+submission datasets, then run the kernel (GPU quota!)"
	@echo "  make pipeline-eval-save   same push but --no-run (review the kernel first)"
	@echo "  make pipeline-run VARIANT=smoke      kernel CONFIG の variant を差し替えて実行（推奨）"
	@echo "  make pipeline-run-save VARIANT=x     同上を --no-run（push だけ、GPU は動かさない）"
	@echo "kaggle:"
	@echo "  make quota             GPU/TPU quota remaining"
	@echo "  make push-code         upload src/ as dataset $(CODE_DATASET)"
	@echo "  make push-submission   upload submission/ as dataset $(SUB_DATASET)"
	@echo "  make push-eval         push kernels/eval and START the run (GPU quota!)"
	@echo "  make push-eval-save    push kernels/eval without running (--no-run)"
	@echo "  make probe                CPU probe kernel を実行（GPU quota 0）: python/wheel tag/mount 実測"
	@echo "  make probe-out            probe の results/probe を落とす"
	@echo "  make push-train        push kernels/train and START training (GPU quota!)"
	@echo "  make status            latest run status of both kernels"
	@echo "  make logs-follow RUN=run01   live tail of the eval kernel"
	@echo "  make out-eval RUN=run01      download eval outputs to results/run01"
	@echo "  make out-train               download trained adapter to adapters_out"
	@echo "  make submit                  kaggle competitions submit -f submission.zip"
	@echo "  make submissions             list submissions + scores"

RUN ?= run01

pipeline-variants:
	$(PIPELINE) --list-variants

pipeline-plan:
	$(PIPELINE) --variant $(VARIANT) --stages $(STAGES) --plan

pipeline-local:
	$(PIPELINE) --variant $(VARIANT) --stages preflight

pipeline-post:
	$(PIPELINE) --variant $(VARIANT) --stages post

# The eval kernel now imports src/pipeline from the code dataset, so a push must ship both.
pipeline-eval: push-code push-submission
	$(KAGGLE) kernels push -p kernels/eval

pipeline-eval-save: push-code push-submission
	$(KAGGLE) kernels push -p kernels/eval --no-run

validate:
	python3 src/package/build_submission.py --source submission --check-only

zip: validate
	python3 src/package/build_submission.py --source submission --output submission.zip

collect:
	python3 src/collect/collect_trajectories.py --tasks data/tasks.jsonl --output data/trajectories.jsonl

patch-check:
	python3 src/evaluate/evaluate_harness.py --mode patch --tasks data/tasks.jsonl --limit 129

analyze:
	python3 src/analyze/analyze_results.py --results results/$(RUN)

quota:
	$(KAGGLE) quota

# -r tar が必要: kaggle CLI の既定 dir-mode は **ディレクトリを skip する**（prompts/ 等が消える）。
# tar mode は top-level dir を 1 段フラットにするので、mount 側は
# .../gemma4-dev-agent-code/pipeline/run_pipeline.py になる（src/ prefix が無い。stage 側は両方探す）。
push-code:
	@mkdir -p .kaggle/dataset-code
	@rm -rf .kaggle/dataset-code/src && cp -r src .kaggle/dataset-code/src
	@find .kaggle/dataset-code -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@$(KAGGLE) datasets version -p .kaggle/dataset-code -m "sync src/ $$(date -u +%Y-%m-%dT%H:%M:%SZ)" -r tar -q \
	  || $(KAGGLE) datasets create -p .kaggle/dataset-code -r tar -q

push-submission: validate
	@mkdir -p .kaggle/dataset-submission
	@rm -rf .kaggle/dataset-submission/agent.yaml .kaggle/dataset-submission/prompts \
	       .kaggle/dataset-submission/sub_agents .kaggle/dataset-submission/configs \
	       .kaggle/dataset-submission/eval_config.yaml .kaggle/dataset-submission/adapters
	@cp submission/agent.yaml submission/eval_config.yaml .kaggle/dataset-submission/
	@cp -r submission/prompts submission/configs submission/sub_agents .kaggle/dataset-submission/
	@if [ -d submission/adapters ] && [ -n "$$(find submission/adapters -name 'adapter_config.json')" ]; then \
	  cp -r submission/adapters .kaggle/dataset-submission/adapters; fi
	@$(KAGGLE) datasets version -p .kaggle/dataset-submission -m "sync submission/ $$(date -u +%Y-%m-%dT%H:%M:%SZ)" -r tar -q \
	  || $(KAGGLE) datasets create -p .kaggle/dataset-submission -r tar -q

push-eval: push-code push-submission
	$(KAGGLE) kernels push -p kernels/eval

push-eval-save: push-code push-submission
	$(KAGGLE) kernels push -p kernels/eval --no-run

# kernel には引数を渡せないので、variant は kernel CONFIG の文字列に書き込んでから push する。
# VARIANT=<name> を付けると pipeline-run / pipeline-run-save がそれを反映する。
set-variant:
	@test -n "$(VARIANT)" || (echo "使い方: make pipeline-run VARIANT=smoke"; exit 2)
	@python3 src/pipeline/set_kernel_variant.py $(VARIANT)

# kernel は .ipynb で push する（script kernel は Kaggle 側で nbconvert に落ちた実測がある）。
# .py が source of truth で、.ipynb は生成物（make kernel-ipynb で作る / 手で触らない）。
kernel-ipynb:
	python3 src/pipeline/make_kernel_notebook.py --source kernels/eval/gemma4_e4b_eval.py \
	  --title gemma4-e4b-agent-eval --code-slug $(notdir $(CODE_DATASET)) \
	  --metadata kernels/eval/kernel-metadata.json
	python3 -c "import nbformat;nbformat.validate(nbformat.read('kernels/eval/gemma4_e4b_eval.ipynb',as_version=4));print('[kernel-ipynb] nbformat validate OK')"

# セル実行版の測定ノートブック（kernels/measure/）。あなたは Kaggle 画面で 1 個ずつ回す。
measure-notebook:
	python3 src/pipeline/make_measure_notebook.py

colab-notebook:                              # 既定の 1 冊（Kaggle 資材モード / Secrets の KAGGLE_API_TOKEN を使う）
	python3 src/pipeline/make_colab_notebook.py

colab-notebook-hf:                           # 同じ notebook の HF 資材モード（Kaggle を触らない）
	python3 src/pipeline/make_colab_notebook.py --source hf

parity:                                      # 公式 harness 出力と自分の pipeline 出力を突き合わせる
	python3 src/pipeline/parity_check.py --a $(A) --b $(B) --label 公式harness 自分のpipeline

scoreboard:                                  # 測定台帳に 1 行足す（make scoreboard NAME=colab12 RESULTS=... HARNESS=official-pin ENV=colab-l4）
	python3 src/pipeline/scoreboard.py --add --name $(NAME) --results $(RESULTS) --harness $(HARNESS) --env $(ENV) $(NOTE:%=--note %)

scoreboard-official:                         # Kaggle 提出一覧（公式スコア）を台帳に反映
	python3 src/pipeline/scoreboard.py --official

failures:                                  # results dir から task 別の不出来（failures.csv / failures.md）を出す
	python3 src/pipeline/failures_report.py --results $(RESULTS) $(FAILURES_ARGS)

scoreboard-table:                            # 測定値と公式スコアを同じ表で見る
	python3 src/pipeline/scoreboard.py --table

push-measure-notebook: push-code measure-notebook
	kaggle kernels push -p kernels/measure --no-run

# CPU probe kernel: GPU quota を 1 分も使わずに base image の Python / preinstall / wheel tag / mount を実測する
kernel-ipynb-probe:
	python3 src/pipeline/make_kernel_notebook.py --source kernels/probe/gemma4_env_probe.py \
	  --title gemma4-env-probe --code-slug $(notdir $(CODE_DATASET)) \
	  --metadata kernels/probe/kernel-metadata.json
	python3 -c "import nbformat;nbformat.validate(nbformat.read('kernels/probe/gemma4_env_probe.ipynb',as_version=4));print('[kernel-ipynb-probe] nbformat validate OK')"

probe: kernel-ipynb-probe push-code
	$(KAGGLE) kernels push -p kernels/probe

probe-out:
	@mkdir -p results
	$(KAGGLE) kernels output $(USERNAME)/gemma4-env-probe -p results/probe
	@echo "[probe] results/probe/ を見る: probe_env.json と gemma4-env-probe.log"

pipeline-run: set-variant kernel-ipynb push-code push-submission
	$(KAGGLE) kernels push -p kernels/eval

pipeline-run-save: set-variant kernel-ipynb push-code push-submission
	$(KAGGLE) kernels push -p kernels/eval --no-run

push-train: push-code
	$(KAGGLE) kernels push -p kernels/train

status:
	-$(KAGGLE) kernels status $(EVAL_KERNEL)
	-$(KAGGLE) kernels status $(TRAIN_KERNEL)

logs:
	-$(KAGGLE) kernels logs $(EVAL_KERNEL)

logs-follow:
	$(KAGGLE) kernels logs $(EVAL_KERNEL) --follow

# ログ全文を results/logs/<RUN>.json に落として、stage 行だけ人間が読める形にする（実行中でも使える）。
# 例: make logs-dump RUN=smoke LOG_STAGE="server eval"
LOG_STAGE ?= server eval
logs-dump:
	@mkdir -p results/logs
	$(KAGGLE) kernels logs $(EVAL_KERNEL) > results/logs/$(RUN).json
	@python3 src/pipeline/dump_kernel_log.py results/logs/$(RUN).json \
	  --stage $(LOG_STAGE) --out results/logs/$(RUN).log

out-eval:
	$(KAGGLE) kernels output $(EVAL_KERNEL) -p results/$(RUN) -o -q

out-train:
	$(KAGGLE) kernels output $(TRAIN_KERNEL) -p adapters_out -o -q

submit: submission.zip
	$(KAGGLE) competitions submit -c gemma-4-developer-agent -f submission.zip \
	  -m "gemma-4-e4b agent (prompt+config variant)"

submissions:
	$(KAGGLE) competitions submissions -c gemma-4-developer-agent

clean:
	rm -rf submission.zip .kaggle/dataset-code/src .kaggle/dataset-submission/prompts \
	       .kaggle/dataset-submission/configs .kaggle/dataset-submission/sub_agents \
	       .kaggle/dataset-submission/adapters .kaggle/dataset-submission/agent.yaml \
	       .kaggle/dataset-submission/eval_config.yaml
