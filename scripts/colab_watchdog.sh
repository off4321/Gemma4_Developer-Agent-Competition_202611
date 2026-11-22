#!/usr/bin/env bash
# Colab session を生かし続け、 job が死んでたら起こす番人（この端末で回す）。
#
#   bash scripts/colab_watchdog.sh g4c fetch
#   bash scripts/colab_watchdog.sh g4c measure:colab12 [interval_s]
#
# 状態は /tmp/colab_watch.log （`tail -25 /tmp/colab_watch.log` で読む）。
# 実測の教訓（2026-11-21）:
#   - `colab status` だけ見ていると 15 分ほどで VM が預けられる -> heartbeat は exec で kernel を走らせる
#   - exec は "Connection was lost." を時々出す -> 同じコマンドの再試行で通る（この loop がやる）
#   - session が消えたら make_from_scratch: new -> upload repo -> ensure_repo -> upload creds -> prepare_env
#   - 長時間の仕事は VM 側 detached（colab_fetch.py / colab_measure.py）。watchdog は生存只看る
set -u
S=${1:?session name 必須}
JOB=${2:?fetch または measure:<variant>}
N=${3:-75}
W=/tmp/colab_watch.log
REPO=/workspace/Gemma4_Developer-Agent-Competition_202611
LOOP=0
cd "$REPO" || exit 1

say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$W"; }
variant_of() { local j=${1#measure:}; echo "${j%%:*}"; }

make_from_scratch() {
  say "session 作り直し: $S"
  timeout 300 colab new -s "$S" --gpu L4 >> "$W" 2>&1
  tar czf /tmp/repo.tgz src submission
  timeout 300 colab upload /tmp/repo.tgz /content/repo.tgz -s "$S" >> "$W" 2>&1
  timeout 250 colab exec -s "$S" -f src/pipeline/colab_ensure_repo.py --timeout 200 >> "$W" 2>&1
  timeout 300 colab upload "$HOME/.kaggle/credentials.json" /root/.kaggle/credentials.json -s "$S" >> "$W" 2>&1
  timeout 900 colab exec -s "$S" -f src/pipeline/colab_prepare_env.py --timeout 800 >> "$W" 2>&1
}

launch_fetch() {
  say "fetch 起動/再開（既存 file は skip される）"
  timeout 250 colab exec -s "$S" -f src/pipeline/colab_fetch.py \
    --env WITH_MODEL=1 --env TASK_LIMIT=12 --timeout 200 >> "$W" 2>&1
}

launch_measure() {
  local v=$(variant_of "$JOB")
  say "measure 起動/再開 variant=$v"
  timeout 250 colab exec -s "$S" -f src/pipeline/colab_launch_measure.py \
    --env VARIANT="$v" --timeout 200 >> "$W" 2>&1
}

fetch_done() { grep -q '"wheelhouse"' /tmp/watch_fetch.log 2>/dev/null; }

say "watch start session=$S job=$JOB interval=${N}s"
while :; do
  LOOP=$((LOOP + 1))
  st=$(timeout 90 colab status -s "$S" 2>&1)
  if echo "$st" | grep -qi "not found\|No active sessions"; then
    make_from_scratch
  fi

  out=$(timeout 90 colab exec -s "$S" -f src/pipeline/colab_kalive.py --timeout 60 2>&1)
  say "hb: $(echo "$out" | grep -E '^(KA|FETCH_RUNNING|PIPE_RUNNING|MEASURE_RUNNING|CREDS|LOG|DISK)' | tr '\n' ' ')"
  if ! echo "$out" | grep -q '^KA'; then
    say "hb 失敗: $(echo "$out" | tail -1 | cut -c1-70)"
    sleep "$N"; continue
  fi

  # session を作り直すと creds も CLI 修理も消える（実測で fetch が 1 個目で死んだ）。ここで見る。
  if echo "$out" | grep -q "CREDS=0"; then
    say "creds 補う"
    timeout 300 colab upload "$HOME/.kaggle/credentials.json" /root/.kaggle/credentials.json -s "$S" >> "$W" 2>&1
    timeout 250 colab exec -s "$S" -f src/pipeline/colab_fix_creds.py --timeout 200 >> "$W" 2>&1
  fi
  if echo "$out" | grep -q "KAGGLE_OK=0"; then
    say "kaggle CLI / package 修理 (prepare_env)"
    timeout 900 colab exec -s "$S" -f src/pipeline/colab_prepare_env.py --timeout 800 >> "$W" 2>&1
  fi

  timeout 90 colab download /content/bootstrap_all.log /tmp/watch_fetch.log -s "$S" >/dev/null 2>&1

  if [ "$JOB" = "fetch" ]; then
    if fetch_done; then say "FETCH 完了: $(grep '\"wheelhouse\"' /tmp/watch_fetch.log | tail -1 | cut -c1-150)"; break; fi
    echo "$out" | grep -q "FETCH_RUNNING=1" || launch_fetch
  else
    timeout 90 colab download /content/measure.log /tmp/watch_measure.log -s "$S" >/dev/null 2>&1
    if grep -qE '^\[measure .* DONE' /tmp/watch_measure.log 2>/dev/null; then
      say "MEASURE DONE: $(grep '^\[pipeline\]' /tmp/watch_measure.log | tail -2 | tr '\n' ' ' | cut -c1-220)"
      break
    fi
    if grep -qE '^\[measure .* FAILED' /tmp/watch_measure.log 2>/dev/null; then
      say "MEASURE FAILED: $(tail -3 /tmp/watch_measure.log | tr '\n' ' ' | cut -c1-220)"
      break
    fi
    fetch_done || { echo "$out" | grep -q "FETCH_RUNNING=1" || launch_fetch; }
    echo "$out" | grep -q "MEASURE_RUNNING=1" || launch_measure
  fi

  if [ "$LOOP" -ge 150 ]; then say "loop 150 超 -> 終了（手で見る）"; break; fi
  sleep "$N"
done
say "watch end loop=$LOOP"
