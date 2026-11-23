#!/usr/bin/env bash
# 경로 계획 콘솔 실행. 일반 터미널에서 실행한다 (GPU 사용). 끝낼 때는 이 터미널에서 Ctrl+C.
# 같은 GPU를 쓰는 다른 모델 프로그램이 돌고 있으면 먼저 끈다 (12 GB GPU에 Qwen3-VL-4B 약 8.8 GB).
set -u
cd "$(dirname "$0")"
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
URL=http://127.0.0.1:8765
if curl -s -m 1 "$URL/api/status" >/dev/null 2>&1; then
  echo "이미 실행 중입니다: $URL"
  xdg-open "$URL" >/dev/null 2>&1 || true
  exit 0
fi
( sleep 3; xdg-open "$URL" >/dev/null 2>&1 || echo "브라우저에서 $URL 을 여세요" ) &
exec "$PY" -B server.py
