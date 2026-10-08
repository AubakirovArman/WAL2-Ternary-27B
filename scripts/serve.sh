#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# -eq 1 && -f "$1" ]] || { echo "Использование: CUDA_VISIBLE_DEVICES=<разрешённая карта> bash $0 /path/model.gguf" >&2; exit 2; }
[[ -n ${CUDA_VISIBLE_DEVICES:-} && $CUDA_VISIBLE_DEVICES != *,* ]] || {
  echo 'Выберите ровно одну разрешённую GPU через CUDA_VISIBLE_DEVICES.' >&2; exit 2;
}
prism_server=${PRISM_SERVER:-"${PRISM_SOURCE:-$repo_root/tools/prism-source}/build/bin/llama-server"}
[[ -x "$prism_server" ]] || { echo "Не найден runtime: $prism_server. Сначала build_runtime.sh." >&2; exit 1; }
for value in "${SLOTS:-1}" "${CONTEXT:-8192}" "${PORT:-8080}"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'SLOTS, CONTEXT и PORT должны быть положительными целыми.' >&2; exit 2; }
done
python3 "$repo_root/scripts/verify_model.py" "$1" --layout-only
echo "Прямой PQ2 запуск: $1; GPU=$CUDA_VISIBLE_DEVICES; слотов=${SLOTS:-1}; общий контекст=${CONTEXT:-8192}."
exec "$prism_server" --model "$1" --alias vol2 --host 127.0.0.1 --port "${PORT:-8080}" \
  --n-gpu-layers 999 --parallel "${SLOTS:-1}" --ctx-size "${CONTEXT:-8192}"
