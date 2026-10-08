#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
prism_revision=9a9394a895b96003ca842a6041cb28ac49a108f7
prism_source=${PRISM_SOURCE:-"$repo_root/tools/prism-source"}
command -v git >/dev/null
command -v cmake >/dev/null
command -v nvcc >/dev/null || { echo 'Нужен установленный CUDA toolkit с nvcc.' >&2; exit 1; }
if [[ ! -d "$prism_source/.git" ]]; then
  [[ ! -e "$prism_source" ]] || { echo 'Путь PRISM_SOURCE уже занят; выберите другой.' >&2; exit 1; }
  mkdir -p "$(dirname -- "$prism_source")"
  git clone --no-checkout https://github.com/PrismML-Eng/llama.cpp.git "$prism_source"
  git -C "$prism_source" checkout --detach "$prism_revision"
fi
[[ $(git -C "$prism_source" rev-parse HEAD) == "$prism_revision" ]] || {
  echo "Нужен commit $prism_revision. Существующее дерево автоматически не меняется." >&2; exit 1;
}
[[ -z $(git -C "$prism_source" status --porcelain) ]] || {
  echo 'Дерево Prism содержит изменения; сборка прекращена для сохранения версии.' >&2; exit 1;
}
cmake -S "$prism_source" -B "$prism_source/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$prism_source/build" --target llama-server llama-cli -j "${BUILD_JOBS:-8}"
echo "Готово: $prism_source/build/bin/llama-server"
