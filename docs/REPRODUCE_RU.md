# Воспроизводимость, экспорт и запуск

## Что уже проверено и что входит в пакет

Публикация включает 147 исходных research scripts,36 адаптированных core tools, отдельные launcher/verifier tools, лицензии и числовые evidence. `reference/scripts` — архив исходного кода с заменой host paths и GPU UUID; эти файлы **не являются готовыми launchers**. Их fixed paths и orchestration рассчитаны на исходный layout экспериментальных каталогов.

`scripts/ternary_core.py` принимает `VOL2_ROOT`, `VOL2_SOURCE`, явное `CUDA_VISIBLE_DEVICES`. Core и вспомогательные модули пригодны для дальнейшей адаптации. Исторические `train_v2`/`train_e020` всё ещё требуют соответствующих parent/data/cache manifests; перенос root не создаёт отсутствующие зависимости. Одной командой из пустого checkout нельзя точно пересоздать E020 или историю E004–E018. Нужны frozen исходная модель, сохранённые parent latent, mixed datasets, teacher caches и идентичный код/env. Они не выдаются за опубликованные, если не входят в release.

Свежий E022-Fixed128 и контрольный B1024 имеют отдельные native GGUF и base3 пакеты, подготовленные отдельно от Git. Model GGUF достаточен для inference, но не равен FP32 QAT master или optimizer checkpoint. Возобновление точного optimizer run из GGUF не поддерживается: требуется latent checkpoint. Master одного 27B run занимает около 100 ГиБ, поэтому в Git его нет.

## Сборка runtime

Linux, git, CMake, C++ toolchain и подходящий NVIDIA CUDA toolkit. Скрипт клонирует только публичный Prism fork и закрепляет commit; запускает CMake CUDA build. Он не останавливает процессы и не запускает модель. По умолчанию 8 CPU build jobs; можно задать BUILD_JOBS. CUDA_VISIBLE_DEVICES ограничивает inference, но не выбор архитектур компиляции.

```bash
bash scripts/build_runtime.sh
```

Если binary уже собран, укажите `PRISM_SERVER=/absolute/path/llama-server`; иначе используется `tools/prism-source/build/bin/llama-server`. Основной upstream может не поддерживать type142/PQ2 и H128 metadata. Версия runtime намеренно закреплена.

## Проверка файла и прямой native запуск

```bash
python3 scripts/verify_model.py /absolute/path/vol2-e022-fixed128-pq2.gguf
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh /absolute/path/vol2-e022-fixed128-pq2.gguf
```

CPU verifier проверяет размер/SHA256 и header: архитектуру qwen35, H128 identity,402PQ2_0/449F32. Он не создаёт GPU tensors. Сервер — loopback, порт 8080, один слот,context8192 и GPU offload всех слоёв. На shared host пользователь сам выбирает разрешённую карту. Можно задать PORT, SLOTS, CONTEXT; в llama.cpp context — общий бюджет сервера, поэтому для нескольких слотов умножайте нужный бюджет на число слотов и проверяйте VRAM. Большое число слотов/длинный context увеличивает KV memory.

Минимальный API-запрос:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vol2","messages":[{"role":"user","content":"Сколько будет 17 × 23?"}],"temperature":0,"max_tokens":256}'
```

Для исторического GSM8K надо воспроизвести отдельный benchmark protocol, включая thinking/template, parser и лимит. Пример curl не является benchmark.

## Зависимости Python

CPU verification evidence/GGUF и docs проверяются обычным Python3.12 без torch. Конвертация/обучение требует PyTorch CUDA, Transformers с Qwen3_5, accelerate, safetensors, numpy и Triton. Исторические версии приведены в `evidence/environment.json`; training snapshot pins — в requirements-training.txt. FLA использовалась из отдельного исходного дерева и фиксируется в этом evidence, если обнаружена. Сам training requirements файл не устанавливает CUDA driver и не гарантирует подходящий wheel для чужой системы.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-training.txt
```

Для gguf Python tools используется `PRISM_SOURCE` или `tools/prism-source/gguf-py` после сборки/клонирования. Тренер требует две явно выбранные GPU, каждая минимум 100 ГиБ свободно по guard; такая проверка не обещает, что любая длина пройдёт без OOM. Source path задаётся `VOL2_SOURCE`; скачайте/разместите лицензированную frozen исходную модель самостоятельно. Все запускные решения пользователь принимает под реальные свободные ресурсы.

## Экспорт своего замороженного кандидата

1. Latent checkpoint должен иметь `manifest.json` и safetensors weight/scales, а template base3 —config/tokenizer/manifest с теми же формами. Нельзя подавать неизвестный remote pickle как trusted optimizer state.
2. CPU exporter quantizes веса, проверяет каждый символ и записывает base3.
3. GGUF converter берёт совместимый metadata/template файл. Только schema/tokenizer/config; template weight tensors не копируются. Template обязан соответствовать Qwen35 text config.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/export_latent_cpu.py \
  /path/to/frozen-latent /path/to/base3-template /path/to/new-base3
CUDA_VISIBLE_DEVICES='' PRISM_SOURCE="$PWD/tools/prism-source" \
  .venv/bin/python scripts/export_prism_pq2.py \
  /path/to/new-base3 /path/to/new-model.gguf \
  --template /path/to/compatible-qwen35.gguf --name 'My Vol2 candidate'
```

Оба публикуемых кандидата уже проверены такой цепочкой. Новый CLI с явным template проверен на help/import; повторная 27B конвертация всех исторических кандидатов в процессе подготовки публикации не выполнялась. Для нового кандидата требуется собственный export audit и generation parity, а не только успешный write.

## Пересчёт evidence и тесты

```bash
python3 scripts/verify_evidence.py
python3 -m unittest discover -s tests -v
```

Эти проверки не используют GPU, не меняют experiment runs и не оценивают ответы заново. Они проверяют неизменность per-ID scores, согласованность aggregate и математику форматов. GitHub CI запускает те же CPU checks. Наличие зелёного CI не означает, что model quality воспроизведено с нуля.

## Отдельный пакет весов и GitHub

Скрипт `package_model.py` копирует model файл с reflink, где filesystem поддерживает его, без hardlink с рабочим экспериментом; проверяет checksum и добавляет model card/licences/паспорт. Пакет хранится в ignored releases. 7,25 ГБ нельзя помещать обычным blob в Git. Для публичных весов используйте model hub или другой large artifact hosting; адрес в artifact.json обновляется только после реальной загрузки.

```bash
python3 scripts/package_model.py /path/to/verified-model.gguf releases/e022-fixed128-pq2
```

Если GitHub CLI авторизован на своей машине:

```bash
bash scripts/publish_github.sh AubakirovArman/vol2-27b-ternary
```

Скрипт создаёт новый публичный репозиторий и пушит code/docs/evidence; требуется GitHub CLI с правами создания. Он не загружает 7,25 ГБ веса и не публикует статью на Astana Hub.

Для проверки контрольного E020 укажите `--manifest model/e020-b1024.json`. Primary manifest относится к E022-Fixed128.

## Публикация весов на Hugging Face

После `hf auth login` с правом записи запускается загрузка двух вариантов native GGUF и compact base3. Подготовка использует уже проверенные независимые release-копии; большие immutable файлы HF staging связаны hardlink только с этими release-копиями, не с исходными training артефактами.

```bash
python3 scripts/publish_huggingface.py --prepare-only
.venv/bin/python scripts/publish_huggingface.py --repo armanibadboy/WAL2-Ternary-27B
```

По умолчанию8 upload workers. Объём представленных файлов около26,25ГБ до Xet-дедупликации. После загрузки проверяются публичный доступ, inventory и точные SHA256 обоих GGUF в зафиксированном remote commit. Изменённые URL добавляются только после успешной проверки.
