# Vol2 Ternary 27B

A research project converting Qwen3.8-27B-FP8 into ternary weights, recovering quality with QAT/distillation, and exporting directly runnable PQ2 GGUF. It uses the open Prism/llama.cpp runtime; its kernels are upstream work, and no Bonsai weight tensors are included.

The text model has 26.896B parameters, 99.9024% in 402 ternary matrices. Compact base3 tensors occupy 5.846GB (1.739 bits per language parameter); the tokenizer/config directory is about 5.869GB (1.746 bits). Native PQ2 uses 2.125 bits per ternary weight with scale; the 7.253GB GGUF averages 2.157 bits per language parameter including all metadata/exceptions.

Last completed full GSM8K: **E017 1117/1319 (84.69%)**, locally tested **Bonsai 1275/1319 (96.66%)**, identical greedy/medium/2048-token protocol. These numbers do not describe later E020/E022 weights.

Latest completed 136-task development evaluation: **E022-Fixed128 69/136 (50.74%), AIME 0/30**. Reference **E020-B1024 64/136 (47.06%), AIME 2/30**. Both are preserved because improvement is not uniform. The 9049-task broad campaign was interrupted; only 1569 matched partial tasks are reported separately. All four E022 generation evaluations are complete: joint64 67/136,AIME1/30; joint12862/136,AIME0/30.

Code, numerical evidence and detailed Russian documentation are included. Public weights: [Hugging Face](https://huggingface.co/armanibadboy/WAL2-Ternary-27B); code: [GitHub](https://github.com/AubakirovArman/WAL2-Ternary-27B). Weights are separate from Git. Exact historical training replay also requires original datasets, teacher caches and latent checkpoints, which are not bundled. License: Apache-2.0 for our code/model derivative; Qwen Apache-2.0 and Prism runtime MIT retain attribution.

See [Russian README](README.md), [model card](MODEL_CARD.md), [benchmarks](docs/BENCHMARKS_RU.md), [method](docs/METHOD_RU.md), [experiment history](docs/EXPERIMENTS_RU.md), and [reproduction](docs/REPRODUCE_RU.md).
