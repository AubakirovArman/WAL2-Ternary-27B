# Где нужна помощь исследователей

Текущая задача — вернуть устойчивое длинное reasoning при тернарных крупных матрицах. У нас есть полный реализованный путь преобразования, QAT и native inference, но AIME25 и перенос математических навыков остаются далеко от Bonsai. Количество примеров и число шагов уже увеличивались без монотонного роста качества.

1. **Корректность рассуждения и завершение.** Разделить final-answer score, proof validity, output truncation, repetition и calibration; построить независимые procedural families и новый held-out test. Не оптимизировать только teacher CE или длину ответа.
2. **Scale/rotation QAT.** Измерить переходы тритов и представимых BF16-scales, проверить fp32-shadow/error-feedback или иной parameterization, затем H128/H1024 и signs при сохранении корректной функции до округления. Гипотезы проверять раздельно.
3. **On-policy дистилляция.** Подобрать rollout режим, teacher probability signal, смешивание replay и групповой reward; низкая доля полезных групп E021 мешает обучению. Увеличение 16 шагов без проверки objective недостаточно.
4. **Сигнал внутренних представлений.** После базовых ablations рассмотреть feature/attention distillation отдельных слоёв, inspired by [EdgeRazor](https://arxiv.org/html/2605.04062), с отдельной оценкой memory и совместимости GDN. Этот метод пока не реализован у нас.
5. **Скорость обучения.** Реальный pipeline overlap, корректный split выходной головы, chunk/checkpoint/sequence-length sweeps; измерять полные optimizer steps и quality drift. Разработка low-bit backward не подменяется inference kernels.
6. **Упаковка.** Более плотный base3 native kernel с честным bandwidth benchmark против PQ2. Сейчас 1,74-битный storage существует, но быстрый runtime использует 2,125-bit blocks.

Для каждого предложения нужен bounded pilot от зафиксированного B1024, одинаковые данные и seed, teacher/cache provenance, график CE вместе с генерационными показателями и откат при ухудшении. На GPU2/6 другие пользовательские процессы не должны останавливаться автоматически. Контакт: владелец будущего GitHub-репозитория AubakirovArman; после публикации — GitHub Issues. Команда не обещает заранее достижение уровня Bonsai.

Исходные работы: [ParetoQ](https://arxiv.org/html/2502.02631), [QAOPD](https://arxiv.org/html/2609.26708), [PrismML runtime](https://github.com/PrismML-Eng/llama.cpp), [Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf). Это источники для изучения, а не утверждение, что мы воспроизвели их обучение целиком.
