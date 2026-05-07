# transcribe-pipeline

Минимальный пайплайн для транскрибации встречи через `whisper.cpp` и (опционально) пост-обработки через LLM (`Ollama` или `Gemini CLI`).

## Быстрый запуск

1. Проверить и при необходимости поправить пути в `config.yaml`.

2. Сделать скрипты запуска исполняемыми:

```bash
chmod +x run_transcribe.sh run_debug_compare.sh
```

3. Запустить (единственный аргумент — путь к аудио):

```bash
./run_transcribe.sh "/путь/к/встрече.m4a"
```

## Запуск без bash-обёртки (вручную)

1. Создать/активировать виртуальное окружение и установить зависимости:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

2. Запустить:

```bash
python scripts/transcribe_meeting.py "/путь/к/встрече.m4a"
```

Если нужно только получить транскрипт без LLM-постобработки:

```bash
python scripts/transcribe_meeting.py "/путь/к/встрече.m4a" --no-llm
```

## Настройки LLM

Выбор провайдера в `config.yaml`:

```yaml
llm_provider: "ollama" # ollama | gemini_cli
```

### Ollama

В `config.yaml` можно задать уровень "thinking":

```yaml
ollama:
  url: "http://localhost:11434/api/generate"
  model: "gpt-oss:120b-cloud"
  reasoning_effort: "high" # low | medium | high (опционально)
  prompt_template: "~/dyak/transcribe-pipeline/prompts/meeting_summary.txt"
```

Поведение при несовместимости:
- если модель/сервер не поддерживает `reasoning_effort` и возвращает `HTTP 400`,
  скрипт автоматически повторяет запрос **без** этого поля;
- в логах при этом будет предупреждение, после чего пайплайн продолжит работу.

### Gemini CLI

```yaml
gemini_cli:
  binary: "gemini"
  model: "gemini-2.5-pro"
  prompt_template: "~/dyak/transcribe-pipeline/prompts/meeting_summary.txt"
  timeout: 300
  args: []
```

`args` — дополнительные аргументы для `gemini` CLI (опционально).

## Отладочный режим пайплайна

В `scripts/transcribe_meeting.py` доступны этапы:
- `convert`
- `transcribe`
- `llm`
- `update`

### Через `run_debug_compare.sh` (рекомендуется)

Скрипт поднимает `venv`, ставит зависимости и вызывает Python с типичными флагами для отладки.

- По умолчанию: `--cleanup never`, фиксированный `--temp-dir` (если не задан: `/tmp/transcribe_debug_<имя_файла_без_расширения>`).
- Сравнение моделей Ollama за один полный прогон (конвертация → транскрибация → оба запроса к Ollama):

```bash
./run_debug_compare.sh --compare-ollama "gpt-oss:20b,gpt-oss:120b-cloud" "/путь/к/встрече.m4a"
```

- Только этап Ollama+update, если WAV и транскрипт уже лежат в том же `--temp-dir` после предыдущего запуска:

```bash
./run_debug_compare.sh --start-stage llm --end-stage update \
  --temp-dir "/tmp/transcribe_debug_моя_встреча" \
  --compare-ollama "model-a,model-b" \
  "/путь/к/той_же_встрече.m4a"
```

- Сравнение двух бинарников `whisper-cli` (этап до `transcribe`):

```bash
./run_debug_compare.sh --end-stage transcribe \
  --compare-whisper "/path/whisper-a,/path/whisper-b" \
  "/путь/к/встрече.m4a"
```

- Всё, что после `--`, передаётся в `transcribe_meeting.py` без изменений (например `--no-llm` или свой `--config`):

```bash
./run_debug_compare.sh "/путь/к/встрече.m4a" -- --no-llm
```

Справка по опциям: `./run_debug_compare.sh --help`.

Для сравнения моделей при наличии плейсхолдера в заметке создаются файлы рядом с основным: `<meeting>__<model>.md`. Если плейсхолдер уже заполнен, саммари пишутся в `--temp-dir` как `<meeting>__<model>.summary.md`.

### Вручную через Python

Запуск диапазона этапов:

```bash
python scripts/transcribe_meeting.py "/путь/к/встрече.m4a" --start-stage transcribe --end-stage llm --temp-dir "/tmp/my-debug-run"
```

Сравнение `whisper-cli` на одном и том же этапе `transcribe`:

```bash
python scripts/transcribe_meeting.py "/путь/к/встрече.m4a" --end-stage transcribe --temp-dir "/tmp/my-debug-run" --compare-whisper-cli "/path/whisper-a,/path/whisper-b"
```

Сравнение моделей Ollama:

```bash
python scripts/transcribe_meeting.py "/путь/к/встрече.m4a" --start-stage llm --end-stage update --temp-dir "/tmp/my-debug-run" --compare-ollama-model "gpt-oss:20b,gpt-oss:120b-cloud"
```

`--compare-ollama-model` работает только при `llm_provider: ollama`.

Режим очистки временных файлов:
- `--cleanup always` (по умолчанию в обычном запуске без `--temp-dir`),
- `--cleanup on-success`,
- `--cleanup never` (по умолчанию в `run_debug_compare.sh`, пока не переопределите `--cleanup`).