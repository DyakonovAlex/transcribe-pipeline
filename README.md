# transcribe-pipeline

Минимальный пайплайн для транскрибации встречи через `whisper.cpp` и (опционально) пост-обработки через Ollama.

## Быстрый запуск

1. Проверить и при необходимости поправить пути в `config.yaml`.

2. Сделать скрипт запуска исполняемым:

```bash
chmod +x run_transcribe.sh
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

Если нужно только получить транскрипт без Ollama:

```bash
python scripts/transcribe_meeting.py "/путь/к/встрече.m4a" --no-ollama
```

## Настройки Ollama

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