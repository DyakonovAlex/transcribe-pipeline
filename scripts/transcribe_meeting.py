#!/usr/bin/env python3
"""
Автоматическая транскрибация встречи и пост-обработка через Ollama.
Использование: python3 transcribe_meeting.py /путь/к/аудио.m4a [--no-ollama]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import yaml
from colorama import Fore, Style, init

# Инициализация colorama для Windows (если нужно)
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Логирование с цветами по уровням."""
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def load_config(config_path):
    """Загрузка конфигурации из YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    # Раскрываем ~
    for key in ['whisper_cli', 'model_path', 'output_dir', 'notes_dir']:
        if key in config:
            config[key] = os.path.expanduser(config[key])
    if 'ollama' in config and 'prompt_template' in config['ollama']:
        config['ollama']['prompt_template'] = os.path.expanduser(config['ollama']['prompt_template'])
    return config


def validate_config(config, require_ollama: bool) -> None:
    """Проверяет наличие обязательных ключей и валидность путей в конфигурации."""
    missing = []
    errors = []

    required_root_keys = ['whisper_cli', 'model_path', 'whisper_args', 'output_dir', 'notes_dir', 'placeholder']
    for key in required_root_keys:
        if key not in config:
            missing.append(key)

    required_whisper_args = ['language', 'threads', 'beam_size', 'best_of', 'temperature', 'max_len', 'no_timestamps']
    if 'whisper_args' in config:
        if not isinstance(config['whisper_args'], dict):
            errors.append("whisper_args должен быть объектом (словарём)")
        else:
            for key in required_whisper_args:
                if key not in config['whisper_args']:
                    missing.append(f"whisper_args.{key}")

    if require_ollama:
        if 'ollama' not in config:
            missing.append('ollama')
        else:
            for sub in ['url', 'model', 'prompt_template']:
                if sub not in config['ollama']:
                    missing.append(f"ollama.{sub}")

    if missing:
        missing_str = ', '.join(missing)
        errors.append(f"Конфигурация неполная. Отсутствуют ключи: {missing_str}")

    # Если есть структурные ошибки, не продолжаем к валидации путей
    if errors:
        raise SystemExit('\n'.join(errors))

    whisper_cli = Path(config['whisper_cli']).expanduser()
    if not whisper_cli.exists():
        errors.append(f"Файл whisper_cli не найден: {whisper_cli}")
    elif not whisper_cli.is_file():
        errors.append(f"Путь whisper_cli должен указывать на файл: {whisper_cli}")
    elif not os.access(str(whisper_cli), os.X_OK):
        errors.append(f"whisper_cli не является исполняемым файлом: {whisper_cli}")

    model_path = Path(config['model_path']).expanduser()
    if not model_path.exists() or not model_path.is_file():
        errors.append(f"Файл model_path не найден: {model_path}")

    notes_dir = Path(config['notes_dir']).expanduser()
    if not notes_dir.exists():
        errors.append(f"Директория notes_dir не найдена: {notes_dir}")
    elif not notes_dir.is_dir():
        errors.append(f"Путь notes_dir должен указывать на директорию: {notes_dir}")

    placeholder = config.get('placeholder')
    if not isinstance(placeholder, str) or not placeholder.strip():
        errors.append("placeholder должен быть непустой строкой")

    if require_ollama:
        prompt_template = Path(config['ollama']['prompt_template']).expanduser()
        if not prompt_template.exists() or not prompt_template.is_file():
            errors.append(f"Файл ollama.prompt_template не найден: {prompt_template}")

        if 'reasoning_effort' in config['ollama']:
            reasoning_effort = config['ollama']['reasoning_effort']
            allowed = {'low', 'medium', 'high'}
            if reasoning_effort not in allowed:
                errors.append(
                    f"ollama.reasoning_effort должен быть одним из: {', '.join(sorted(allowed))}"
                )

    if errors:
        raise SystemExit('\n'.join(errors))


def run_command(cmd, description, logger):
    """Запуск внешней команды с логированием."""
    logger.debug(f"Выполняется: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"stderr: {result.stderr}")
        logger.info(f"{description} выполнено успешно")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} завершилось с ошибкой: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise


def convert_audio(input_file, output_wav, logger):
    """Конвертация через ffmpeg."""
    cmd = ['ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', output_wav, '-y']
    run_command(cmd, "Конвертация аудио", logger)


def transcribe_audio(wav_file, config, logger):
    """Запуск whisper.cpp."""
    args = config['whisper_args']
    cmd = [
        config['whisper_cli'],
        '-m', config['model_path'],
        '-f', wav_file,
        '-l', args['language'],
        '-t', str(args['threads']),
        '--beam-size', str(args['beam_size']),
        '--best-of', str(args['best_of']),
        '--temperature', str(args['temperature']),
        '--max-len', str(args['max_len']),
        '-otxt'
    ]
    if args['no_timestamps']:
        cmd.append('--no-timestamps')
    run_command(cmd, "Транскрибация", logger)


def read_transcript(wav_file):
    """Читает транскрипт из файла .txt, созданного whisper."""
    transcript_file = wav_file + '.txt'
    # Whisper обычно пишет UTF-8, но на всякий случай читаем байты и
    # декодируем с защитой от редких невалидных последовательностей.
    with open(transcript_file, 'rb') as f:
        data = f.read()
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        # Фолбэк: максимально сохранить содержимое, заменяя битые символы.
        return data.decode('utf-8', errors='replace')


def call_ollama(prompt_text, config, logger):
    """Отправка запроса к Ollama и возврат ответа."""
    payload = {
        "model": config['ollama']['model'],
        "prompt": prompt_text,
        "stream": False
    }
    if 'reasoning_effort' in config['ollama']:
        payload["reasoning_effort"] = config['ollama']['reasoning_effort']

    logger.info(f"Отправка запроса к Ollama (модель {config['ollama']['model']})...")
    try:
        response = requests.post(config['ollama']['url'], json=payload, timeout=300)
        if response.status_code == 400 and "reasoning_effort" in payload:
            logger.warning(
                "Ollama отклонил reasoning_effort (HTTP 400). Повторяем запрос без reasoning_effort."
            )
            payload.pop("reasoning_effort", None)
            response = requests.post(config['ollama']['url'], json=payload, timeout=300)

        response.raise_for_status()
        data = response.json()
        result = data.get('response', '').strip()
        if not result:
            logger.warning("Ollama вернул пустой ответ")
        else:
            logger.info("Ollama обработал запрос")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при обращении к Ollama: {e}")
        raise


def postprocess_with_ollama(transcript, config, logger):
    """Загружает шаблон, заменяет {{transcription}}, вызывает Ollama."""
    prompt_template_path = config['ollama']['prompt_template']
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    full_prompt = template.replace('{{transcription}}', transcript)
    return call_ollama(full_prompt, config, logger)


def update_markdown(markdown_path, new_content, placeholder, logger):
    """Заменяет {{input}} в файле markdown на new_content."""
    if not markdown_path.exists():
        logger.error(f"Целевой файл {markdown_path} не существует!")
        sys.exit(1)

    with open(markdown_path, 'r', encoding='utf-8') as f:
        original = f.read()

    if placeholder not in original:
        logger.warning(f"Плейсхолдер '{placeholder}' не найден в {markdown_path}. Файл не изменён.")
        return

    updated = original.replace(placeholder, new_content)
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(updated)
    logger.info(f"Файл {markdown_path} обновлён.")


def main():
    parser = argparse.ArgumentParser(description="Транскрибация встречи и вставка в markdown")
    parser.add_argument('input_audio', help="Путь к исходному аудиофайлу (m4a, mp4, и т.д.)")
    parser.add_argument('--no-ollama', action='store_true', help="Пропустить пост-обработку и вставку")
    parser.add_argument('--config', default='config.yaml', help="Путь к конфигурационному файлу")
    args = parser.parse_args()

    # Загрузка конфигурации
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Конфигурационный файл {config_path} не найден!", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    # Логгер и валидация конфига
    logger = setup_logging(level=getattr(logging, config.get('log_level', 'INFO')))
    validate_config(config, require_ollama=not args.no_ollama)

    input_file = Path(args.input_audio).expanduser().resolve()
    if not input_file.exists():
        logger.error(f"Исходный файл {input_file} не найден")
        sys.exit(1)

    # Определяем имена файлов
    base_name = input_file.stem
    # Корневая директория для временных файлов
    output_root = Path(config['output_dir']).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Уникальная подпапка для каждого запуска (на случай параллельных запусков)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{base_name}_", dir=str(output_root)))

    wav_path = temp_dir / f"{base_name}.wav"
    notes_dir = Path(config['notes_dir']).expanduser().resolve()
    markdown_path = notes_dir / f"{base_name}.md"
    placeholder = config['placeholder']

    try:
        # Шаг 1: конвертация
        logger.info(f"Обработка файла: {input_file.name}")
        convert_audio(str(input_file), str(wav_path), logger)

        # Шаг 2: транскрибация
        transcribe_audio(str(wav_path), config, logger)

        # Шаг 3: чтение транскрипта
        transcript = read_transcript(str(wav_path))
        logger.info(f"Транскрипт получен (длина: {len(transcript)} символов)")

        # Шаг 4: пост-обработка (если не отключена)
        if not args.no_ollama:
            processed = postprocess_with_ollama(transcript, config, logger)
            if processed:
                # Шаг 5: обновление markdown
                update_markdown(markdown_path, processed, placeholder, logger)
            else:
                logger.warning("Ollama не вернул результат. Markdown не изменён.")
        else:
            logger.info("Пропуск пост-обработки (--no-ollama)")

        logger.info("✅ Готово!")

    except Exception as e:
        logger.exception("Критическая ошибка")
        sys.exit(1)
    finally:
        # Очистка временных файлов директории запуска
        if 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()