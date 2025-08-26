# 🎯 Enhanced Transcriber - Качество 95%+

**Enhanced Transcriber** - система высококачественной транскрипции аудио с целевым качеством **95%+** для русского языка. Специально оптимизирована для интернет-магазина "Гипер Онлайн" с поддержкой e-commerce терминологии.

## 📱 **Быстрый запуск в Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andrew821667/Giper/blob/main/PopovAndrew/enhanced-transcriber/Enhanced_Transcriber_Colab.ipynb)

**🔧 Если кнопка не работает, используйте прямую ссылку:**
```
https://colab.research.google.com/github/Andrew821667/Giper/blob/main/PopovAndrew/enhanced-transcriber/Enhanced_Transcriber_Colab.ipynb
```

**📋 Или откройте Colab и загрузите из GitHub:**
1. Откройте https://colab.research.google.com
2. Вкладка "GitHub" → введите `Andrew821667/Giper`
3. Выберите ветку `main` 
4. Найдите `PopovAndrew/enhanced-transcriber/Enhanced_Transcriber_Colab.ipynb`

**🎯 Протестируйте качество 95%+ прямо в браузере без установки!**

## ✨ Ключевые особенности

- 🎯 **Целевое качество 95%+** через ensemble подход
- 🤖 **Мультимодельная архитектура**: T-one + Whisper + OpenAI
- 🛒 **E-commerce специализация** для "Гипер Онлайн"
- 🇷🇺 **Русский язык приоритет** с T-one моделью
- 🔄 **Weighted consensus** для максимальной точности
- 🎛️ **Clean Architecture** с dependency injection
- ⚡ **Асинхронная обработка** для производительности

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Базовые зависимости
pip install -r requirements.txt

# T-one модель (специально для русского языка)
pip install git+https://github.com/voicekit-team/T-one.git

# FFmpeg для аудио (если не установлен)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### 2. Минимальный пример

```python
import asyncio
from example_usage import EnhancedTranscriberDemo

async def main():
    # Инициализация
    demo = EnhancedTranscriberDemo()
    
    # Настройка моделей (T-one + Whisper Local + OpenAI опционально)
    await demo.setup_models(openai_api_key="sk-your-key")  # или None
    
    # Транскрипция с качеством 95%+
    result = await demo.transcribe_audio(
        audio_file="your_audio.wav",
        language="ru",
        domain="ecommerce",
        use_ensemble=True
    )
    
    # Красивый вывод результата
    demo.print_transcription_result(result)

asyncio.run(main())
```

## 🔧 Доступные модели

| Модель | Язык | Качество | Скорость | Специализация |
|--------|------|----------|----------|---------------|
| **T-one** | RU | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Русский + телефония |
| Whisper Local | Multi | ⭐⭐⭐⭐ | ⚡⚡ | Общий |
| Whisper OpenAI | Multi | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Облачный API |

### Рекомендуемые комбинации

**Максимальное качество (95%+):**
```python
models = [ToneTranscriber(), WhisperLocalTranscriber("medium"), WhisperOpenAITranscriber()]
```

**Баланс качества и скорости:**
```python
models = [ToneTranscriber(), WhisperLocalTranscriber("base")]
```

## 🛒 E-commerce специализация

Автоматические исправления для "Гипер Онлайн":

```python
# Исправления e-commerce терминов
corrections = {
    "закас/зокас" → "заказ",
    "аплата/оплото" → "оплата", 
    "доствка/дастафка" → "доставка",
    "возрат/вазврат" → "возврат",
    "тавар/товорр" → "товар",
    "скитка/скидко" → "скидка",
    "карзина/корзино" → "корзина"
}
```

## 🎯 Цель: Качество транскрипции 95%+ для масштабного проекта Гипер Онлайн