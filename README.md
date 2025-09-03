# 🎯 Enhanced Transcriber - ансамблевая система транскрипции

## 📖 Описание проекта

**Enhanced Transcriber** - это ансамблевая система транскрипции аудио в текст для русского языка.

### 🎯 Ключевые особенности:

- **🤖 Ансамбль из 2 моделей**: T-one (русскоязычная), Whisper Local
- **📊 Система оценки качества**: WER, CER, семантический анализ
- **🎵 Продвинутая обработка аудио**: шумоподавление, улучшение речи
- **🇷🇺 Оптимизация для русского языка**: pymorphy2, spaCy
- **🛒 E-commerce специализация**: терминология интернет-магазина \"Гипер Онлайн\"
- **🏗️ Модульная архитектура**: чистая архитектура, интерфейсы

## 📁 Поддерживаемые форматы

**Аудио:** WAV, MP3, FLAC, M4A, OGG, WEBM  
**Видео:** MP4, AVI, MOV, MKV, WEBM  
**Языки:** Русский (оптимизированный), Английский, и 50+ других

---


## 🔧 Архитектура проекта

```
PopovAndrew/
├── enhanced-transcriber/          # Основной проект
│   ├── core/                     # Ядро системы
│   │   └── interfaces/          # Интерфейсы
│   ├── providers/               # Провайдеры транскрипции
│   │   ├── whisper/            # Whisper модели
│   │   └── tone/               # T-one модель
│   └── services/               # Сервисы
│       ├── audio_processor.py   # Обработка аудио
│       ├── ensemble_service.py  # Ансамбль моделей
│       └── quality_assessor.py  # Оценка качества
└── Enhanced_Transcriber_STRICT_NO_FALLBACK.ipynb  # ⚡ ЕДИНСТВЕННАЯ РЕКОМЕНДУЕМАЯ ВЕРСИЯ
```



## 🎖️ Для разработчиков

### Локальная установка:
```bash
git clone https://github.com/Andrew821667/Giper.git
cd Giper/PopovAndrew/enhanced-transcriber
pip install -r requirements.txt
```

### CLI использование:
```bash
python main.py transcribe audio.wav --language ru
```

---

## 🆘 Поддержка

**Проблемы с ноутбуками:**
1. Убедитесь, что используете GPU runtime в Colab
2. Проверьте формат загружаемых файлов

---

## 📄 Лицензия

MIT License - используйте свободно в коммерческих и некоммерческих проектах.

---

## 👨‍💻 Автор

**Andrew Popov** - Enhanced Transcriber для \"Гипер Онлайн\"

**GitHub:** [Andrew821667](https://github.com/Andrew821667)
