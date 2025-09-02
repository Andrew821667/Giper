"""
Конфигурация для бенчмарков ASR моделей
"""

# Модели для тестирования и их настройки
MODELS_CONFIG = {
    "enhanced_transcriber": {
        "enabled": True,
        "description": "Наша ансамблевая система T-one + Whisper",
        "requires_gpu": True,
        "requires_models": ["tone", "whisper"]
    },
    "faster_whisper": {
        "enabled": True, 
        "description": "Optimized Whisper implementation",
        "requires_gpu": False,
        "install_cmd": "pip install faster-whisper"
    },
    "wav2vec2": {
        "enabled": True,
        "description": "Facebook Wav2Vec 2.0",
        "requires_gpu": False,
        "install_cmd": "pip install transformers torch"
    },
    "vosk": {
        "enabled": False,
        "description": "Lightweight offline ASR",
        "requires_gpu": False,
        "install_cmd": "pip install vosk",
        "model_download": "wget https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip"
    },
    "speechbrain": {
        "enabled": False,
        "description": "SpeechBrain toolkit",
        "requires_gpu": False,
        "install_cmd": "pip install speechbrain"
    },
    "nemo_asr": {
        "enabled": False,
        "description": "NVIDIA NeMo ASR",
        "requires_gpu": True,
        "install_cmd": "pip install nemo_toolkit"
    }
}

# Настройки тестирования
TEST_CONFIG = {
    "min_confidence_threshold": 0.5,
    "max_processing_time": 300,  # 5 minutes per file
    "required_accuracy_wer": 0.15,  # WER < 15% для прохождения
    "target_rtf": 0.5,  # Real-time factor < 0.5x желательно
}

# Пути и директории
PATHS = {
    "models_cache": "models/",
    "test_data": "test_data/",
    "results": "benchmarks/results/",
    "reports": "benchmarks/reports/"
}
