"""
Пример использования Enhanced Transcriber для качества 95%+
Example usage of Enhanced Transcriber for 95%+ quality
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Импорты нашего проекта
from providers.tone import ToneTranscriber
from providers.whisper import WhisperLocalTranscriber, WhisperOpenAITranscriber
from services.ensemble_service import EnsembleTranscriptionService
from core.models.transcription_result import TranscriptionResult

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_transcriber.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class EnhancedTranscriberDemo:
    """
    Демонстрация Enhanced Transcriber для достижения качества 95%+
    """
    
    def __init__(self):
        """Инициализация демо"""
        self.models = []
        self.ensemble_service = None
    
    async def setup_models(self, openai_api_key: str = None) -> List[Dict[str, Any]]:
        """
        Настройка моделей для ensemble
        
        Args:
            openai_api_key: API ключ OpenAI (опционально)
            
        Returns:
            List: Информация о доступных моделях
        """
        model_info = []
        
        # 1. T-one (лучший для русского языка)
        try:
            tone_model = ToneTranscriber("voicekit/tone-ru")
            self.models.append(tone_model)
            model_info.append({
                "name": "T-one Russian",
                "status": "available",
                "specialization": "Russian language, telephony domain"
            })
            logger.info("✅ T-one model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ T-one model not available: {e}")
            model_info.append({
                "name": "T-one Russian", 
                "status": "unavailable",
                "reason": str(e)
            })
        
        # 2. Whisper Local (базовая модель для консенсуса)
        try:
            whisper_local = WhisperLocalTranscriber("base", device="auto")
            self.models.append(whisper_local)
            model_info.append({
                "name": "Whisper Local Base",
                "status": "available", 
                "specialization": "Multilingual, general domain"
            })
            logger.info("✅ Whisper Local model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Whisper Local model not available: {e}")
            model_info.append({
                "name": "Whisper Local Base",
                "status": "unavailable",
                "reason": str(e)
            })
        
        # 3. OpenAI Whisper (если есть API ключ)
        if openai_api_key:
            try:
                whisper_openai = WhisperOpenAITranscriber(openai_api_key)
                self.models.append(whisper_openai)
                model_info.append({
                    "name": "OpenAI Whisper",
                    "status": "available",
                    "specialization": "Cloud-based, high quality"
                })
                logger.info("✅ OpenAI Whisper API available")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI Whisper not available: {e}")
                model_info.append({
                    "name": "OpenAI Whisper",
                    "status": "unavailable", 
                    "reason": str(e)
                })
        else:
            model_info.append({
                "name": "OpenAI Whisper",
                "status": "skipped",
                "reason": "No API key provided"
            })
        
        # Инициализация ensemble сервиса
        if len(self.models) >= 2:
            self.ensemble_service = EnsembleTranscriptionService(
                models=self.models,
                target_quality_threshold=0.95  # 95%
            )
            logger.info(f"🚀 Ensemble service initialized with {len(self.models)} models")
        elif len(self.models) == 1:
            logger.info("ℹ️ Only one model available - ensemble disabled")
        else:
            logger.error("❌ No models available for transcription")
        
        return model_info
    
    async def transcribe_audio(
        self, 
        audio_file: str,
        language: str = "ru",
        domain: str = "ecommerce",
        use_ensemble: bool = True
    ) -> TranscriptionResult:
        """
        Транскрипция аудио файла с максимальным качеством
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык
            domain: Домен для постобработки
            use_ensemble: Использовать ensemble (если доступен)
            
        Returns:
            TranscriptionResult: Результат транскрипции
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Аудио файл не найден: {audio_file}")
        
        logger.info(f"🎵 Начинаем транскрипцию: {Path(audio_file).name}")
        logger.info(f"📊 Параметры: язык={language}, домен={domain}, ensemble={use_ensemble}")
        
        # Ensemble транскрипция
        if use_ensemble and self.ensemble_service:
            logger.info("🔄 Запуск ensemble транскрипции для качества 95%+")
            result = await self.ensemble_service.transcribe_with_quality_target(
                audio_file=audio_file,
                language=language,
                domain=domain,
                max_iterations=3
            )
            
        # Одиночная модель (fallback)
        elif self.models:
            logger.info(f"🔧 Использование одиночной модели: {self.models[0].model_name}")
            result = await self.models[0].transcribe(audio_file, language)
            
        else:
            raise RuntimeError("Ни одной модели не доступно для транскрипции")
        
        return result
    
    def print_transcription_result(self, result: TranscriptionResult):
        """Красивый вывод результата транскрипции"""
        print("\\n" + "="*80)
        print("📝 РЕЗУЛЬТАТ ТРАНСКРИПЦИИ")
        print("="*80)
        
        # Основная информация
        print(f"🎯 Модель: {result.model_used}")
        print(f"🌍 Язык: {result.language_detected}")
        print(f"⏱️ Время обработки: {result.processing_time:.2f} сек")
        print(f"📊 Уверенность: {result.confidence:.1%}")
        print(f"📁 Размер файла: {result.file_size / 1024 / 1024:.2f} МБ")
        
        if result.audio_duration:
            print(f"🕐 Длительность аудио: {result.audio_duration:.1f} сек")
        
        # Информация о качестве
        if result.quality_metrics:
            print(f"\\n🏆 ОЦЕНКА КАЧЕСТВА:")
            print(f"   Общий счет: {result.quality_metrics.overall_score:.1%}")
            print(f"   Уровень: {result.quality_metrics.quality_level.value}")
            if result.quality_metrics.word_error_rate is not None:
                print(f"   WER: {result.quality_metrics.word_error_rate:.1%}")
            if result.quality_metrics.improvement_suggestions:
                print(f"   Рекомендации: {', '.join(result.quality_metrics.improvement_suggestions)}")
        
        # Метаданные ensemble
        if result.provider_metadata and 'ensemble_size' in result.provider_metadata:
            print(f"\\n🤖 ENSEMBLE ИНФОРМАЦИЯ:")
            print(f"   Количество моделей: {result.provider_metadata['ensemble_size']}")
            print(f"   Модели: {', '.join(result.provider_metadata['models_used'])}")
            print(f"   Средняя уверенность: {result.provider_metadata.get('avg_confidence', 0):.1%}")
        
        # Текст транскрипции
        print(f"\\n💬 ТЕКСТ:")
        print("-" * 80)
        print(result.text)
        print("-" * 80)
        
        # Статус
        status_emoji = "✅" if result.status.value == "completed" else "❌"
        print(f"\\n{status_emoji} Статус: {result.status.value}")
        
        if result.error_message:
            print(f"⚠️ Ошибка: {result.error_message}")
        
        print("=" * 80 + "\\n")


async def main():
    """Главная функция демонстрации"""
    print("🚀 Enhanced Transcriber Demo - Качество 95%+")
    print("=" * 60)
    
    # Инициализация демо
    demo = EnhancedTranscriberDemo()
    
    # Настройка моделей
    print("\\n🔧 Настройка моделей...")
    
    # Здесь можно указать OpenAI API ключ для дополнительной модели
    # openai_key = "sk-your-openai-key-here"
    openai_key = None  # Пока без OpenAI
    
    model_info = await demo.setup_models(openai_key)
    
    # Вывод информации о моделях
    print("\\n📋 Доступные модели:")
    for info in model_info:
        status_emoji = {"available": "✅", "unavailable": "❌", "skipped": "⏭️"}
        emoji = status_emoji.get(info["status"], "❓")
        print(f"   {emoji} {info['name']}: {info['status']}")
        if info.get("reason"):
            print(f"      Причина: {info['reason']}")
    
    # Проверка наличия аудио файла для демо
    demo_audio_files = [
        "demo_audio.wav",
        "test_audio.mp3", 
        "../audio/sample.wav",
        "sample_audio.wav"
    ]
    
    demo_file = None
    for audio_file in demo_audio_files:
        if Path(audio_file).exists():
            demo_file = audio_file
            break
    
    if demo_file:
        print(f"\\n🎵 Найден демо файл: {demo_file}")
        
        try:
            # Транскрипция
            result = await demo.transcribe_audio(
                audio_file=demo_file,
                language="ru",
                domain="ecommerce",
                use_ensemble=True
            )
            
            # Вывод результата
            demo.print_transcription_result(result)
            
        except Exception as e:
            print(f"❌ Ошибка транскрипции: {e}")
            logger.error(f"Transcription failed: {e}")
    
    else:
        print("\\n⚠️ Демо аудио файл не найден.")
        print("   Поместите аудио файл (demo_audio.wav, test_audio.mp3 и т.д.) в текущую папку")
        print("   или укажите путь к существующему файлу в коде.")
        
        # Показать информацию об ensemble
        if demo.ensemble_service:
            ensemble_info = demo.ensemble_service.get_ensemble_info()
            print("\\n🤖 Информация об Ensemble сервисе:")
            print(f"   Модели: {ensemble_info['models_count']}")
            print(f"   Целевое качество: {ensemble_info['target_quality_threshold']:.1%}")
            print(f"   Метод консенсуса: {ensemble_info['consensus_method']}")


if __name__ == "__main__":
    # Запуск демо
    asyncio.run(main())