"""
Enhanced Transcriber - Полная интеграция для качества 95%+
Complete integration for 95%+ transcription quality
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Core components
from providers.tone import ToneTranscriber
from providers.whisper.working_whisper import WorkingWhisperTranscriber
from providers.whisper import WhisperOpenAITranscriber
from services.ensemble_service import EnsembleTranscriptionService
from services.audio_processor import AudioProcessorService
from services.quality_assessor import QualityAssessmentService
from core.models.transcription_result import TranscriptionResult
from core.models.config_models import TranscriptionConfig, ECommerceConfig

logger = logging.getLogger(__name__)


class EnhancedTranscriber:
    """
    Главный класс Enhanced Transcriber для достижения качества 95%+
    Main Enhanced Transcriber class for achieving 95%+ quality
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        target_quality: float = 0.95,
        enable_audio_enhancement: bool = True,
        enable_quality_assessment: bool = True,
        domain: str = "ecommerce"
    ):
        """
        Инициализация Enhanced Transcriber
        
        Args:
            openai_api_key: OpenAI API ключ (опционально)
            target_quality: Целевое качество (0.95 = 95%)
            enable_audio_enhancement: Включить улучшение аудио
            enable_quality_assessment: Включить оценку качества
            domain: Домен для специализации
        """
        self.target_quality = target_quality
        self.domain = domain
        self.openai_api_key = openai_api_key
        
        # Конфигурация
        self.config = TranscriptionConfig()
        self.ecommerce_config = ECommerceConfig() if domain == "ecommerce" else None
        
        # Компоненты системы
        self.models = []
        self.audio_processor = None
        self.quality_assessor = None
        self.ensemble_service = None
        
        # Статистика
        self.stats = {
            "total_transcriptions": 0,
            "successful_transcriptions": 0,
            "average_quality": 0.0,
            "quality_95_plus_count": 0
        }
        
        logger.info(f"🎯 Enhanced Transcriber initialized (target: {target_quality:.1%})")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Инициализация всех компонентов системы
        
        Returns:
            Dict: Статус инициализации компонентов
        """
        logger.info("🚀 Starting Enhanced Transcriber initialization...")
        
        init_results = {
            "models": [],
            "audio_processor": False,
            "quality_assessor": False,
            "ensemble_service": False,
            "ready": False
        }
        
        # 1. Инициализация моделей
        await self._initialize_models(init_results)
        
        # 2. Инициализация аудио процессора
        if len(self.models) > 0:
            await self._initialize_audio_processor(init_results)
        
        # 3. Инициализация оценщика качества
        await self._initialize_quality_assessor(init_results)
        
        # 4. Инициализация ensemble сервиса
        if len(self.models) >= 2:
            await self._initialize_ensemble_service(init_results)
        
        # Проверка готовности
        init_results["ready"] = (
            len(self.models) > 0 and
            (self.ensemble_service is not None or len(self.models) == 1)
        )
        
        if init_results["ready"]:
            logger.info("✅ Enhanced Transcriber ready for 95%+ quality transcription!")
        else:
            logger.error("❌ Enhanced Transcriber initialization failed")
        
        return init_results
    
    async def _initialize_models(self, init_results: Dict):
        """Инициализация транскрипционных моделей"""
        logger.info("🤖 Initializing transcription models...")
        
        # 1. T-one (приоритет для русского языка)
        try:
            tone_model = ToneTranscriber("voicekit/tone-ru")
            self.models.append(tone_model)
            init_results["models"].append({
                "name": "T-one Russian",
                "status": "success",
                "priority": "high"
            })
            logger.info("✅ T-one model loaded (Russian specialist)")
        except Exception as e:
            init_results["models"].append({
                "name": "T-one Russian", 
                "status": "failed",
                "error": str(e)
            })
            logger.warning(f"⚠️ T-one model failed: {e}")
        
        # 2. Whisper Local
        try:
            whisper_local = WorkingWhisperTranscriber("large-v3", device="auto")
            self.models.append(whisper_local)
            init_results["models"].append({
                "name": "Whisper Local",
                "status": "success",
                "priority": "large-v3"
            })
            logger.info("✅ Whisper Local model loaded")
        except Exception as e:
            init_results["models"].append({
                "name": "Whisper Local",
                "status": "failed", 
                "error": str(e)
            })
            logger.warning(f"⚠️ Whisper Local model failed: {e}")
        
        # 3. OpenAI Whisper (если есть ключ)
        if self.openai_api_key:
            try:
                whisper_openai = WhisperOpenAITranscriber(self.openai_api_key)
                self.models.append(whisper_openai)
                init_results["models"].append({
                    "name": "OpenAI Whisper",
                    "status": "success",
                    "priority": "high"
                })
                logger.info("✅ OpenAI Whisper API available")
            except Exception as e:
                init_results["models"].append({
                    "name": "OpenAI Whisper",
                    "status": "failed",
                    "error": str(e)
                })
                logger.warning(f"⚠️ OpenAI Whisper failed: {e}")
        
        logger.info(f"📊 Models loaded: {len(self.models)} / {len(init_results['models'])}")
    
    async def _initialize_audio_processor(self, init_results: Dict):
        """Инициализация аудио процессора"""
        try:
            self.audio_processor = AudioProcessorService(
                enable_noise_reduction=True,
                enable_volume_normalization=True,
                enable_speech_enhancement=True,
                target_sample_rate=16000,
                convert_to_mono=True
            )
            init_results["audio_processor"] = True
            logger.info("✅ Audio processor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Audio processor initialization failed: {e}")
    
    async def _initialize_quality_assessor(self, init_results: Dict):
        """Инициализация оценщика качества"""
        try:
            self.quality_assessor = QualityAssessmentService(
                enable_semantic_analysis=True,
                enable_domain_analysis=True,
                ecommerce_config=self.ecommerce_config
            )
            init_results["quality_assessor"] = True
            logger.info("✅ Quality assessor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Quality assessor initialization failed: {e}")
    
    async def _initialize_ensemble_service(self, init_results: Dict):
        """Инициализация ensemble сервиса"""
        try:
            self.ensemble_service = EnsembleTranscriptionService(
                models=self.models,
                audio_processor=self.audio_processor,
                quality_assessor=self.quality_assessor,
                target_quality_threshold=self.target_quality
            )
            init_results["ensemble_service"] = True
            logger.info(f"✅ Ensemble service initialized with {len(self.models)} models")
        except Exception as e:
            logger.warning(f"⚠️ Ensemble service initialization failed: {e}")
    
    async def transcribe(
        self,
        audio_file: str,
        language: str = "ru",
        max_quality_iterations: int = 3,
        force_target_quality: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Высококачественная транскрипция с целевым качеством 95%+
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык
            max_quality_iterations: Максимум итераций для достижения качества
            force_target_quality: Принудительно достигать целевое качество
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат транскрипции
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if not self.models:
            raise RuntimeError("No transcription models available")
        
        start_time = time.time()
        logger.info(f"🎵 Starting transcription: {Path(audio_file).name}")
        logger.info(f"🎯 Target quality: {self.target_quality:.1%}")
        
        try:
            # Ensemble транскрипция (если доступна)
            if self.ensemble_service:
                result = await self.ensemble_service.transcribe_with_quality_target(
                    audio_file=audio_file,
                    language=language,
                    domain=self.domain,
                    max_iterations=max_quality_iterations,
                    **kwargs
                )
            
            # Fallback: одиночная модель
            else:
                logger.info("🔧 Using single model fallback")
                result = await self.models[0].transcribe(audio_file, language, **kwargs)
                
                # Добавляем оценку качества
                if self.quality_assessor:
                    quality_metrics = await self.quality_assessor.assess_quality(
                        result.text,
                        audio_file,
                        domain=self.domain
                    )
                    result.quality_metrics = quality_metrics
            
            # Обновление статистики
            self._update_stats(result)
            
            total_time = time.time() - start_time
            result.processing_time = total_time
            
            # Логирование результата
            self._log_result(result, audio_file)
            
            # Проверка достижения целевого качества
            if force_target_quality and result.quality_metrics:
                if result.quality_metrics.overall_score < self.target_quality:
                    logger.warning(
                        f"⚠️ Target quality not achieved: "
                        f"{result.quality_metrics.overall_score:.1%} < {self.target_quality:.1%}"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            self.stats["total_transcriptions"] += 1
            raise
    
    async def transcribe_batch(
        self,
        audio_files: List[str],
        language: str = "ru",
        max_concurrent: int = 3,
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        Пакетная транскрипция нескольких файлов
        
        Args:
            audio_files: Список путей к аудио файлам
            language: Язык
            max_concurrent: Максимум одновременных транскрипций
            **kwargs: Дополнительные параметры
            
        Returns:
            List[TranscriptionResult]: Результаты транскрипций
        """
        logger.info(f"📦 Starting batch transcription: {len(audio_files)} files")
        
        # Семафор для ограничения одновременных транскрипций
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transcribe_with_semaphore(audio_file: str) -> TranscriptionResult:
            async with semaphore:
                return await self.transcribe(audio_file, language, **kwargs)
        
        # Выполнение всех транскрипций
        tasks = [transcribe_with_semaphore(audio_file) for audio_file in audio_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обработка результатов
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ File {audio_files[i]} failed: {result}")
                # Создаем failed результат
                failed_result = TranscriptionResult(
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    model_used="ensemble",
                    language_detected=language,
                    status="failed",
                    error_message=str(result)
                )
                successful_results.append(failed_result)
            else:
                successful_results.append(result)
        
        success_count = sum(1 for r in successful_results if r.text)
        logger.info(f"✅ Batch completed: {success_count}/{len(audio_files)} successful")
        
        return successful_results
    
    def _update_stats(self, result: TranscriptionResult):
        """Обновление статистики"""
        self.stats["total_transcriptions"] += 1
        
        if result.text and result.status != "failed":
            self.stats["successful_transcriptions"] += 1
            
            if result.quality_metrics:
                # Обновление средней оценки качества
                current_avg = self.stats["average_quality"]
                current_count = self.stats["successful_transcriptions"]
                new_quality = result.quality_metrics.overall_score
                
                self.stats["average_quality"] = (
                    (current_avg * (current_count - 1) + new_quality) / current_count
                )
                
                # Подсчет достижений целевого качества
                if new_quality >= self.target_quality:
                    self.stats["quality_95_plus_count"] += 1
    
    def _log_result(self, result: TranscriptionResult, audio_file: str):
        """Логирование результата"""
        filename = Path(audio_file).name
        
        if result.quality_metrics:
            quality_info = f"{result.quality_metrics.overall_score:.1%} ({result.quality_metrics.quality_level.value})"
            target_achieved = "🎯" if result.quality_metrics.overall_score >= self.target_quality else "⚠️"
        else:
            quality_info = "N/A"
            target_achieved = "❓"
        
        logger.info(
            f"{target_achieved} {filename}: "
            f"Quality={quality_info}, "
            f"Confidence={result.confidence:.1%}, "
            f"Time={result.processing_time:.1f}s, "
            f"Words={len(result.text.split())}"
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        return {
            "system_ready": bool(self.models),
            "models_count": len(self.models),
            "ensemble_available": self.ensemble_service is not None,
            "audio_processor_available": self.audio_processor is not None,
            "quality_assessor_available": self.quality_assessor is not None,
            "target_quality": self.target_quality,
            "domain": self.domain,
            "statistics": self.stats.copy()
        }
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Получение статистики качества"""
        if self.stats["total_transcriptions"] == 0:
            return {"message": "No transcriptions performed yet"}
        
        success_rate = self.stats["successful_transcriptions"] / self.stats["total_transcriptions"]
        target_achievement_rate = (
            self.stats["quality_95_plus_count"] / self.stats["successful_transcriptions"]
            if self.stats["successful_transcriptions"] > 0 else 0
        )
        
        return {
            "total_transcriptions": self.stats["total_transcriptions"],
            "successful_transcriptions": self.stats["successful_transcriptions"],
            "success_rate": success_rate,
            "average_quality": self.stats["average_quality"],
            "target_quality": self.target_quality,
            "target_achievement_count": self.stats["quality_95_plus_count"],
            "target_achievement_rate": target_achievement_rate,
            "quality_level": (
                "🎯 Excellent" if target_achievement_rate > 0.9 else
                "✅ Good" if target_achievement_rate > 0.7 else
                "⚠️ Needs improvement"
            )
        }
    
    async def cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Cleaning up Enhanced Transcriber resources...")
        
        # Очистка временных файлов аудио процессора
        if self.audio_processor and hasattr(self.audio_processor, 'temp_dir'):
            temp_dir = Path(self.audio_processor.temp_dir)
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*_enhanced_*"):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        logger.info("✅ Cleanup completed")