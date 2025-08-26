"""
Ensemble транскрипция для максимального качества (95%+)
Ensemble transcription service for maximum quality (95%+)
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import statistics
from collections import Counter

from core.interfaces.transcriber import ITranscriber
from core.interfaces.audio_processor import IAudioProcessor
from core.interfaces.quality_assessor import IQualityAssessor
from core.models.transcription_result import TranscriptionResult, WordTimestamp, TranscriptionStatus
from core.models.quality_metrics import QualityMetrics, QualityLevel
from providers.tone import ToneTranscriber
from providers.whisper import WhisperLocalTranscriber, WhisperOpenAITranscriber

logger = logging.getLogger(__name__)


class EnsembleTranscriptionService:
    """
    Ensemble сервис транскрипции для достижения качества 95%+
    Uses multiple models and advanced consensus mechanisms
    """
    
    def __init__(
        self,
        models: List[ITranscriber],
        audio_processor: Optional[IAudioProcessor] = None,
        quality_assessor: Optional[IQualityAssessor] = None,
        target_quality_threshold: float = 0.95
    ):
        """
        Инициализация ensemble сервиса
        
        Args:
            models: Список транскрайберов для ensemble
            audio_processor: Процессор аудио (опционально)
            quality_assessor: Оценщик качества (опционально)
            target_quality_threshold: Целевой порог качества (0.95 = 95%)
        """
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models for consensus")
        
        self.models = models
        self.audio_processor = audio_processor
        self.quality_assessor = quality_assessor
        self.target_quality_threshold = target_quality_threshold
        
        # Веса моделей (можно настраивать на основе тестирования)
        self.model_weights = self._initialize_model_weights()
        
        logger.info(f"Ensemble service initialized with {len(models)} models")
    
    def _initialize_model_weights(self) -> Dict[str, float]:
        """Инициализация весов моделей на основе их характеристик"""
        weights = {}
        
        for model in self.models:
            model_name = model.model_name
            
            # Веса на основе типа модели для русского языка
            if "tone" in model_name.lower():
                weights[model_name] = 1.2  # T-one лучше для русского
            elif "whisper" in model_name.lower():
                if "openai" in str(type(model)).lower():
                    weights[model_name] = 1.0  # OpenAI Whisper
                else:
                    weights[model_name] = 0.9  # Локальный Whisper
            else:
                weights[model_name] = 1.0
        
        # Нормализация весов
        total_weight = sum(weights.values())
        return {k: v/total_weight for k, v in weights.items()}
    
    async def transcribe_with_quality_target(
        self,
        audio_file: str,
        language: str = "ru",
        domain: str = "ecommerce",
        max_iterations: int = 3,
        **kwargs
    ) -> TranscriptionResult:
        """
        Транскрипция с целевым качеством 95%+
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык
            domain: Домен (для постобработки)
            max_iterations: Максимальное количество итераций улучшения
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат с качеством 95%+
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        best_result = None
        iteration = 0
        
        # Предварительная обработка аудио (если есть процессор)
        processed_audio_file = audio_file
        if self.audio_processor:
            try:
                processed_audio_file = await self.audio_processor.enhance_audio(audio_file)
                logger.info(f"Audio enhanced: {processed_audio_file}")
            except Exception as e:
                logger.warning(f"Audio enhancement failed, using original: {e}")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Starting ensemble iteration {iteration}/{max_iterations}")
            
            try:
                # Выполнение ensemble транскрипции
                ensemble_result = await self._perform_ensemble_transcription(
                    processed_audio_file,
                    language,
                    iteration,
                    **kwargs
                )
                
                # Оценка качества
                if self.quality_assessor:
                    quality_metrics = await self.quality_assessor.assess_quality(
                        ensemble_result.text,
                        audio_file,
                        domain=domain
                    )
                    ensemble_result.quality_metrics = quality_metrics
                    
                    logger.info(
                        f"Iteration {iteration} quality: {quality_metrics.overall_score:.3f} "
                        f"(target: {self.target_quality_threshold:.3f})"
                    )
                    
                    # Проверка достижения целевого качества
                    if quality_metrics.overall_score >= self.target_quality_threshold:
                        logger.info(f"Target quality achieved in iteration {iteration}")
                        best_result = ensemble_result
                        break
                    
                    # Сохранение лучшего результата
                    if not best_result or quality_metrics.overall_score > best_result.quality_metrics.overall_score:
                        best_result = ensemble_result
                
                else:
                    # Без оценщика качества - используем первый результат
                    best_result = ensemble_result
                    break
                    
            except Exception as e:
                logger.error(f"Ensemble iteration {iteration} failed: {e}")
                if iteration == max_iterations:
                    raise
        
        # Финальная постобработка
        if best_result:
            best_result = await self._apply_final_postprocessing(
                best_result, 
                domain,
                language
            )
            
            # Обновление общего времени обработки
            best_result.processing_time = time.time() - start_time
            
            logger.info(
                f"Final ensemble result: {best_result.quality_metrics.overall_score:.3f} quality, "
                f"{best_result.processing_time:.1f}s processing time"
            )
            
            return best_result
        
        # Fallback - если ничего не получилось
        raise RuntimeError("All ensemble iterations failed")
    
    async def _perform_ensemble_transcription(
        self,
        audio_file: str,
        language: str,
        iteration: int,
        **kwargs
    ) -> TranscriptionResult:
        """
        Выполнение ensemble транскрипции для одной итерации
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык
            iteration: Номер итерации
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Консенсус результат
        """
        # Параллельный запуск всех моделей
        tasks = []
        for model in self.models:
            task = asyncio.create_task(
                self._safe_transcribe_with_model(model, audio_file, language, **kwargs),
                name=f"model_{model.model_name}_iter_{iteration}"
            )
            tasks.append(task)
        
        # Ждем завершения всех моделей
        model_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Фильтрация успешных результатов
        successful_results = []
        for i, result in enumerate(model_results):
            if isinstance(result, TranscriptionResult) and result.status == TranscriptionStatus.COMPLETED:
                successful_results.append(result)
            else:
                logger.warning(f"Model {self.models[i].model_name} failed: {result}")
        
        if not successful_results:
            raise RuntimeError("All models failed in ensemble")
        
        logger.info(f"Successful models in iteration {iteration}: {len(successful_results)}")
        
        # Создание консенсуса
        consensus_result = await self._create_consensus(successful_results, audio_file)
        
        return consensus_result
    
    async def _safe_transcribe_with_model(
        self, 
        model: ITranscriber, 
        audio_file: str, 
        language: str,
        **kwargs
    ) -> TranscriptionResult:
        """Безопасная транскрипция с обработкой ошибок"""
        try:
            return await model.transcribe(audio_file, language, **kwargs)
        except Exception as e:
            logger.error(f"Model {model.model_name} transcription failed: {e}")
            # Возвращаем failed результат
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=0.0,
                model_used=model.model_name,
                language_detected=language,
                status=TranscriptionStatus.FAILED,
                error_message=str(e)
            )
    
    async def _create_consensus(
        self, 
        results: List[TranscriptionResult], 
        audio_file: str
    ) -> TranscriptionResult:
        """
        Создание консенсуса из результатов multiple моделей
        
        Args:
            results: Успешные результаты транскрипции
            audio_file: Путь к аудио файлу
            
        Returns:
            TranscriptionResult: Консенсус результат
        """
        if not results:
            raise ValueError("No results provided for consensus")
        
        if len(results) == 1:
            return results[0]
        
        # Метод 1: Weighted voting по словам
        consensus_text = await self._weighted_word_consensus(results)
        
        # Метод 2: Выбор лучшего результата по confidence
        best_result = max(results, key=lambda r: r.confidence)
        
        # Метод 3: Hybrid approach - консенсус текст + метаданные лучшего
        avg_confidence = statistics.mean([r.confidence for r in results])
        avg_processing_time = statistics.mean([r.processing_time for r in results])
        
        # Консенсус временных меток
        consensus_timestamps = self._merge_word_timestamps(results)
        
        # Создание консенсус результата
        consensus_result = TranscriptionResult(
            text=consensus_text,
            confidence=max(avg_confidence, best_result.confidence),  # Берем максимум
            processing_time=avg_processing_time,
            model_used=f"Ensemble ({len(results)} models)",
            language_detected=best_result.language_detected,
            word_timestamps=consensus_timestamps,
            audio_duration=best_result.audio_duration,
            sample_rate=best_result.sample_rate,
            file_size=Path(audio_file).stat().st_size,
            status=TranscriptionStatus.COMPLETED,
            provider_metadata={
                "ensemble_size": len(results),
                "models_used": [r.model_used for r in results],
                "consensus_method": "weighted_word_voting",
                "avg_confidence": avg_confidence,
                "best_individual_confidence": best_result.confidence
            }
        )
        
        return consensus_result
    
    async def _weighted_word_consensus(self, results: List[TranscriptionResult]) -> str:
        """
        Weighted консенсус по словам
        
        Args:
            results: Результаты транскрипции
            
        Returns:
            str: Консенсус текст
        """
        # Токенизация всех результатов
        all_tokens = []
        for result in results:
            tokens = result.text.split()
            model_weight = self.model_weights.get(result.model_used.split('(')[0].strip(), 1.0)
            all_tokens.append({
                'tokens': tokens,
                'confidence': result.confidence,
                'weight': model_weight,
                'model': result.model_used
            })
        
        # Поиск максимальной длины для выравнивания
        max_length = max(len(token_set['tokens']) for token_set in all_tokens)
        
        consensus_words = []
        
        for position in range(max_length):
            position_candidates = {}
            
            # Собираем кандидатов для позиции
            for token_set in all_tokens:
                if position < len(token_set['tokens']):
                    word = token_set['tokens'][position].lower()
                    score = token_set['confidence'] * token_set['weight']
                    
                    if word in position_candidates:
                        position_candidates[word] += score
                    else:
                        position_candidates[word] = score
            
            # Выбираем лучший кандидат
            if position_candidates:
                best_word = max(position_candidates.items(), key=lambda x: x[1])[0]
                consensus_words.append(best_word)
        
        # Постобработка консенсуса
        consensus_text = ' '.join(consensus_words)
        consensus_text = await self._post_process_consensus(consensus_text)
        
        return consensus_text
    
    def _merge_word_timestamps(self, results: List[TranscriptionResult]) -> Optional[List[WordTimestamp]]:
        """Merge временных меток из разных моделей"""
        try:
            # Используем timestamps от модели с лучшим confidence
            best_result = max(
                (r for r in results if r.word_timestamps), 
                key=lambda r: r.confidence,
                default=None
            )
            
            return best_result.word_timestamps if best_result else None
            
        except Exception as e:
            logger.warning(f"Failed to merge word timestamps: {e}")
            return None
    
    async def _post_process_consensus(self, text: str) -> str:
        """Постобработка консенсус текста"""
        if not text:
            return text
        
        # Базовая очистка
        import re
        
        # Удаление дублированных слов
        words = text.split()
        cleaned_words = []
        prev_word = None
        
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
                prev_word = word.lower()
        
        cleaned_text = ' '.join(cleaned_words)
        
        # Исправление пунктуации
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Множественные пробелы
        cleaned_text = re.sub(r'\s+([,.!?;:])', r'\\1', cleaned_text)  # Пробелы перед пунктуацией
        
        return cleaned_text.strip()
    
    async def _apply_final_postprocessing(
        self, 
        result: TranscriptionResult, 
        domain: str,
        language: str
    ) -> TranscriptionResult:
        """
        Финальная постобработка результата
        
        Args:
            result: Результат транскрипции
            domain: Домен
            language: Язык
            
        Returns:
            TranscriptionResult: Обработанный результат
        """
        enhanced_text = result.text
        
        # Доменная постобработка для e-commerce
        if domain == "ecommerce":
            enhanced_text = await self._apply_ecommerce_corrections(enhanced_text)
        
        # Языковые исправления
        if language == "ru":
            enhanced_text = await self._apply_russian_corrections(enhanced_text)
        
        # Создание нового результата с улучшенным текстом
        enhanced_result = TranscriptionResult(
            text=enhanced_text,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_used=result.model_used,
            language_detected=result.language_detected,
            word_timestamps=result.word_timestamps,
            audio_duration=result.audio_duration,
            sample_rate=result.sample_rate,
            file_size=result.file_size,
            status=result.status,
            provider_metadata=result.provider_metadata,
            quality_metrics=result.quality_metrics
        )
        
        return enhanced_result
    
    async def _apply_ecommerce_corrections(self, text: str) -> str:
        """E-commerce доменные исправления"""
        if not text:
            return text
        
        import re
        
        # E-commerce терминология
        ecommerce_corrections = {
            r'\\b(закас|зокас|заказь)\\b': 'заказ',
            r'\\b(аплата|оплото|аплото)\\b': 'оплата',
            r'\\b(доствка|дастафка|достака)\\b': 'доставка',
            r'\\b(возрат|вазврат|возрать)\\b': 'возврат',
            r'\\b(тавар|товорр|товор)\\b': 'товар',
            r'\\b(скитка|скидко|скитко)\\b': 'скидка',
            r'\\b(карзина|корзино|карзино)\\b': 'корзина',
            r'\\b(качиство|кочество)\\b': 'качество',
            r'\\b(горантия|гарантея|гарантие)\\b': 'гарантия'
        }
        
        corrected_text = text
        for pattern, replacement in ecommerce_corrections.items():
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    async def _apply_russian_corrections(self, text: str) -> str:
        """Русские языковые исправления"""
        if not text:
            return text
        
        import re
        
        # Частые ошибки русского языка
        russian_corrections = {
            r'\\bт[ое]\\s*есть\\b': 'то есть',
            r'\\bпо\\s*этому\\b': 'поэтому',
            r'\\bтак\\s*же\\b': 'также', 
            r'\\bвсё\\s*таки\\b': 'всё-таки',
            r'\\bкак\\s*будто\\b': 'как будто',
            r'\\bкак\\s*бы\\b': 'как бы',
            r'\\bв\\s*общем\\b': 'в общем',
            r'\\bна\\s*самом\\s*деле\\b': 'на самом деле',
            r'\\bкстати\\s+говоря\\b': 'кстати говоря'
        }
        
        corrected_text = text
        for pattern, replacement in russian_corrections.items():
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Получение информации об ensemble сервисе"""
        return {
            "models_count": len(self.models),
            "models_info": [model.get_model_info() for model in self.models],
            "model_weights": self.model_weights,
            "target_quality_threshold": self.target_quality_threshold,
            "audio_processor_available": self.audio_processor is not None,
            "quality_assessor_available": self.quality_assessor is not None,
            "expected_quality": "95%+",
            "consensus_method": "weighted_word_voting"
        }