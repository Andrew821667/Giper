"""
Локальный Whisper провайдер для транскрипции
Local Whisper provider for transcription
"""

import asyncio
import time
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.interfaces.transcriber import ITranscriber
from core.models.transcription_result import TranscriptionResult, WordTimestamp, TranscriptionStatus

logger = logging.getLogger(__name__)


class WhisperLocalTranscriber(ITranscriber):
    """
    Локальный Whisper транскрайбер
    Local Whisper transcriber using whisper library
    """
    
    def __init__(
        self, 
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: str = "int8"
    ):
        """
        Инициализация локального Whisper транскрайбера
        
        Args:
            model_name: Размер модели (tiny, base, small, medium, large, large-v2, large-v3)
            device: Устройство (cpu, cuda, auto)
            compute_type: Тип вычислений (int8, int16, float16, float32)
        """
        self.model_name_str = model_name
        self.device = device or "auto"
        self.compute_type = compute_type
        self._model = None
        self._supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        self._supported_languages = ['ru', 'en', 'auto']
        
        # Попытка инициализации модели
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели Whisper"""
        try:
            import whisper
            
            # Загрузка модели
            self._model = whisper.load_model(
                name=self.model_name_str,
                device=self.device,
            )
            
            logger.info(f"Whisper model '{self.model_name_str}' loaded on {self.device}")
            
        except ImportError:
            logger.error("whisper package not installed. Install with: pip install openai-whisper")
            raise ImportError(
                "OpenAI Whisper не установлен. Установите: pip install openai-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise RuntimeError(f"Не удалось инициализировать модель Whisper: {e}")
    
    async def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Транскрипция аудио файла с использованием Whisper
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык ('ru', 'en', 'auto' или None для автодетекции)
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат транскрипции
        """
        if not self._model:
            raise RuntimeError("Whisper model not initialized")
        
        if not self.is_supported_format(audio_file):
            raise ValueError(f"Unsupported file format: {Path(audio_file).suffix}")
        
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Запуск в отдельном потоке для неблокирующего выполнения
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._sync_transcribe, 
                audio_file, 
                language,
                kwargs
            )
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=result["text"],
                confidence=result.get("avg_logprob_confidence", 0.8),
                processing_time=processing_time,
                model_used=f"Whisper ({self.model_name_str})",
                language_detected=result.get("language", language or "auto"),
                word_timestamps=result.get("word_timestamps"),
                audio_duration=result.get("duration"),
                sample_rate=result.get("sample_rate", 16000),
                file_size=Path(audio_file).stat().st_size,
                status=TranscriptionStatus.COMPLETED,
                provider_metadata={
                    "model_size": self.model_name_str,
                    "provider": "whisper_local",
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "segments_count": len(result.get("segments", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"Whisper ({self.model_name_str})",
                language_detected=language or "unknown",
                status=TranscriptionStatus.FAILED,
                error_message=str(e)
            )
    
    def _sync_transcribe(
        self, 
        audio_file: str, 
        language: Optional[str], 
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Синхронная транскрипция для выполнения в отдельном потоке
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык
            kwargs: Дополнительные параметры
            
        Returns:
            Dict: Результат транскрипции с метаданными
        """
        try:
            # Параметры транскрипции
            transcribe_options = {
                "verbose": False,
                "word_timestamps": kwargs.get("word_timestamps", True),
                "initial_prompt": kwargs.get("initial_prompt"),
                "temperature": kwargs.get("temperature", 0.0),
            }
            
            # Язык
            if language and language != "auto":
                transcribe_options["language"] = language
            
            # Основная транскрипция
            result = self._model.transcribe(audio_file, **transcribe_options)
            
            # Обработка результата
            text = result["text"].strip()
            
            # Постобработка в зависимости от языка
            if result.get("language") == "ru" or language == "ru":
                text = self._enhance_russian_text(text)
            
            # Извлечение временных меток
            word_timestamps = self._extract_word_timestamps(result)
            
            # Расчет средней уверенности
            avg_confidence = self._calculate_average_confidence(result)
            
            # Анализ аудио метаданных
            audio_metadata = self._analyze_audio_metadata(audio_file)
            
            return {
                "text": text,
                "language": result.get("language", "unknown"),
                "avg_logprob_confidence": avg_confidence,
                "word_timestamps": word_timestamps,
                "segments": result.get("segments", []),
                "duration": audio_metadata.get("duration"),
                "sample_rate": audio_metadata.get("sample_rate")
            }
            
        except Exception as e:
            logger.error(f"Sync transcription failed: {e}")
            raise
    
    def _extract_word_timestamps(self, result: Dict) -> Optional[List[WordTimestamp]]:
        """Извлечение временных меток слов из результата Whisper"""
        try:
            if not result.get("segments"):
                return None
                
            word_timestamps = []
            
            for segment in result["segments"]:
                if "words" in segment and segment["words"]:
                    for word_info in segment["words"]:
                        timestamp = WordTimestamp(
                            word=word_info.get("word", "").strip(),
                            start_time=word_info.get("start", 0.0),
                            end_time=word_info.get("end", 0.0),
                            confidence=word_info.get("probability", 0.8)
                        )
                        word_timestamps.append(timestamp)
                else:
                    # Если нет word-level timestamps, используем segment-level
                    words = segment.get("text", "").strip().split()
                    segment_duration = segment.get("end", 0) - segment.get("start", 0)
                    word_duration = segment_duration / max(len(words), 1)
                    
                    for i, word in enumerate(words):
                        word_start = segment.get("start", 0) + (i * word_duration)
                        word_end = word_start + word_duration
                        
                        timestamp = WordTimestamp(
                            word=word,
                            start_time=word_start,
                            end_time=word_end,
                            confidence=segment.get("avg_logprob", 0.8)
                        )
                        word_timestamps.append(timestamp)
            
            return word_timestamps
            
        except Exception as e:
            logger.warning(f"Failed to extract word timestamps: {e}")
            return None
    
    def _calculate_average_confidence(self, result: Dict) -> float:
        """Расчет средней уверенности на основе логарифмических вероятностей"""
        try:
            if "segments" not in result:
                return 0.8
            
            total_logprob = 0.0
            total_tokens = 0
            
            for segment in result["segments"]:
                if "avg_logprob" in segment:
                    # Конвертация log probability в обычную вероятность
                    segment_prob = min(1.0, max(0.0, (segment["avg_logprob"] + 1.0)))
                    segment_tokens = len(segment.get("tokens", [1]))
                    
                    total_logprob += segment_prob * segment_tokens
                    total_tokens += segment_tokens
            
            if total_tokens > 0:
                return total_logprob / total_tokens
            
            return 0.8
            
        except Exception as e:
            logger.warning(f"Failed to calculate average confidence: {e}")
            return 0.8
    
    def _enhance_russian_text(self, text: str) -> str:
        """
        Постобработка русского текста
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Улучшенный текст
        """
        if not text:
            return text
        
        # Базовые исправления для русского языка
        import re
        
        enhanced_text = text
        
        # Исправление частых ошибок Whisper для русского
        corrections = {
            r'\\bт[ое]\\s*есть\\b': 'то есть',
            r'\\bпо\\s*этому\\b': 'поэтому',
            r'\\bтак\\s*же\\b': 'также',
            r'\\bвсё\\s*таки\\b': 'всё-таки',
            r'\\bкак\\s*будто\\b': 'как будто',
            r'\\bкак\\s*бы\\b': 'как бы',
            r'\\bв\\s*общем\\b': 'в общем',
            r'\\bна\\s*самом\\s*деле\\b': 'на самом деле'
        }
        
        for pattern, replacement in corrections.items():
            enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
        
        # Улучшение пунктуации
        enhanced_text = self._improve_punctuation(enhanced_text)
        
        return enhanced_text.strip()
    
    def _improve_punctuation(self, text: str) -> str:
        """Базовое улучшение пунктуации"""
        import re
        
        # Добавление точек в конце предложений
        sentences = re.split(r'[.!?]+', text)
        improved_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:
                # Простая логика определения конца предложения
                if not sentence.endswith(('.', '!', '?', ',')):
                    sentence += '.'
                improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)
    
    def _analyze_audio_metadata(self, audio_file: str) -> Dict[str, Any]:
        """Анализ базовых метаданных аудио файла"""
        try:
            import librosa
            
            # Загрузка аудио для анализа метаданных
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[0]
            }
            
        except ImportError:
            logger.warning("librosa not available for audio analysis")
            return {"duration": 0.0, "sample_rate": 16000}
        except Exception as e:
            logger.warning(f"Failed to analyze audio metadata: {e}")
            return {"duration": 0.0, "sample_rate": 16000}
    
    def is_supported_format(self, file_path: str) -> bool:
        """
        Проверка поддерживаемого формата файла
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если формат поддерживается
        """
        return Path(file_path).suffix.lower() in self._supported_formats
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Dict: Информация о модели
        """
        model_specs = {
            "tiny": {"params": "39M", "vram": "~1GB", "speed": "~32x"},
            "base": {"params": "74M", "vram": "~1GB", "speed": "~16x"},
            "small": {"params": "244M", "vram": "~2GB", "speed": "~6x"},
            "medium": {"params": "769M", "vram": "~5GB", "speed": "~2x"},
            "large": {"params": "1550M", "vram": "~10GB", "speed": "~1x"},
            "large-v2": {"params": "1550M", "vram": "~10GB", "speed": "~1x"},
            "large-v3": {"params": "1550M", "vram": "~10GB", "speed": "~1x"}
        }
        
        specs = model_specs.get(self.model_name_str, {"params": "Unknown", "vram": "Unknown", "speed": "Unknown"})
        
        return {
            "name": self.model_name_str,
            "provider": "whisper_local",
            "type": "Transformer-based ASR",
            "architecture": "Whisper Transformer",
            "parameters": specs["params"],
            "vram_usage": specs["vram"],
            "relative_speed": specs["speed"],
            "specialization": "Multilingual, general domain",
            "supported_languages": self._supported_languages,
            "supported_formats": list(self._supported_formats),
            "device": self.device,
            "compute_type": self.compute_type,
            "license": "MIT"
        }
    
    @property
    def model_name(self) -> str:
        """Название модели"""
        return self.model_name_str
    
    @property
    def supported_languages(self) -> List[str]:
        """Поддерживаемые языки"""
        return self._supported_languages.copy()