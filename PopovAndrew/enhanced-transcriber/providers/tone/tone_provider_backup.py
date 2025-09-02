"""
T-one provider для транскрипции русского языка
T-one provider for Russian language transcription
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.interfaces.transcriber import ITranscriber
from core.models.transcription_result import TranscriptionResult, WordTimestamp, TranscriptionStatus

logger = logging.getLogger(__name__)


class ToneTranscriber(ITranscriber):
    """
    T-one транскрайбер для высококачественной русской транскрипции
    T-one transcriber for high-quality Russian transcription
    """
    
    def __init__(self, model_name: str = "voicekit/tone-ru"):
        """
        Инициализация T-one транскрайбера
        
        Args:
            model_name: Название модели T-one
        """
        self.model_name_str = model_name
        self._model = None
        self._supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        self._supported_languages = ['ru']
        
        # Попытка инициализации модели
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели T-one"""
        try:
            # Правильная инициализация T-one
            import tone
            # Создаем T-one pipeline
            model = tone.StreamingCTCModel.from_hugging_face()
            logprob_splitter = tone.StreamingLogprobSplitter()
            decoder = tone.GreedyCTCDecoder()
            self._model = tone.StreamingCTCPipeline(
                model=model,
                logprob_splitter=logprob_splitter,
                decoder=decoder
            )
            logger.info(f"T-one model {self.model_name_str} initialized successfully")
        except ImportError:
            logger.error("tone-asr package not installed. Install with: pip install tone-asr")
            raise ImportError(
                "T-one ASR не установлен. Установите: pip install tone-asr"
            )
        except Exception as e:
            logger.error(f"Failed to initialize T-one model: {e}")
            raise RuntimeError(f"Не удалось инициализировать модель T-one: {e}")
    
    async def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Транскрипция аудио файла с использованием T-one
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык (игнорируется, T-one только для русского)
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат транскрипции
        """
        if not self._model:
            raise RuntimeError("T-one model not initialized")
        
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
                kwargs
            )
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=result["text"],
                confidence=result.get("confidence", 0.85),
                processing_time=processing_time,
                model_used=f"T-one ({self.model_name_str})",
                language_detected="ru",
                word_timestamps=result.get("word_timestamps"),
                audio_duration=result.get("audio_duration"),
                sample_rate=result.get("sample_rate"),
                file_size=Path(audio_file).stat().st_size,
                status=TranscriptionStatus.COMPLETED,
                provider_metadata={
                    "model_name": self.model_name_str,
                    "provider": "tone",
                    "streaming_used": result.get("streaming_used", False),
                    "chunk_size": result.get("chunk_size", 300)
                }
            )
            
        except Exception as e:
            logger.error(f"T-one transcription failed: {e}")
            
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"T-one ({self.model_name_str})",
                language_detected="ru",
                status=TranscriptionStatus.FAILED,
                error_message=str(e)
            )
    
    def _sync_transcribe(self, audio_file: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Синхронная транскрипция для выполнения в отдельном потоке
        
        Args:
            audio_file: Путь к аудио файлу
            kwargs: Дополнительные параметры
            
        Returns:
            Dict: Результат транскрипции с метаданными
        """
        try:
            # Основная транскрипция
            # Правильное использование T-one pipeline
            import tone
            import numpy as np

            # Загружаем аудио и обрезаем в допустимый диапазон
            audio_data = tone.read_audio(audio_file)
            audio_data = np.clip(audio_data, -32768, 32767)

            # Выполняем транскрипцию
            phrases = self._model.forward_offline(audio_data)

            # Собираем текст из фраз
            if hasattr(phrases, '__iter__'):
                text_parts = []
                for phrase in phrases:
                    if hasattr(phrase, 'text'):
                        text_parts.append(phrase.text)
                    else:
                        text_parts.append(str(phrase))
                result = {"text": " ".join(text_parts)}
            else:
                result = {"text": str(phrases)}
            
            # Обработка результата в зависимости от типа
            if hasattr(result, 'text'):
                # Объект результата
                text = result.text
                confidence = getattr(result, 'confidence', 0.85)
                word_timestamps = self._extract_word_timestamps(result)
            elif isinstance(result, dict):
                # Словарь результата
                text = result.get('text', '')
                confidence = result.get('confidence', 0.85)
                word_timestamps = self._extract_word_timestamps_dict(result)
            else:
                # Строка результата
                text = str(result)
                confidence = 0.85
                word_timestamps = None
            
            # Постобработка текста
            enhanced_text = self._enhance_russian_text(text)
            
            # Анализ аудио метаданных
            audio_metadata = self._analyze_audio_metadata(audio_file)
            
            return {
                "text": enhanced_text,
                "confidence": confidence,
                "word_timestamps": word_timestamps,
                "audio_duration": audio_metadata.get("duration"),
                "sample_rate": audio_metadata.get("sample_rate"),
                "streaming_used": kwargs.get("streaming", False),
                "chunk_size": kwargs.get("chunk_size", 300)
            }
            
        except Exception as e:
            logger.error(f"Sync transcription failed: {e}")
            raise
    
    def _extract_word_timestamps(self, result) -> Optional[List[WordTimestamp]]:
        """Извлечение временных меток слов из результата T-one"""
        try:
            if hasattr(result, 'word_timestamps') and result.word_timestamps:
                timestamps = []
                for word_info in result.word_timestamps:
                    timestamp = WordTimestamp(
                        word=word_info.get('word', ''),
                        start_time=word_info.get('start', 0.0),
                        end_time=word_info.get('end', 0.0),
                        confidence=word_info.get('confidence', 0.8)
                    )
                    timestamps.append(timestamp)
                return timestamps
        except Exception as e:
            logger.warning(f"Failed to extract word timestamps: {e}")
        
        return None
    
    def _extract_word_timestamps_dict(self, result: Dict) -> Optional[List[WordTimestamp]]:
        """Извлечение временных меток из словаря результата"""
        try:
            if 'word_timestamps' in result and result['word_timestamps']:
                return self._extract_word_timestamps(result)
        except Exception as e:
            logger.warning(f"Failed to extract word timestamps from dict: {e}")
        
        return None
    
    def _enhance_russian_text(self, text: str) -> str:
        """
        Постобработка русского текста для улучшения качества
        
        Args:
            text: Исходный текст транскрипции
            
        Returns:
            str: Улучшенный текст
        """
        if not text:
            return text
        
        # Базовые исправления для русского языка
        enhanced_text = text
        
        # Исправление частых ошибок T-one для русского
        corrections = {
            r'\bэто\s+самое\b': 'это самое',
            r'\bну\s+вот\b': 'ну вот', 
            r'\bто\s+есть\b': 'то есть',
            r'\bпо\s+этому\b': 'поэтому',
            r'\bтак\s+же\b': 'также',
            r'\bвсё\s+таки\b': 'всё-таки',
            r'\bкак\s+будто\b': 'как будто',
            r'\bкак\s+бы\b': 'как бы'
        }
        
        import re
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
            if sentence:
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
            return {}
        except Exception as e:
            logger.warning(f"Failed to analyze audio metadata: {e}")
            return {}
    
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
        return {
            "name": self.model_name_str,
            "provider": "tone",
            "type": "ASR (Automatic Speech Recognition)",
            "architecture": "Conformer-based CTC",
            "parameters": "71M",
            "specialization": "Russian language, telephony domain",
            "supported_languages": self._supported_languages,
            "supported_formats": list(self._supported_formats),
            "streaming_support": True,
            "real_time_factor": 1.2,
            "accuracy": "Lowest WER for Russian",
            "license": "Apache 2.0"
        }
    
    @property
    def model_name(self) -> str:
        """Название модели"""
        return self.model_name_str
    
    @property
    def supported_languages(self) -> List[str]:
        """Поддерживаемые языки"""
        return self._supported_languages.copy()


class ToneProvider:
    """
    Провайдер T-one моделей
    Provider for T-one models
    """
    
    @staticmethod
    def create_transcriber(model_name: str = "voicekit/tone-ru") -> ToneTranscriber:
        """
        Создание экземпляра T-one транскрайбера
        
        Args:
            model_name: Название модели
            
        Returns:
            ToneTranscriber: Готовый к использованию транскрайбер
        """
        return ToneTranscriber(model_name)
    
    @staticmethod
    def is_available() -> bool:
        """
        Проверка доступности T-one
        
        Returns:
            bool: True если T-one доступен
        """
        try:
            import tone_asr
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Получение списка доступных T-one моделей
        
        Returns:
            List[str]: Список доступных моделей
        """
        return [
            "voicekit/tone-ru",
            # Добавить другие модели T-one когда появятся
        ]
    
    @staticmethod
    def get_system_requirements() -> Dict[str, str]:
        """
        Получение системных требований
        
        Returns:
            Dict: Системные требования
        """
        return {
            "python": ">=3.9",
            "memory": ">=4GB RAM",
            "disk_space": ">=2GB free space", 
            "dependencies": "torch, torchaudio, librosa",
            "optional_gpu": "CUDA compatible GPU for faster processing"
        }