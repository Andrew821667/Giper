"""
OpenAI Whisper API провайдер для транскрипции
OpenAI Whisper API provider for transcription
"""

import asyncio
import time
import logging
import aiohttp
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.interfaces.transcriber import ITranscriber
from core.models.transcription_result import TranscriptionResult, WordTimestamp, TranscriptionStatus

logger = logging.getLogger(__name__)


class WhisperOpenAITranscriber(ITranscriber):
    """
    OpenAI Whisper API транскрайбер
    OpenAI Whisper API transcriber
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        Инициализация OpenAI Whisper транскрайбера
        
        Args:
            api_key: API ключ OpenAI
            model: Модель (whisper-1)
            base_url: Базовый URL API
        """
        self.api_key = api_key
        self.model_name_str = model
        self.base_url = base_url
        self._supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        self._supported_languages = ['ru', 'en', 'auto']
        self._max_file_size_mb = 25  # Ограничение OpenAI API
        
        # Проверка API ключа
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Транскрипция аудио файла через OpenAI Whisper API
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык (ISO 639-1 код: 'ru', 'en' или None для автодетекции)
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат транскрипции
        """
        if not self.is_supported_format(audio_file):
            raise ValueError(f"Unsupported file format: {Path(audio_file).suffix}")
        
        file_path = Path(audio_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Проверка размера файла
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self._max_file_size_mb:
            raise ValueError(f"File size ({file_size_mb:.1f}MB) exceeds OpenAI limit ({self._max_file_size_mb}MB)")
        
        start_time = time.time()
        
        try:
            # Транскрипция через API
            result = await self._call_openai_api(
                audio_file, 
                language, 
                kwargs
            )
            
            processing_time = time.time() - start_time
            
            # Обработка результата
            text = result.get("text", "").strip()
            
            # Постобработка для русского языка
            detected_language = result.get("language", language or "unknown")
            if detected_language == "ru" or language == "ru":
                text = self._enhance_russian_text(text)
            
            # Извлечение временных меток (если доступны)
            word_timestamps = self._extract_word_timestamps(result)
            
            # Анализ аудио метаданных
            audio_metadata = self._analyze_audio_metadata(audio_file)
            
            return TranscriptionResult(
                text=text,
                confidence=0.85,  # OpenAI API не возвращает confidence
                processing_time=processing_time,
                model_used=f"OpenAI Whisper ({self.model_name_str})",
                language_detected=detected_language,
                word_timestamps=word_timestamps,
                audio_duration=audio_metadata.get("duration"),
                sample_rate=audio_metadata.get("sample_rate", 16000),
                file_size=file_path.stat().st_size,
                status=TranscriptionStatus.COMPLETED,
                provider_metadata={
                    "model": self.model_name_str,
                    "provider": "whisper_openai",
                    "api_version": "v1",
                    "response_format": result.get("response_format", "json"),
                    "temperature": kwargs.get("temperature", 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI Whisper API transcription failed: {e}")
            
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"OpenAI Whisper ({self.model_name_str})",
                language_detected=language or "unknown",
                status=TranscriptionStatus.FAILED,
                error_message=str(e)
            )
    
    async def _call_openai_api(
        self, 
        audio_file: str, 
        language: Optional[str], 
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Вызов OpenAI Whisper API
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык
            kwargs: Дополнительные параметры
            
        Returns:
            Dict: Ответ API
        """
        url = f"{self.base_url}/audio/transcriptions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Подготовка данных запроса
        data = {
            "model": self.model_name_str,
            "response_format": kwargs.get("response_format", "verbose_json"),
            "temperature": kwargs.get("temperature", 0.0),
        }
        
        # Добавление языка если указан
        if language and language != "auto":
            data["language"] = language
        
        # Добавление промпта если указан
        if kwargs.get("prompt"):
            data["prompt"] = kwargs["prompt"]
        
        # Настройка timestamp granularities
        if kwargs.get("timestamp_granularities"):
            data["timestamp_granularities"] = kwargs["timestamp_granularities"]
        elif kwargs.get("word_timestamps", True):
            data["timestamp_granularities"] = ["word"]
        
        try:
            async with aiohttp.ClientSession() as session:
                with open(audio_file, 'rb') as audio:
                    form_data = aiohttp.FormData()
                    
                    # Добавление файла
                    form_data.add_field(
                        'file', 
                        audio, 
                        filename=Path(audio_file).name,
                        content_type='audio/wav'
                    )
                    
                    # Добавление параметров
                    for key, value in data.items():
                        if isinstance(value, list):
                            for item in value:
                                form_data.add_field(key, item)
                        else:
                            form_data.add_field(key, str(value))
                    
                    async with session.post(
                        url,
                        headers=headers,
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=300)  # 5 минут timeout
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"OpenAI API transcription successful")
                            return result
                        else:
                            error_text = await response.text()
                            logger.error(f"OpenAI API error {response.status}: {error_text}")
                            
                            try:
                                error_json = json.loads(error_text)
                                error_message = error_json.get("error", {}).get("message", error_text)
                            except:
                                error_message = error_text
                            
                            raise Exception(f"OpenAI API error: {error_message}")
                            
        except asyncio.TimeoutError:
            raise Exception("OpenAI API request timeout")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _extract_word_timestamps(self, result: Dict) -> Optional[List[WordTimestamp]]:
        """Извлечение временных меток слов из ответа OpenAI API"""
        try:
            word_timestamps = []
            
            # Новый формат с word-level timestamps
            if "words" in result and result["words"]:
                for word_info in result["words"]:
                    timestamp = WordTimestamp(
                        word=word_info.get("word", "").strip(),
                        start_time=word_info.get("start", 0.0),
                        end_time=word_info.get("end", 0.0),
                        confidence=0.85  # OpenAI не предоставляет word-level confidence
                    )
                    word_timestamps.append(timestamp)
                
                return word_timestamps
            
            # Старый формат с сегментами
            elif "segments" in result and result["segments"]:
                for segment in result["segments"]:
                    if "words" in segment and segment["words"]:
                        for word_info in segment["words"]:
                            timestamp = WordTimestamp(
                                word=word_info.get("word", "").strip(),
                                start_time=word_info.get("start", 0.0),
                                end_time=word_info.get("end", 0.0),
                                confidence=0.85
                            )
                            word_timestamps.append(timestamp)
                    else:
                        # Приблизительные timestamps на основе сегментов
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
                                confidence=0.85
                            )
                            word_timestamps.append(timestamp)
                
                return word_timestamps
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract word timestamps: {e}")
            return None
    
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
        
        # Исправление частых ошибок OpenAI Whisper для русского
        corrections = {
            r'\\bто\\s*есть\\b': 'то есть',
            r'\\bпо\\s*этому\\b': 'поэтому', 
            r'\\bтак\\s*же\\b': 'также',
            r'\\bвсё\\s*таки\\b': 'всё-таки',
            r'\\bкак\\s*будто\\b': 'как будто',
            r'\\bкак\\s*бы\\b': 'как бы',
            r'\\bв\\s*общем\\b': 'в общем',
            r'\\bтак\\s*как\\b': 'так как',
            r'\\bпотому\\s*что\\b': 'потому что'
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
        return {
            "name": self.model_name_str,
            "provider": "whisper_openai",
            "type": "Cloud-based ASR API",
            "architecture": "Whisper Transformer (large-v2)",
            "parameters": "1550M",
            "specialization": "Multilingual, general domain",
            "supported_languages": self._supported_languages,
            "supported_formats": list(self._supported_formats),
            "max_file_size_mb": self._max_file_size_mb,
            "pricing": "$0.006 per minute",
            "api_version": "v1",
            "license": "Commercial API"
        }
    
    @property
    def model_name(self) -> str:
        """Название модели"""
        return self.model_name_str
    
    @property
    def supported_languages(self) -> List[str]:
        """Поддерживаемые языки"""
        return self._supported_languages.copy()