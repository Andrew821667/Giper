"""
Модели данных для результатов транскрипции
Data models for transcription results
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TranscriptionStatus(Enum):
    """Статус транскрипции"""
    PENDING = "pending"           # В ожидании
    PROCESSING = "processing"     # Обрабатывается
    COMPLETED = "completed"       # Завершено
    FAILED = "failed"            # Ошибка
    RETRYING = "retrying"        # Повторная попытка


@dataclass
class WordTimestamp:
    """
    Временная метка для слова
    Word-level timestamp information
    """
    word: str                     # Само слово
    start_time: float            # Время начала (секунды)
    end_time: float              # Время окончания (секунды) 
    confidence: float            # Уверенность модели (0-1)
    is_punctuation: bool = False # Является ли пунктуацией
    
    @property
    def duration(self) -> float:
        """Длительность произнесения слова"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "word": self.word,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "duration": self.duration,
            "is_punctuation": self.is_punctuation
        }


@dataclass  
class TranscriptionResult:
    """
    Результат транскрипции аудио
    Complete transcription result with metadata
    """
    text: str                                    # Транскрибированный текст
    confidence: float                            # Общая уверенность (0-1)
    processing_time: float                       # Время обработки (секунды)
    model_used: str                             # Использованная модель
    language_detected: str                      # Определенный язык
    
    # Опциональные поля
    word_timestamps: Optional[List[WordTimestamp]] = None  # Временные метки слов
    audio_duration: Optional[float] = None                 # Длительность аудио
    sample_rate: Optional[int] = None                     # Частота дискретизации
    file_size: Optional[int] = None                       # Размер файла в байтах
    
    # Качество и метрики
    quality_metrics: Optional[Dict[str, float]] = None    # Метрики качества
    preprocessing_applied: Optional[List[str]] = None     # Примененная предобработка
    
    # Метаданные обработки
    status: TranscriptionStatus = TranscriptionStatus.COMPLETED
    created_at: datetime = field(default_factory=datetime.now)
    provider_metadata: Optional[Dict[str, Any]] = None    # Метаданные провайдера
    
    # Информация об ошибках
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def word_count(self) -> int:
        """Количество слов в тексте"""
        return len(self.text.split())
    
    @property
    def character_count(self) -> int:
        """Количество символов"""
        return len(self.text)
    
    @property
    def words_per_minute(self) -> Optional[float]:
        """Скорость речи (слов в минуту)"""
        if self.audio_duration and self.audio_duration > 0:
            return (self.word_count / self.audio_duration) * 60
        return None
    
    @property
    def is_successful(self) -> bool:
        """Успешна ли транскрипция"""
        return self.status == TranscriptionStatus.COMPLETED and self.error_message is None
    
    def get_low_confidence_words(self, threshold: float = 0.5) -> List[WordTimestamp]:
        """
        Получение слов с низкой уверенностью
        Get words with low confidence scores
        
        Args:
            threshold: Пороговое значение уверенности
            
        Returns:
            List[WordTimestamp]: Слова с низкой уверенностью
        """
        if not self.word_timestamps:
            return []
        
        return [word for word in self.word_timestamps if word.confidence < threshold]
    
    def get_average_word_confidence(self) -> Optional[float]:
        """Средняя уверенность по словам"""
        if not self.word_timestamps:
            return None
        
        confidences = [word.confidence for word in self.word_timestamps]
        return sum(confidences) / len(confidences) if confidences else None
    
    def get_text_segments(self, max_segment_length: int = 100) -> List[str]:
        """
        Разбивка текста на сегменты
        Split text into segments
        
        Args:
            max_segment_length: Максимальная длина сегмента в словах
            
        Returns:
            List[str]: Список текстовых сегментов
        """
        words = self.text.split()
        segments = []
        
        for i in range(0, len(words), max_segment_length):
            segment = " ".join(words[i:i + max_segment_length])
            segments.append(segment)
        
        return segments
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON serialization"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "language_detected": self.language_detected,
            "word_timestamps": [w.to_dict() for w in self.word_timestamps] if self.word_timestamps else None,
            "audio_duration": self.audio_duration,
            "sample_rate": self.sample_rate,
            "file_size": self.file_size,
            "quality_metrics": self.quality_metrics,
            "preprocessing_applied": self.preprocessing_applied,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "provider_metadata": self.provider_metadata,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            # Calculated properties
            "word_count": self.word_count,
            "character_count": self.character_count,
            "words_per_minute": self.words_per_minute,
            "is_successful": self.is_successful,
            "average_word_confidence": self.get_average_word_confidence()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionResult':
        """Создание экземпляра из словаря"""
        # Конвертация word_timestamps
        word_timestamps = None
        if data.get("word_timestamps"):
            word_timestamps = [
                WordTimestamp(**word_data) 
                for word_data in data["word_timestamps"]
            ]
        
        # Конвертация даты
        created_at = datetime.now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        # Конвертация статуса
        status = TranscriptionStatus.COMPLETED
        if data.get("status"):
            status = TranscriptionStatus(data["status"])
        
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            processing_time=data["processing_time"],
            model_used=data["model_used"],
            language_detected=data["language_detected"],
            word_timestamps=word_timestamps,
            audio_duration=data.get("audio_duration"),
            sample_rate=data.get("sample_rate"),
            file_size=data.get("file_size"),
            quality_metrics=data.get("quality_metrics"),
            preprocessing_applied=data.get("preprocessing_applied"),
            status=status,
            created_at=created_at,
            provider_metadata=data.get("provider_metadata"),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0)
        )