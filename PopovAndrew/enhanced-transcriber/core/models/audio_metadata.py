"""
Модели данных для аудио метаданных
Data models for audio metadata and analysis
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from enum import Enum
from pathlib import Path


class AudioFormat(Enum):
    """Форматы аудио файлов"""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"
    UNKNOWN = "unknown"


class AudioQuality(Enum):
    """Уровни качества аудио"""
    EXCELLENT = "excellent"  # > 0.9
    GOOD = "good"           # 0.7 - 0.9
    FAIR = "fair"           # 0.5 - 0.7
    POOR = "poor"           # 0.3 - 0.5
    VERY_POOR = "very_poor" # < 0.3


@dataclass
class SpeechSegment:
    """
    Сегмент речи в аудио
    Speech segment information
    """
    start_time: float        # Время начала (секунды)
    end_time: float         # Время окончания (секунды)
    confidence: float       # Уверенность детекции речи (0-1)
    speaker_id: Optional[str] = None  # ID говорящего (если есть)
    
    @property
    def duration(self) -> float:
        """Длительность сегмента"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id
        }


@dataclass
class AudioMetadata:
    """
    Метаданные и характеристики аудио файла
    Complete audio file metadata and analysis
    """
    file_path: str                    # Путь к файлу
    file_size: int                    # Размер файла в байтах
    duration: float                   # Длительность в секундах
    sample_rate: int                  # Частота дискретизации (Hz)
    channels: int                     # Количество каналов
    format: AudioFormat               # Формат файла
    bit_depth: Optional[int] = None   # Битность (16, 24, 32)
    
    # Качественные характеристики
    overall_quality: AudioQuality = AudioQuality.FAIR
    quality_score: float = 0.5        # Общий score качества (0-1)
    snr_db: Optional[float] = None    # Отношение сигнал/шум в дБ
    
    # Анализ речи
    speech_segments: List[SpeechSegment] = field(default_factory=list)
    total_speech_duration: float = 0.0      # Общая длительность речи
    speech_to_silence_ratio: float = 0.0    # Соотношение речь/тишина
    
    # Детекция языка и содержимого
    detected_language: str = "unknown"       # Определенный язык
    language_confidence: float = 0.0         # Уверенность в языке
    estimated_speakers_count: int = 1        # Оценка количества говорящих
    
    # Технические характеристики
    average_volume_db: Optional[float] = None    # Средняя громкость в дБ
    peak_volume_db: Optional[float] = None       # Пиковая громкость в дБ
    dynamic_range_db: Optional[float] = None     # Динамический диапазон
    
    # Анализ шума
    background_noise_level: float = 0.0     # Уровень фонового шума (0-1)
    noise_type: Optional[str] = None        # Тип шума (если определен)
    
    # Рекомендации для обработки
    needs_noise_reduction: bool = False      # Нужно ли шумоподавление
    needs_volume_normalization: bool = False # Нужна ли нормализация громкости
    optimal_transcription_model: Optional[str] = None  # Рекомендуемая модель
    
    # Метаданные анализа
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_duration: float = 0.0          # Время анализа в секундах
    
    @property
    def file_name(self) -> str:
        """Имя файла без пути"""
        return Path(self.file_path).name
    
    @property
    def file_extension(self) -> str:
        """Расширение файла"""
        return Path(self.file_path).suffix.lower().lstrip('.')
    
    @property
    def duration_formatted(self) -> str:
        """Форматированная длительность (MM:SS)"""
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def file_size_mb(self) -> float:
        """Размер файла в МБ"""
        return self.file_size / (1024 * 1024)
    
    @property
    def bitrate_kbps(self) -> Optional[int]:
        """Битрейт в кбит/с"""
        if self.bit_depth and self.duration > 0:
            total_bits = self.file_size * 8
            return int(total_bits / self.duration / 1000)
        return None
    
    @property 
    def is_mono(self) -> bool:
        """Моно ли аудио"""
        return self.channels == 1
    
    @property
    def is_stereo(self) -> bool:
        """Стерео ли аудио"""
        return self.channels == 2
    
    @property
    def speech_percentage(self) -> float:
        """Процент речи от общей длительности"""
        if self.duration > 0:
            return (self.total_speech_duration / self.duration) * 100
        return 0.0
    
    def is_suitable_for_transcription(self) -> bool:
        """
        Подходит ли файл для транскрипции
        Check if audio is suitable for transcription
        """
        criteria = [
            self.duration >= 1.0,                    # Минимум 1 секунда
            self.speech_percentage >= 10.0,          # Минимум 10% речи
            self.overall_quality != AudioQuality.VERY_POOR,  # Не очень плохое качество
            self.sample_rate >= 8000,                # Минимум 8kHz
        ]
        
        return all(criteria)
    
    def get_transcription_recommendations(self) -> Dict[str, Any]:
        """
        Получение рекомендаций для транскрипции
        Get transcription recommendations
        """
        recommendations = {
            "preprocessing_needed": [],
            "optimal_model": None,
            "expected_quality": "medium",
            "estimated_processing_time": self.duration * 0.1  # ~10% от длительности
        }
        
        # Рекомендации по предобработке
        if self.needs_noise_reduction:
            recommendations["preprocessing_needed"].append("noise_reduction")
        
        if self.needs_volume_normalization:
            recommendations["preprocessing_needed"].append("volume_normalization")
        
        if self.sample_rate < 16000:
            recommendations["preprocessing_needed"].append("resample_to_16khz")
        
        if not self.is_mono:
            recommendations["preprocessing_needed"].append("convert_to_mono")
        
        # Рекомендация модели
        if self.detected_language == "ru" and self.overall_quality in [AudioQuality.GOOD, AudioQuality.EXCELLENT]:
            recommendations["optimal_model"] = "tone"
            recommendations["expected_quality"] = "high"
        elif self.detected_language == "ru":
            recommendations["optimal_model"] = "ensemble"  # T-one + Whisper
            recommendations["expected_quality"] = "medium"
        else:
            recommendations["optimal_model"] = "whisper"
            recommendations["expected_quality"] = "medium"
        
        # Корректировка времени обработки
        if recommendations["optimal_model"] == "ensemble":
            recommendations["estimated_processing_time"] *= 2
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON serialization"""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_size_mb": self.file_size_mb,
            "duration": self.duration,
            "duration_formatted": self.duration_formatted,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format.value,
            "bit_depth": self.bit_depth,
            "bitrate_kbps": self.bitrate_kbps,
            "overall_quality": self.overall_quality.value,
            "quality_score": self.quality_score,
            "snr_db": self.snr_db,
            "speech_segments": [segment.to_dict() for segment in self.speech_segments],
            "total_speech_duration": self.total_speech_duration,
            "speech_to_silence_ratio": self.speech_to_silence_ratio,
            "speech_percentage": self.speech_percentage,
            "detected_language": self.detected_language,
            "language_confidence": self.language_confidence,
            "estimated_speakers_count": self.estimated_speakers_count,
            "average_volume_db": self.average_volume_db,
            "peak_volume_db": self.peak_volume_db,
            "dynamic_range_db": self.dynamic_range_db,
            "background_noise_level": self.background_noise_level,
            "noise_type": self.noise_type,
            "needs_noise_reduction": self.needs_noise_reduction,
            "needs_volume_normalization": self.needs_volume_normalization,
            "optimal_transcription_model": self.optimal_transcription_model,
            "analyzed_at": self.analyzed_at.isoformat(),
            "analysis_duration": self.analysis_duration,
            "is_suitable_for_transcription": self.is_suitable_for_transcription(),
            "transcription_recommendations": self.get_transcription_recommendations()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioMetadata':
        """Создание экземпляра из словаря"""
        # Конвертация speech_segments
        speech_segments = []
        if data.get("speech_segments"):
            speech_segments = [
                SpeechSegment(**segment_data)
                for segment_data in data["speech_segments"]
            ]
        
        # Конвертация enum'ов
        audio_format = AudioFormat.UNKNOWN
        if data.get("format"):
            try:
                audio_format = AudioFormat(data["format"])
            except ValueError:
                audio_format = AudioFormat.UNKNOWN
        
        overall_quality = AudioQuality.FAIR
        if data.get("overall_quality"):
            try:
                overall_quality = AudioQuality(data["overall_quality"])
            except ValueError:
                overall_quality = AudioQuality.FAIR
        
        # Конвертация даты
        analyzed_at = datetime.now()
        if data.get("analyzed_at"):
            analyzed_at = datetime.fromisoformat(data["analyzed_at"])
        
        return cls(
            file_path=data["file_path"],
            file_size=data["file_size"],
            duration=data["duration"],
            sample_rate=data["sample_rate"],
            channels=data["channels"],
            format=audio_format,
            bit_depth=data.get("bit_depth"),
            overall_quality=overall_quality,
            quality_score=data.get("quality_score", 0.5),
            snr_db=data.get("snr_db"),
            speech_segments=speech_segments,
            total_speech_duration=data.get("total_speech_duration", 0.0),
            speech_to_silence_ratio=data.get("speech_to_silence_ratio", 0.0),
            detected_language=data.get("detected_language", "unknown"),
            language_confidence=data.get("language_confidence", 0.0),
            estimated_speakers_count=data.get("estimated_speakers_count", 1),
            average_volume_db=data.get("average_volume_db"),
            peak_volume_db=data.get("peak_volume_db"),
            dynamic_range_db=data.get("dynamic_range_db"),
            background_noise_level=data.get("background_noise_level", 0.0),
            noise_type=data.get("noise_type"),
            needs_noise_reduction=data.get("needs_noise_reduction", False),
            needs_volume_normalization=data.get("needs_volume_normalization", False),
            optimal_transcription_model=data.get("optimal_transcription_model"),
            analyzed_at=analyzed_at,
            analysis_duration=data.get("analysis_duration", 0.0)
        )