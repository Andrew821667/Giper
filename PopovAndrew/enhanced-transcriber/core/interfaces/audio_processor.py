"""
Интерфейсы для обработки аудио
Abstract interfaces for audio processing
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from core.models.audio_metadata import AudioMetadata


class IAudioProcessor(ABC):
    """
    Абстрактный интерфейс для обработки аудио
    Abstract interface for audio processing
    """
    
    @abstractmethod
    async def enhance_audio(
        self, 
        input_file: str, 
        output_file: Optional[str] = None
    ) -> str:
        """
        Улучшение качества аудио
        Enhance audio quality
        
        Args:
            input_file: Путь к входному файлу
            output_file: Путь к выходному файлу (если None - генерируется автоматически)
            
        Returns:
            str: Путь к обработанному файлу
        """
        pass
    
    @abstractmethod
    def reduce_noise(
        self, 
        audio_data: Any, 
        sample_rate: int,
        reduction_strength: float = 0.8
    ) -> Any:
        """
        Шумоподавление
        Noise reduction
        
        Args:
            audio_data: Аудио данные (numpy array)
            sample_rate: Частота дискретизации
            reduction_strength: Сила подавления шума (0-1)
            
        Returns:
            Any: Обработанные аудио данные
        """
        pass
    
    @abstractmethod
    def normalize_volume(self, audio_data: Any, target_db: float = -20.0) -> Any:
        """
        Нормализация громкости
        Volume normalization
        
        Args:
            audio_data: Аудио данные
            target_db: Целевой уровень в дБ
            
        Returns:
            Any: Нормализованные аудио данные
        """
        pass
    
    @abstractmethod
    def resample_audio(
        self, 
        audio_data: Any, 
        original_sr: int, 
        target_sr: int
    ) -> Any:
        """
        Ресэмплинг аудио
        Resample audio to target sample rate
        
        Args:
            audio_data: Аудио данные
            original_sr: Исходная частота
            target_sr: Целевая частота
            
        Returns:
            Any: Ресэмплированные данные
        """
        pass


class IAudioAnalyzer(ABC):
    """
    Интерфейс для анализа аудио
    Interface for audio analysis
    """
    
    @abstractmethod
    async def analyze_audio(self, audio_file: str) -> AudioMetadata:
        """
        Комплексный анализ аудио файла
        Comprehensive audio file analysis
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            AudioMetadata: Метаданные и характеристики аудио
        """
        pass
    
    @abstractmethod
    def detect_language(self, audio_file: str) -> str:
        """
        Определение языка аудио
        Detect audio language
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            str: Код языка (ru, en, etc.)
        """
        pass
    
    @abstractmethod
    def assess_quality(self, audio_file: str) -> Dict[str, float]:
        """
        Оценка качества аудио
        Assess audio quality
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            Dict[str, float]: Метрики качества
        """
        pass
    
    @abstractmethod
    def detect_speech_segments(
        self, 
        audio_file: str
    ) -> List[Tuple[float, float]]:
        """
        Детекция речевых сегментов
        Detect speech segments in audio
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            List[Tuple[float, float]]: Список (start_time, end_time) для речевых сегментов
        """
        pass
    
    @abstractmethod
    def calculate_snr(self, audio_file: str) -> float:
        """
        Расчет отношения сигнал/шум
        Calculate Signal-to-Noise Ratio
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            float: SNR в дБ
        """
        pass
    
    @abstractmethod
    def estimate_duration(self, audio_file: str) -> float:
        """
        Оценка длительности аудио
        Estimate audio duration
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            float: Длительность в секундах
        """
        pass


class IAudioConverter(ABC):
    """
    Интерфейс для конвертации аудио форматов
    Interface for audio format conversion
    """
    
    @abstractmethod
    def convert_to_wav(
        self, 
        input_file: str, 
        output_file: str,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> str:
        """
        Конвертация в WAV формат
        Convert audio to WAV format
        
        Args:
            input_file: Входной файл
            output_file: Выходной файл
            sample_rate: Частота дискретизации
            channels: Количество каналов (1 - моно, 2 - стерео)
            
        Returns:
            str: Путь к конвертированному файлу
        """
        pass
    
    @abstractmethod
    def is_format_supported(self, file_path: str) -> bool:
        """
        Проверка поддержки формата
        Check if format is supported
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если формат поддерживается
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Получение списка поддерживаемых форматов
        Get list of supported formats
        
        Returns:
            List[str]: Список расширений файлов
        """
        pass