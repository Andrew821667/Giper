"""
Интерфейсы для транскрайберов
Abstract interfaces for transcription providers
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from core.models.transcription_result import TranscriptionResult
from core.models.audio_metadata import AudioMetadata


class ITranscriber(ABC):
    """
    Абстрактный интерфейс для транскрайберов
    Abstract interface for all transcription providers
    """
    
    @abstractmethod
    async def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Транскрипция аудио файла
        Transcribe audio file to text
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык аудио (ru, en, auto)
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат транскрипции
        """
        pass
    
    @abstractmethod
    def is_supported_format(self, file_path: str) -> bool:
        """
        Проверка поддерживаемого формата
        Check if file format is supported
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если формат поддерживается
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Информация о модели
        Get model information
        
        Returns:
            Dict: Информация о модели (name, version, capabilities)
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Название модели"""
        pass
    
    @property
    @abstractmethod 
    def supported_languages(self) -> List[str]:
        """Поддерживаемые языки"""
        pass


class ITranscriberProvider(ABC):
    """
    Интерфейс провайдера транскрайберов
    Provider interface for managing transcribers
    """
    
    @abstractmethod
    def get_transcriber(self, provider_name: str) -> ITranscriber:
        """
        Получение транскрайбера по имени
        Get transcriber by provider name
        
        Args:
            provider_name: Название провайдера (tone, whisper, openai)
            
        Returns:
            ITranscriber: Экземпляр транскрайбера
        """
        pass
    
    @abstractmethod
    def list_available_transcribers(self) -> List[str]:
        """
        Список доступных транскрайберов
        List all available transcribers
        
        Returns:
            List[str]: Список названий провайдеров
        """
        pass
    
    @abstractmethod
    async def health_check(self, provider_name: str) -> bool:
        """
        Проверка работоспособности провайдера
        Health check for provider
        
        Args:
            provider_name: Название провайдера
            
        Returns:
            bool: True если провайдер работает
        """
        pass


class IEnsembleTranscriber(ABC):
    """
    Интерфейс для ансамблевых транскрайберов
    Interface for ensemble transcription methods
    """
    
    @abstractmethod
    async def transcribe_ensemble(
        self,
        audio_file: str,
        providers: List[str],
        voting_method: str = "weighted",
        **kwargs
    ) -> TranscriptionResult:
        """
        Ансамблевая транскрипция
        Ensemble transcription using multiple providers
        
        Args:
            audio_file: Путь к аудио файлу
            providers: Список провайдеров для использования
            voting_method: Метод голосования (weighted, confidence, majority)
            **kwargs: Дополнительные параметры
            
        Returns:
            TranscriptionResult: Результат ансамблевой транскрипции
        """
        pass
    
    @abstractmethod
    def calculate_ensemble_confidence(
        self, 
        results: List[TranscriptionResult]
    ) -> float:
        """
        Расчет confidence для ансамбля
        Calculate ensemble confidence score
        
        Args:
            results: Список результатов от разных провайдеров
            
        Returns:
            float: Итоговый confidence score
        """
        pass