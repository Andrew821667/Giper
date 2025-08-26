"""
Общий провайдер для всех Whisper моделей
Common provider for all Whisper models
"""

from typing import Dict, List, Optional, Any, Union
from .whisper_local import WhisperLocalTranscriber
from .whisper_openai import WhisperOpenAITranscriber


class WhisperProvider:
    """
    Фабрика провайдеров Whisper моделей
    Factory for Whisper model providers
    """
    
    LOCAL_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    OPENAI_MODELS = ["whisper-1"]
    
    @staticmethod
    def create_local_transcriber(
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: str = "int8"
    ) -> WhisperLocalTranscriber:
        """
        Создание локального Whisper транскрайбера
        
        Args:
            model_name: Размер модели
            device: Устройство (cpu, cuda, auto)
            compute_type: Тип вычислений
            
        Returns:
            WhisperLocalTranscriber: Готовый транскрайбер
        """
        if model_name not in WhisperProvider.LOCAL_MODELS:
            raise ValueError(f"Unsupported local model: {model_name}. Available: {WhisperProvider.LOCAL_MODELS}")
        
        return WhisperLocalTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type
        )
    
    @staticmethod
    def create_openai_transcriber(
        api_key: str,
        model: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1"
    ) -> WhisperOpenAITranscriber:
        """
        Создание OpenAI Whisper транскрайбера
        
        Args:
            api_key: API ключ OpenAI
            model: Модель
            base_url: Базовый URL API
            
        Returns:
            WhisperOpenAITranscriber: Готовый транскрайбер
        """
        if model not in WhisperProvider.OPENAI_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {model}. Available: {WhisperProvider.OPENAI_MODELS}")
        
        return WhisperOpenAITranscriber(
            api_key=api_key,
            model=model,
            base_url=base_url
        )
    
    @staticmethod
    def create_transcriber(
        provider_type: str,
        **kwargs
    ) -> Union[WhisperLocalTranscriber, WhisperOpenAITranscriber]:
        """
        Универсальное создание Whisper транскрайбера
        
        Args:
            provider_type: Тип провайдера ('local' или 'openai')
            **kwargs: Параметры для конкретного провайдера
            
        Returns:
            Whisper транскрайбер
        """
        if provider_type == "local":
            return WhisperProvider.create_local_transcriber(**kwargs)
        elif provider_type == "openai":
            return WhisperProvider.create_openai_transcriber(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. Available: 'local', 'openai'")
    
    @staticmethod
    def is_local_available() -> bool:
        """
        Проверка доступности локального Whisper
        
        Returns:
            bool: True если доступен
        """
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_openai_available(api_key: str) -> bool:
        """
        Проверка доступности OpenAI API (базовая проверка ключа)
        
        Args:
            api_key: API ключ
            
        Returns:
            bool: True если ключ задан
        """
        return bool(api_key and api_key.startswith('sk-'))
    
    @staticmethod
    def get_available_local_models() -> List[str]:
        """
        Получение доступных локальных моделей
        
        Returns:
            List[str]: Список моделей
        """
        return WhisperProvider.LOCAL_MODELS.copy()
    
    @staticmethod
    def get_available_openai_models() -> List[str]:
        """
        Получение доступных OpenAI моделей
        
        Returns:
            List[str]: Список моделей
        """
        return WhisperProvider.OPENAI_MODELS.copy()
    
    @staticmethod
    def get_model_recommendations(
        audio_quality: str,
        processing_time_priority: str = "balanced",
        cost_priority: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Получение рекомендаций по выбору модели
        
        Args:
            audio_quality: Качество аудио (poor, fair, good, excellent)
            processing_time_priority: Приоритет времени обработки (fast, balanced, quality)
            cost_priority: Приоритет стоимости (free, balanced, premium)
            
        Returns:
            Dict: Рекомендации по моделям
        """
        recommendations = {
            "primary_choice": None,
            "alternatives": [],
            "reasoning": ""
        }
        
        # Логика выбора на основе приоритетов
        if cost_priority == "free":
            # Только локальные модели
            if processing_time_priority == "fast":
                recommendations["primary_choice"] = {"provider": "local", "model": "tiny"}
                recommendations["alternatives"] = [
                    {"provider": "local", "model": "base"},
                ]
                recommendations["reasoning"] = "Fastest free option for quick transcription"
                
            elif processing_time_priority == "balanced":
                if audio_quality in ["poor", "fair"]:
                    recommendations["primary_choice"] = {"provider": "local", "model": "small"}
                else:
                    recommendations["primary_choice"] = {"provider": "local", "model": "base"}
                recommendations["alternatives"] = [
                    {"provider": "local", "model": "medium"},
                    {"provider": "local", "model": "tiny"}
                ]
                recommendations["reasoning"] = "Balanced free option with good quality"
                
            else:  # quality priority
                recommendations["primary_choice"] = {"provider": "local", "model": "large-v3"}
                recommendations["alternatives"] = [
                    {"provider": "local", "model": "large-v2"},
                    {"provider": "local", "model": "medium"}
                ]
                recommendations["reasoning"] = "Best free quality option"
                
        elif cost_priority == "premium":
            # Предпочтение OpenAI API для высокого качества
            recommendations["primary_choice"] = {"provider": "openai", "model": "whisper-1"}
            recommendations["alternatives"] = [
                {"provider": "local", "model": "large-v3"},
                {"provider": "local", "model": "medium"}
            ]
            recommendations["reasoning"] = "Premium cloud option with consistent quality"
            
        else:  # balanced cost
            # Комбинированный подход
            if audio_quality in ["excellent", "good"]:
                recommendations["primary_choice"] = {"provider": "local", "model": "base"}
                recommendations["alternatives"] = [
                    {"provider": "openai", "model": "whisper-1"},
                    {"provider": "local", "model": "small"}
                ]
            else:
                recommendations["primary_choice"] = {"provider": "openai", "model": "whisper-1"}
                recommendations["alternatives"] = [
                    {"provider": "local", "model": "medium"},
                    {"provider": "local", "model": "small"}
                ]
            recommendations["reasoning"] = "Balanced approach considering cost and quality"
        
        return recommendations
    
    @staticmethod
    def get_system_requirements() -> Dict[str, Dict[str, str]]:
        """
        Получение системных требований для разных моделей
        
        Returns:
            Dict: Системные требования
        """
        return {
            "local": {
                "python": ">=3.8",
                "memory": ">=4GB RAM (>=16GB for large models)",
                "disk_space": ">=2GB free space",
                "dependencies": "torch, whisper, ffmpeg",
                "gpu_optional": "CUDA for faster processing",
                "notes": "Large models require more VRAM"
            },
            "openai": {
                "python": ">=3.7",
                "memory": ">=1GB RAM",
                "disk_space": ">=100MB",
                "dependencies": "aiohttp",
                "internet": "Required for API access",
                "api_key": "OpenAI API key required",
                "rate_limits": "Check OpenAI pricing page"
            }
        }
    
    @staticmethod
    def estimate_processing_time(
        audio_duration_seconds: float,
        model_name: str,
        provider_type: str,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Оценка времени обработки
        
        Args:
            audio_duration_seconds: Длительность аудио в секундах
            model_name: Название модели
            provider_type: Тип провайдера
            device: Устройство (cpu/cuda)
            
        Returns:
            Dict: Оценки времени
        """
        if provider_type == "openai":
            return {
                "estimated_seconds": audio_duration_seconds * 0.1,  # ~10% от длительности
                "min_seconds": audio_duration_seconds * 0.05,
                "max_seconds": audio_duration_seconds * 0.3,
                "factors": ["Network latency", "API load"]
            }
        
        # Локальные модели - зависит от размера модели и устройства
        model_multipliers = {
            "tiny": 0.05,
            "base": 0.1,
            "small": 0.25,
            "medium": 0.5,
            "large": 1.0,
            "large-v2": 1.0,
            "large-v3": 1.0
        }
        
        base_multiplier = model_multipliers.get(model_name, 0.5)
        
        # Корректировка для GPU
        if device == "cuda":
            base_multiplier *= 0.3  # GPU примерно в 3 раза быстрее
        
        estimated_time = audio_duration_seconds * base_multiplier
        
        return {
            "estimated_seconds": estimated_time,
            "min_seconds": estimated_time * 0.7,
            "max_seconds": estimated_time * 2.0,
            "factors": ["Model size", "Device type", "System load"]
        }