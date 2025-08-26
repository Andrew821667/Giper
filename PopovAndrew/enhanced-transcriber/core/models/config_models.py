"""
Модели конфигурации для Enhanced Transcriber
Configuration data models
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path


class ModelProvider(Enum):
    """Провайдеры моделей транскрипции"""
    TONE = "tone"                 # T-one от VoiceKit
    WHISPER_LOCAL = "whisper_local"  # Локальный Whisper
    WHISPER_OPENAI = "whisper_openai"  # OpenAI Whisper API
    ENSEMBLE = "ensemble"         # Ансамбль моделей


class TranscriptionStrategy(Enum):
    """Стратегии транскрипции"""
    FAST = "fast"                     # Быстро (одна модель)
    BALANCED = "balanced"             # Сбалансированно (smart selection)
    MAXIMUM_QUALITY = "maximum_quality"  # Максимальное качество (ensemble)
    COST_OPTIMIZED = "cost_optimized"   # Оптимизация по стоимости


@dataclass
class ModelConfig:
    """
    Конфигурация отдельной модели
    Individual model configuration
    """
    name: str                        # Название модели
    provider: ModelProvider          # Провайдер
    language_support: List[str]      # Поддерживаемые языки
    max_file_size_mb: int           # Максимальный размер файла в МБ
    cost_per_minute: float          # Стоимость за минуту (в рублях)
    quality_score: int              # Оценка качества 1-10
    processing_speed: float         # Скорость обработки (x от real-time)
    
    # Технические характеристики
    optimal_sample_rate: int = 16000        # Оптимальная частота дискретизации
    supports_streaming: bool = False        # Поддержка потоковой обработки
    supports_timestamps: bool = True        # Поддержка временных меток
    supports_confidence: bool = True        # Поддержка confidence scores
    
    # Специализация
    best_for_domains: List[str] = field(default_factory=list)  # Лучшие домены
    best_for_audio_quality: List[str] = field(default_factory=list)  # Лучшее качество аудио
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "language_support": self.language_support,
            "max_file_size_mb": self.max_file_size_mb,
            "cost_per_minute": self.cost_per_minute,
            "quality_score": self.quality_score,
            "processing_speed": self.processing_speed,
            "optimal_sample_rate": self.optimal_sample_rate,
            "supports_streaming": self.supports_streaming,
            "supports_timestamps": self.supports_timestamps,
            "supports_confidence": self.supports_confidence,
            "best_for_domains": self.best_for_domains,
            "best_for_audio_quality": self.best_for_audio_quality
        }


@dataclass
class ECommerceConfig:
    """
    Конфигурация для e-commerce домена
    E-commerce domain configuration
    """
    # Словарь терминов для исправления
    term_corrections: Dict[str, List[str]] = field(default_factory=dict)
    
    # Важные термины для отслеживания
    important_terms: List[str] = field(default_factory=list)
    
    # Контекстные правила
    context_rules: Dict[str, str] = field(default_factory=dict)
    
    # Настройки постобработки
    enable_price_detection: bool = True      # Детекция цен
    enable_product_name_correction: bool = True  # Коррекция названий товаров
    enable_currency_normalization: bool = True   # Нормализация валют
    
    def get_default_ecommerce_terms(self) -> Dict[str, List[str]]:
        """Получение стандартных e-commerce терминов"""
        return {
            # Основные операции
            "заказ": ["закас", "зокас", "заказь", "закз"],
            "оплата": ["аплата", "оплото", "аплото", "оплать"],
            "доставка": ["доствка", "дастафка", "доставко", "достака"],
            "возврат": ["возрат", "вазврат", "возврот", "возрать"],
            
            # Товары и каталог
            "товар": ["тавар", "товорр", "тавор", "товор"],
            "каталог": ["каталок", "католог", "каталог", "каталог"],
            "ассортимент": ["асортимент", "ассартимент", "асартимент"],
            "скидка": ["скитка", "скидко", "скидка", "скитко"],
            "акция": ["акцыя", "акцие", "акция", "окция"],
            
            # Качество и сервис  
            "качество": ["качиство", "кочество", "качества", "качество"],
            "гарантия": ["горантия", "гарантея", "гарантия", "гарантие"],
            "отзыв": ["отзыф", "отзив", "отзыв", "отзыв"],
            "рейтинг": ["рэйтинг", "рейтынг", "ретинг", "рэйтынг"],
            
            # Цены и валюта
            "рубль": ["рубел", "рублей", "рублив", "рубль"],
            "цена": ["цэна", "цына", "цена", "цэно"],
            "стоимость": ["стамость", "стоимасть", "стоимость", "стамасть"],
            
            # Интернет-магазин специфика
            "корзина": ["карзина", "корзино", "корзина", "карзино"],
            "сайт": ["сайт", "саит", "сойт", "сайть"],
            "интернет": ["интэрнет", "интернэт", "интернет", "интэрнэт"],
            "онлайн": ["анлайн", "он-лайн", "онлайн", "анлайн"]
        }
    
    def __post_init__(self):
        """Инициализация значений по умолчанию"""
        if not self.term_corrections:
            self.term_corrections = self.get_default_ecommerce_terms()
        
        if not self.important_terms:
            self.important_terms = [
                "заказ", "оплата", "доставка", "возврат", "товар", 
                "скидка", "цена", "качество", "гарантия"
            ]
        
        if not self.context_rules:
            self.context_rules = {
                "заказать": "оформить заказ",
                "купить": "приобрести товар", 
                "продать": "реализовать товар",
                "вернуть": "осуществить возврат"
            }


@dataclass
class TranscriptionConfig:
    """
    Основная конфигурация транскрипции
    Main transcription configuration
    """
    # Основные настройки
    default_strategy: TranscriptionStrategy = TranscriptionStrategy.BALANCED
    default_language: str = "ru"
    quality_threshold: float = 0.8          # Минимальный порог качества
    
    # Модели
    available_models: Dict[str, ModelConfig] = field(default_factory=dict)
    model_selection_rules: Dict[str, str] = field(default_factory=dict)
    
    # Audio processing
    enable_audio_enhancement: bool = True
    enable_noise_reduction: bool = True
    enable_volume_normalization: bool = True
    target_sample_rate: int = 16000
    convert_to_mono: bool = True
    
    # Quality assessment
    enable_quality_assessment: bool = True
    auto_retry_on_low_quality: bool = True
    max_retry_attempts: int = 2
    
    # Domain configuration
    domain_configs: Dict[str, Any] = field(default_factory=dict)
    default_domain: str = "ecommerce"
    
    # Performance settings
    max_parallel_jobs: int = 2
    processing_timeout_minutes: int = 30
    enable_caching: bool = True
    cache_duration_hours: int = 24
    
    # Output settings
    include_word_timestamps: bool = True
    include_confidence_scores: bool = True
    include_quality_metrics: bool = True
    
    # API settings
    api_keys: Dict[str, str] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    
    def get_default_models(self) -> Dict[str, ModelConfig]:
        """Получение конфигураций моделей по умолчанию"""
        return {
            "tone_ru": ModelConfig(
                name="T-one Russian",
                provider=ModelProvider.TONE,
                language_support=["ru"],
                max_file_size_mb=100,
                cost_per_minute=0.0,  # Open source
                quality_score=9,      # Отлично для русского
                processing_speed=1.2,
                best_for_domains=["ecommerce", "business", "general"],
                best_for_audio_quality=["good", "excellent", "fair"]
            ),
            "whisper_local": ModelConfig(
                name="Whisper Local",
                provider=ModelProvider.WHISPER_LOCAL,
                language_support=["ru", "en", "multi"],
                max_file_size_mb=500,
                cost_per_minute=0.0,  # Локальная
                quality_score=7,
                processing_speed=0.8,
                best_for_domains=["general", "education"],
                best_for_audio_quality=["good", "excellent"]
            ),
            "whisper_openai": ModelConfig(
                name="Whisper OpenAI",
                provider=ModelProvider.WHISPER_OPENAI,
                language_support=["ru", "en", "multi"],
                max_file_size_mb=25,
                cost_per_minute=0.36,  # $0.006 * 60 руб
                quality_score=8,
                processing_speed=2.0,  # Быстрее через API
                best_for_domains=["general", "technical"],
                best_for_audio_quality=["excellent", "good"]
            )
        }
    
    def get_default_domain_configs(self) -> Dict[str, Any]:
        """Получение конфигураций доменов по умолчанию"""
        return {
            "ecommerce": ECommerceConfig(),
            "general": {
                "enable_post_processing": True,
                "common_corrections": {
                    "ну": ["нуу", "нуу"],
                    "это": ["эта", "ето"],
                    "конечно": ["канешно", "конешно"]
                }
            }
        }
    
    def __post_init__(self):
        """Инициализация значений по умолчанию"""
        if not self.available_models:
            self.available_models = self.get_default_models()
        
        if not self.domain_configs:
            self.domain_configs = self.get_default_domain_configs()
        
        if not self.model_selection_rules:
            self.model_selection_rules = {
                "russian_high_quality": "tone_ru",
                "multilingual": "whisper_openai", 
                "cost_sensitive": "whisper_local",
                "maximum_quality": "ensemble"
            }
    
    def get_optimal_model(
        self, 
        language: str, 
        audio_quality: str,
        domain: str = "general"
    ) -> str:
        """
        Выбор оптимальной модели
        Select optimal model based on criteria
        """
        if language == "ru":
            if audio_quality in ["good", "excellent"]:
                return "tone_ru"
            else:
                return "ensemble"  # T-one + Whisper для низкого качества
        elif language in ["en", "multi"]:
            return "whisper_openai"
        else:
            return "whisper_local"  # Fallback
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "default_strategy": self.default_strategy.value,
            "default_language": self.default_language,
            "quality_threshold": self.quality_threshold,
            "available_models": {k: v.to_dict() for k, v in self.available_models.items()},
            "model_selection_rules": self.model_selection_rules,
            "enable_audio_enhancement": self.enable_audio_enhancement,
            "enable_noise_reduction": self.enable_noise_reduction,
            "enable_volume_normalization": self.enable_volume_normalization,
            "target_sample_rate": self.target_sample_rate,
            "convert_to_mono": self.convert_to_mono,
            "enable_quality_assessment": self.enable_quality_assessment,
            "auto_retry_on_low_quality": self.auto_retry_on_low_quality,
            "max_retry_attempts": self.max_retry_attempts,
            "domain_configs": self.domain_configs,
            "default_domain": self.default_domain,
            "max_parallel_jobs": self.max_parallel_jobs,
            "processing_timeout_minutes": self.processing_timeout_minutes,
            "enable_caching": self.enable_caching,
            "cache_duration_hours": self.cache_duration_hours,
            "include_word_timestamps": self.include_word_timestamps,
            "include_confidence_scores": self.include_confidence_scores,
            "include_quality_metrics": self.include_quality_metrics,
            "api_keys": self.api_keys,
            "rate_limits": self.rate_limits
        }