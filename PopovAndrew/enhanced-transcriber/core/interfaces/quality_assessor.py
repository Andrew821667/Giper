"""
Интерфейсы для оценки качества транскрипции
Abstract interfaces for transcription quality assessment
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from core.models.quality_metrics import QualityMetrics
from core.models.transcription_result import TranscriptionResult


class IQualityAssessor(ABC):
    """
    Интерфейс для оценки качества транскрипции
    Abstract interface for transcription quality assessment
    """
    
    @abstractmethod
    def assess_quality(
        self,
        transcription: str,
        reference_text: Optional[str] = None,
        audio_file: Optional[str] = None,
        domain: str = "general"
    ) -> QualityMetrics:
        """
        Комплексная оценка качества транскрипции
        Comprehensive transcription quality assessment
        
        Args:
            transcription: Текст транскрипции
            reference_text: Эталонный текст для сравнения (если есть)
            audio_file: Путь к оригинальному аудио файлу
            domain: Домен для специфической оценки (ecommerce, general, etc.)
            
        Returns:
            QualityMetrics: Метрики качества
        """
        pass
    
    @abstractmethod
    def calculate_wer(self, hypothesis: str, reference: str) -> float:
        """
        Расчет Word Error Rate (WER)
        Calculate Word Error Rate
        
        Args:
            hypothesis: Текст транскрипции
            reference: Эталонный текст
            
        Returns:
            float: WER значение (0-1, где 0 - идеально)
        """
        pass
    
    @abstractmethod
    def calculate_cer(self, hypothesis: str, reference: str) -> float:
        """
        Расчет Character Error Rate (CER)
        Calculate Character Error Rate
        
        Args:
            hypothesis: Текст транскрипции
            reference: Эталонный текст
            
        Returns:
            float: CER значение (0-1, где 0 - идеально)
        """
        pass
    
    @abstractmethod
    def assess_fluency(self, text: str, language: str = "ru") -> float:
        """
        Оценка беглости и естественности текста
        Assess text fluency and naturalness
        
        Args:
            text: Текст для оценки
            language: Язык текста
            
        Returns:
            float: Оценка беглости (0-1, где 1 - отлично)
        """
        pass
    
    @abstractmethod
    def assess_domain_accuracy(self, text: str, domain: str) -> float:
        """
        Оценка точности доменной терминологии
        Assess domain-specific terminology accuracy
        
        Args:
            text: Текст для оценки
            domain: Домен (ecommerce, medical, legal, etc.)
            
        Returns:
            float: Точность доменных терминов (0-1)
        """
        pass


class IQualityMetrics(ABC):
    """
    Интерфейс для работы с метриками качества
    Interface for quality metrics operations
    """
    
    @abstractmethod
    def calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Расчет общей оценки качества
        Calculate overall quality score
        
        Args:
            metrics: Словарь с отдельными метриками
            
        Returns:
            float: Общая оценка (0-1)
        """
        pass
    
    @abstractmethod
    def get_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """
        Генерация рекомендаций по улучшению
        Generate improvement recommendations
        
        Args:
            metrics: Метрики качества
            
        Returns:
            List[str]: Список рекомендаций
        """
        pass
    
    @abstractmethod
    def compare_results(
        self, 
        result1: TranscriptionResult, 
        result2: TranscriptionResult
    ) -> Dict[str, Any]:
        """
        Сравнение двух результатов транскрипции
        Compare two transcription results
        
        Args:
            result1: Первый результат
            result2: Второй результат
            
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        pass
    
    @abstractmethod
    def is_quality_acceptable(
        self, 
        metrics: QualityMetrics, 
        threshold: float = 0.8
    ) -> bool:
        """
        Проверка приемлемости качества
        Check if quality is acceptable
        
        Args:
            metrics: Метрики качества
            threshold: Пороговое значение
            
        Returns:
            bool: True если качество приемлемо
        """
        pass


class IDomainSpecificAssessor(ABC):
    """
    Интерфейс для доменной оценки качества
    Interface for domain-specific quality assessment
    """
    
    @abstractmethod
    def load_domain_dictionary(self, domain: str) -> Dict[str, List[str]]:
        """
        Загрузка доменного словаря
        Load domain-specific dictionary
        
        Args:
            domain: Название домена
            
        Returns:
            Dict[str, List[str]]: Словарь терминов и их вариантов
        """
        pass
    
    @abstractmethod
    def validate_domain_terms(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """
        Валидация доменных терминов
        Validate domain-specific terms
        
        Args:
            text: Текст для проверки
            domain: Домен
            
        Returns:
            List[Dict[str, Any]]: Список найденных/отсутствующих терминов
        """
        pass
    
    @abstractmethod
    def suggest_corrections(
        self, 
        text: str, 
        domain: str
    ) -> List[Dict[str, str]]:
        """
        Предложение исправлений
        Suggest corrections for domain terms
        
        Args:
            text: Текст с возможными ошибками
            domain: Домен
            
        Returns:
            List[Dict[str, str]]: Список предложений {original: corrected}
        """
        pass