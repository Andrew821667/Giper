"""
Модели данных для метрик качества транскрипции
Data models for transcription quality metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class QualityLevel(Enum):
    """Уровни качества"""
    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"           # 0.7-0.9
    FAIR = "fair"           # 0.5-0.7
    POOR = "poor"           # 0.3-0.5
    VERY_POOR = "very_poor" # <0.3


class DomainType(Enum):
    """Типы доменов для специализированной оценки"""
    ECOMMERCE = "ecommerce"         # Интернет-торговля
    GENERAL = "general"             # Общий
    BUSINESS = "business"           # Бизнес
    EDUCATION = "education"         # Образование
    MEDICAL = "medical"             # Медицина
    LEGAL = "legal"                # Юриспруденция
    TECHNICAL = "technical"         # Техническая документация


@dataclass
class DomainAccuracy:
    """
    Точность доменной терминологии
    Domain-specific terminology accuracy
    """
    domain: DomainType                          # Тип домена
    total_terms_found: int                      # Всего найдено терминов
    correct_terms: int                          # Правильно распознанных
    incorrect_terms: int                        # Неправильно распознанных
    missed_terms: int                          # Пропущенных терминов
    accuracy_score: float                      # Общая точность (0-1)
    
    # Детали по терминам
    correct_terms_list: List[str] = field(default_factory=list)
    incorrect_terms_list: List[Dict[str, str]] = field(default_factory=list)  # {found: expected}
    missed_terms_list: List[str] = field(default_factory=list)
    
    @property
    def precision(self) -> float:
        """Precision: правильные / (правильные + неправильные)"""
        total_recognized = self.correct_terms + self.incorrect_terms
        if total_recognized == 0:
            return 0.0
        return self.correct_terms / total_recognized
    
    @property
    def recall(self) -> float:
        """Recall: правильные / (правильные + пропущенные)"""
        total_should_be_found = self.correct_terms + self.missed_terms
        if total_should_be_found == 0:
            return 1.0
        return self.correct_terms / total_should_be_found
    
    @property
    def f1_score(self) -> float:
        """F1-Score: гармоническое среднее precision и recall"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "domain": self.domain.value,
            "total_terms_found": self.total_terms_found,
            "correct_terms": self.correct_terms,
            "incorrect_terms": self.incorrect_terms,
            "missed_terms": self.missed_terms,
            "accuracy_score": self.accuracy_score,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "correct_terms_list": self.correct_terms_list,
            "incorrect_terms_list": self.incorrect_terms_list,
            "missed_terms_list": self.missed_terms_list
        }


@dataclass
class QualityMetrics:
    """
    Комплексные метрики качества транскрипции
    Comprehensive transcription quality metrics
    """
    
    # Основные метрики точности
    word_error_rate: Optional[float] = None      # WER (0-1, где 0 - идеально)
    character_error_rate: Optional[float] = None # CER (0-1, где 0 - идеально)
    semantic_similarity: Optional[float] = None  # Семантическое сходство (0-1)
    
    # Качественные оценки
    fluency_score: float = 0.0                  # Беглость речи (0-1)
    naturalness_score: float = 0.0              # Естественность (0-1)
    readability_score: float = 0.0              # Читабельность (0-1)
    
    # Пунктуация и форматирование
    punctuation_accuracy: float = 0.0           # Точность пунктуации (0-1)
    capitalization_accuracy: float = 0.0        # Точность заглавных букв (0-1)
    sentence_structure_score: float = 0.0       # Качество структуры предложений (0-1)
    
    # Доменная специфика
    domain_accuracy: Optional[DomainAccuracy] = None  # Точность доменной терминологии
    
    # Confidence метрики
    average_word_confidence: Optional[float] = None   # Средняя уверенность по словам
    low_confidence_words_count: int = 0              # Количество слов с низкой уверенностью
    low_confidence_percentage: float = 0.0           # Процент слов с низкой уверенностью
    
    # Метрики содержания
    word_count: int = 0                             # Количество слов
    unique_words_count: int = 0                     # Количество уникальных слов
    vocabulary_richness: float = 0.0                # Богатство словаря (unique/total)
    
    # Временные метрики
    speech_rate_wpm: Optional[float] = None         # Скорость речи (слов в минуту)
    pause_analysis: Optional[Dict[str, float]] = None # Анализ пауз
    
    # Общая оценка
    overall_score: float = 0.0                      # Общая оценка качества (0-1)
    quality_level: QualityLevel = QualityLevel.FAIR # Уровень качества
    
    # Метаданные оценки
    evaluated_at: datetime = field(default_factory=datetime.now)
    evaluation_method: str = "automatic"            # automatic, manual, hybrid
    reference_available: bool = False               # Был ли доступен reference text
    
    # Рекомендации
    improvement_suggestions: List[str] = field(default_factory=list)
    needs_manual_review: bool = False
    retry_recommended: bool = False
    
    def calculate_overall_score(self) -> float:
        """
        Расчет общей оценки качества
        Calculate overall quality score
        """
        scores = []
        weights = []
        
        # Основные метрики (если доступны)
        if self.word_error_rate is not None:
            scores.append(1 - self.word_error_rate)  # Инвертируем WER
            weights.append(0.3)
        
        if self.character_error_rate is not None:
            scores.append(1 - self.character_error_rate)  # Инвертируем CER
            weights.append(0.2)
        
        if self.semantic_similarity is not None:
            scores.append(self.semantic_similarity)
            weights.append(0.2)
        
        # Качественные метрики
        scores.extend([
            self.fluency_score,
            self.naturalness_score,
            self.punctuation_accuracy
        ])
        weights.extend([0.1, 0.1, 0.05])
        
        # Доменная точность
        if self.domain_accuracy:
            scores.append(self.domain_accuracy.accuracy_score)
            weights.append(0.15)
        
        # Confidence метрики
        if self.average_word_confidence:
            scores.append(self.average_word_confidence)
            weights.append(0.1)
        
        # Нормализация весов
        if weights:
            total_weight = sum(weights[:len(scores)])
            normalized_weights = [w/total_weight for w in weights[:len(scores)]]
            
            # Взвешенное среднее
            weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
            return max(0.0, min(1.0, weighted_score))
        
        return 0.0
    
    def update_overall_assessment(self):
        """Обновление общей оценки и уровня качества"""
        self.overall_score = self.calculate_overall_score()
        
        # Определение уровня качества
        if self.overall_score >= 0.9:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.overall_score >= 0.7:
            self.quality_level = QualityLevel.GOOD
        elif self.overall_score >= 0.5:
            self.quality_level = QualityLevel.FAIR
        elif self.overall_score >= 0.3:
            self.quality_level = QualityLevel.POOR
        else:
            self.quality_level = QualityLevel.VERY_POOR
        
        # Генерация рекомендаций
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Генерация рекомендаций по улучшению"""
        self.improvement_suggestions.clear()
        
        if self.punctuation_accuracy < 0.7:
            self.improvement_suggestions.append(
                "Рекомендуется ручная корректировка пунктуации"
            )
        
        if self.domain_accuracy and self.domain_accuracy.accuracy_score < 0.8:
            self.improvement_suggestions.append(
                "Проверьте корректность специальных терминов"
            )
        
        if self.low_confidence_percentage > 0.3:
            self.improvement_suggestions.append(
                "Высокий процент слов с низкой уверенностью - проверьте качество аудио"
            )
        
        if self.fluency_score < 0.6:
            self.improvement_suggestions.append(
                "Текст требует редактирования для улучшения читабельности"
            )
        
        if self.overall_score < 0.6:
            self.retry_recommended = True
            self.improvement_suggestions.append(
                "Рекомендуется повторная транскрипция с другими параметрами"
            )
        
        if self.overall_score < 0.4:
            self.needs_manual_review = True
            self.improvement_suggestions.append(
                "Требуется ручная проверка и корректировка"
            )
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Получение детального отчета о качестве
        Get detailed quality report
        """
        return {
            "summary": {
                "overall_score": self.overall_score,
                "quality_level": self.quality_level.value,
                "needs_manual_review": self.needs_manual_review,
                "retry_recommended": self.retry_recommended
            },
            "accuracy_metrics": {
                "word_error_rate": self.word_error_rate,
                "character_error_rate": self.character_error_rate,
                "semantic_similarity": self.semantic_similarity
            },
            "quality_scores": {
                "fluency": self.fluency_score,
                "naturalness": self.naturalness_score,
                "readability": self.readability_score,
                "punctuation": self.punctuation_accuracy,
                "capitalization": self.capitalization_accuracy
            },
            "confidence_analysis": {
                "average_confidence": self.average_word_confidence,
                "low_confidence_words": self.low_confidence_words_count,
                "low_confidence_percentage": self.low_confidence_percentage
            },
            "domain_analysis": self.domain_accuracy.to_dict() if self.domain_accuracy else None,
            "content_analysis": {
                "word_count": self.word_count,
                "unique_words": self.unique_words_count,
                "vocabulary_richness": self.vocabulary_richness,
                "speech_rate_wpm": self.speech_rate_wpm
            },
            "recommendations": self.improvement_suggestions,
            "metadata": {
                "evaluated_at": self.evaluated_at.isoformat(),
                "evaluation_method": self.evaluation_method,
                "reference_available": self.reference_available
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON serialization"""
        return {
            "word_error_rate": self.word_error_rate,
            "character_error_rate": self.character_error_rate,
            "semantic_similarity": self.semantic_similarity,
            "fluency_score": self.fluency_score,
            "naturalness_score": self.naturalness_score,
            "readability_score": self.readability_score,
            "punctuation_accuracy": self.punctuation_accuracy,
            "capitalization_accuracy": self.capitalization_accuracy,
            "sentence_structure_score": self.sentence_structure_score,
            "domain_accuracy": self.domain_accuracy.to_dict() if self.domain_accuracy else None,
            "average_word_confidence": self.average_word_confidence,
            "low_confidence_words_count": self.low_confidence_words_count,
            "low_confidence_percentage": self.low_confidence_percentage,
            "word_count": self.word_count,
            "unique_words_count": self.unique_words_count,
            "vocabulary_richness": self.vocabulary_richness,
            "speech_rate_wpm": self.speech_rate_wpm,
            "pause_analysis": self.pause_analysis,
            "overall_score": self.overall_score,
            "quality_level": self.quality_level.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evaluation_method": self.evaluation_method,
            "reference_available": self.reference_available,
            "improvement_suggestions": self.improvement_suggestions,
            "needs_manual_review": self.needs_manual_review,
            "retry_recommended": self.retry_recommended,
            "quality_report": self.get_quality_report()
        }