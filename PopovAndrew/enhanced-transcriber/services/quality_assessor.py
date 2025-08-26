"""
Система оценки качества транскрипции для достижения целевого качества 95%+
Quality assessment system for achieving 95%+ transcription quality
"""

import asyncio
import time
import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from collections import Counter
import statistics

from core.interfaces.quality_assessor import IQualityAssessor
from core.models.quality_metrics import QualityMetrics, QualityLevel, DomainAccuracy, DomainType
from core.models.config_models import ECommerceConfig

logger = logging.getLogger(__name__)


class QualityAssessmentService(IQualityAssessor):
    """
    Сервис оценки качества транскрипции
    Quality assessment service for transcription evaluation
    """
    
    def __init__(
        self,
        use_reference_text: bool = False,
        enable_semantic_analysis: bool = True,
        enable_domain_analysis: bool = True,
        confidence_threshold: float = 0.8,
        ecommerce_config: Optional[ECommerceConfig] = None
    ):
        """
        Инициализация сервиса оценки качества
        
        Args:
            use_reference_text: Использовать эталонный текст для WER/CER
            enable_semantic_analysis: Включить семантический анализ
            enable_domain_analysis: Включить доменный анализ
            confidence_threshold: Порог уверенности для low-confidence анализа
            ecommerce_config: Конфигурация e-commerce домена
        """
        self.use_reference_text = use_reference_text
        self.enable_semantic_analysis = enable_semantic_analysis
        self.enable_domain_analysis = enable_domain_analysis
        self.confidence_threshold = confidence_threshold
        self.ecommerce_config = ecommerce_config or ECommerceConfig()
        
        # Проверка доступности библиотек
        self._check_dependencies()
        
        # Инициализация моделей
        self._initialize_models()
    
    def _check_dependencies(self):
        """Проверка доступности библиотек для анализа"""
        self.jiwer_available = False
        self.sentence_transformers_available = False
        self.nltk_available = False
        self.pymorphy2_available = False
        
        try:
            import jiwer
            self.jiwer_available = True
            logger.info("✅ jiwer available for WER/CER calculation")
        except ImportError:
            logger.warning("⚠️ jiwer not available - WER/CER calculation disabled")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformers_available = True
            logger.info("✅ sentence-transformers available for semantic analysis")
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available - semantic analysis disabled")
        
        try:
            import nltk
            self.nltk_available = True
            logger.info("✅ nltk available for text analysis")
        except ImportError:
            logger.warning("⚠️ nltk not available - advanced text analysis disabled")
        
        try:
            import pymorphy2
            self.pymorphy2_available = True
            logger.info("✅ pymorphy2 available for Russian morphology")
        except ImportError:
            logger.warning("⚠️ pymorphy2 not available - Russian morphology disabled")
    
    def _initialize_models(self):
        """Инициализация моделей для анализа"""
        self.sentence_model = None
        self.morph_analyzer = None
        
        if self.sentence_transformers_available:
            try:
                from sentence_transformers import SentenceTransformer
                # Используем многоязычную модель для русского языка
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("🤖 Semantic similarity model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
        
        if self.pymorphy2_available:
            try:
                import pymorphy2
                self.morph_analyzer = pymorphy2.MorphAnalyzer()
                logger.info("🔤 Russian morphology analyzer loaded")
            except Exception as e:
                logger.warning(f"Failed to load pymorphy2: {e}")
    
    async def assess_quality(
        self,
        transcribed_text: str,
        audio_file: str,
        reference_text: Optional[str] = None,
        domain: str = "general",
        **kwargs
    ) -> QualityMetrics:
        """
        Оценка качества транскрипции
        
        Args:
            transcribed_text: Транскрибированный текст
            audio_file: Путь к аудио файлу
            reference_text: Эталонный текст (опционально)
            domain: Домен для специализированного анализа
            **kwargs: Дополнительные параметры
            
        Returns:
            QualityMetrics: Метрики качества
        """
        if not transcribed_text.strip():
            return self._create_empty_quality_metrics("Empty transcription text")
        
        start_time = time.time()
        logger.info(f"📊 Starting quality assessment for {len(transcribed_text)} characters")
        
        # Инициализация метрик
        metrics = QualityMetrics()
        
        try:
            # 1. Базовые метрики точности (WER/CER)
            if reference_text and self.jiwer_available:
                await self._calculate_error_rates(transcribed_text, reference_text, metrics)
            
            # 2. Семантический анализ
            if self.enable_semantic_analysis:
                await self._analyze_semantic_quality(transcribed_text, reference_text, metrics)
            
            # 3. Качественные метрики
            await self._analyze_text_quality(transcribed_text, metrics)
            
            # 4. Доменная специализация
            if self.enable_domain_analysis:
                await self._analyze_domain_accuracy(transcribed_text, domain, metrics)
            
            # 5. Confidence анализ
            word_confidences = kwargs.get('word_confidences', [])
            if word_confidences:
                await self._analyze_confidence_metrics(word_confidences, metrics)
            
            # 6. Анализ содержания
            await self._analyze_content_metrics(transcribed_text, metrics)
            
            # 7. Временные метрики
            audio_duration = kwargs.get('audio_duration')
            if audio_duration:
                await self._analyze_temporal_metrics(transcribed_text, audio_duration, metrics)
            
            # Финальная оценка
            metrics.update_overall_assessment()
            
            # Метаданные оценки
            metrics.evaluation_method = "automatic_comprehensive"
            metrics.reference_available = reference_text is not None
            
            processing_time = time.time() - start_time
            logger.info(
                f"✅ Quality assessment completed: {metrics.overall_score:.3f} "
                f"({metrics.quality_level.value}) in {processing_time:.1f}s"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return self._create_empty_quality_metrics(f"Assessment error: {str(e)}")
    
    async def _calculate_error_rates(self, transcribed: str, reference: str, metrics: QualityMetrics):
        """Расчет WER и CER"""
        try:
            import jiwer
            
            # Нормализация текстов
            transcribed_norm = self._normalize_text_for_comparison(transcribed)
            reference_norm = self._normalize_text_for_comparison(reference)
            
            # Word Error Rate
            wer = jiwer.wer(reference_norm, transcribed_norm)
            metrics.word_error_rate = min(1.0, max(0.0, wer))
            
            # Character Error Rate  
            cer = jiwer.cer(reference_norm, transcribed_norm)
            metrics.character_error_rate = min(1.0, max(0.0, cer))
            
            logger.info(f"📏 Error rates: WER={wer:.3f}, CER={cer:.3f}")
            
        except Exception as e:
            logger.warning(f"Error rate calculation failed: {e}")
    
    async def _analyze_semantic_quality(self, transcribed: str, reference: Optional[str], metrics: QualityMetrics):
        """Семантический анализ качества"""
        try:
            if not self.sentence_model or not reference:
                return
            
            # Векторное представление текстов
            transcribed_embedding = self.sentence_model.encode([transcribed])
            reference_embedding = self.sentence_model.encode([reference])
            
            # Косинусное сходство
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(transcribed_embedding, reference_embedding)[0][0]
            
            metrics.semantic_similarity = float(similarity)
            logger.info(f"🧠 Semantic similarity: {similarity:.3f}")
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
    
    async def _analyze_text_quality(self, text: str, metrics: QualityMetrics):
        """Анализ качественных характеристик текста"""
        # Беглость речи (fluency)
        metrics.fluency_score = self._calculate_fluency_score(text)
        
        # Естественность (naturalness)
        metrics.naturalness_score = self._calculate_naturalness_score(text)
        
        # Читабельность (readability)
        metrics.readability_score = self._calculate_readability_score(text)
        
        # Пунктуация
        metrics.punctuation_accuracy = self._calculate_punctuation_accuracy(text)
        
        # Заглавные буквы
        metrics.capitalization_accuracy = self._calculate_capitalization_accuracy(text)
        
        # Структура предложений
        metrics.sentence_structure_score = self._calculate_sentence_structure_score(text)
        
        logger.info(f"📝 Text quality: fluency={metrics.fluency_score:.2f}, naturalness={metrics.naturalness_score:.2f}")
    
    def _calculate_fluency_score(self, text: str) -> float:
        """Расчет беглости речи"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return 0.0
        
        scores = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                continue
            
            # Факторы беглости
            factors = []
            
            # Длина предложения (оптимальная 8-20 слов)
            length_score = self._score_sentence_length(len(words))
            factors.append(length_score)
            
            # Повторения слов
            repetition_score = self._score_word_repetitions(words)
            factors.append(repetition_score)
            
            # Переходы между словами
            transition_score = self._score_word_transitions(words)
            factors.append(transition_score)
            
            if factors:
                scores.append(statistics.mean(factors))
        
        return statistics.mean(scores) if scores else 0.5
    
    def _calculate_naturalness_score(self, text: str) -> float:
        """Расчет естественности текста"""
        # Факторы естественности
        factors = []
        
        # Наличие филлеров и паразитов слов
        filler_score = self._score_filler_words(text)
        factors.append(filler_score)
        
        # Разнообразие словаря
        vocabulary_score = self._score_vocabulary_diversity(text)
        factors.append(vocabulary_score)
        
        # Грамматическая корректность (базовая)
        grammar_score = self._score_basic_grammar(text)
        factors.append(grammar_score)
        
        return statistics.mean(factors) if factors else 0.5
    
    def _calculate_readability_score(self, text: str) -> float:
        """Расчет читабельности"""
        sentences = self._split_into_sentences(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Средняя длина предложения
        avg_sentence_length = len(words) / len(sentences)
        
        # Оценка на основе длины предложений
        if 5 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif avg_sentence_length < 5:
            length_score = avg_sentence_length / 5.0
        else:
            length_score = max(0.3, 20.0 / avg_sentence_length)
        
        # Сложность слов (примерная)
        complexity_score = self._score_word_complexity(words)
        
        return (length_score + complexity_score) / 2.0
    
    def _calculate_punctuation_accuracy(self, text: str) -> float:
        """Оценка точности пунктуации"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return 0.0
        
        correct_punctuation = 0
        total_sentences = len(sentences)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Проверка окончания предложения
            if sentence.endswith(('.', '!', '?')):
                correct_punctuation += 1
            
            # Проверка корректности запятых (базовая)
            if self._has_reasonable_comma_usage(sentence):
                correct_punctuation += 0.5
        
        return min(1.0, correct_punctuation / total_sentences) if total_sentences > 0 else 0.0
    
    def _calculate_capitalization_accuracy(self, text: str) -> float:
        """Оценка корректности заглавных букв"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return 0.0
        
        correct_capitalization = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Первая буква предложения должна быть заглавной
            if sentence[0].isupper():
                correct_capitalization += 1
        
        return correct_capitalization / len(sentences) if sentences else 0.0
    
    def _calculate_sentence_structure_score(self, text: str) -> float:
        """Оценка структуры предложений"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return 0.0
        
        structure_scores = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                structure_scores.append(0.3)
                continue
            
            # Базовая оценка структуры
            score = 0.7  # Базовый score
            
            # Бонус за наличие глаголов (примерная эвристика)
            if self._has_verb_like_words(words):
                score += 0.2
            
            # Штраф за очень короткие или длинные предложения
            if len(words) < 3:
                score -= 0.3
            elif len(words) > 30:
                score -= 0.2
            
            structure_scores.append(max(0.0, min(1.0, score)))
        
        return statistics.mean(structure_scores)
    
    async def _analyze_domain_accuracy(self, text: str, domain: str, metrics: QualityMetrics):
        """Анализ доменной точности"""
        if domain == "ecommerce":
            domain_accuracy = await self._analyze_ecommerce_accuracy(text)
            metrics.domain_accuracy = domain_accuracy
        
        # Можно добавить другие домены
    
    async def _analyze_ecommerce_accuracy(self, text: str) -> DomainAccuracy:
        """Анализ точности e-commerce терминологии"""
        text_lower = text.lower()
        
        # Поиск e-commerce терминов
        ecommerce_terms = self.ecommerce_config.get_default_ecommerce_terms()
        
        total_found = 0
        correct_terms = 0
        incorrect_terms = 0
        correct_list = []
        incorrect_list = []
        
        for correct_term, wrong_variants in ecommerce_terms.items():
            # Поиск правильного термина
            if correct_term in text_lower:
                total_found += 1
                correct_terms += 1
                correct_list.append(correct_term)
            
            # Поиск неправильных вариантов
            for wrong_variant in wrong_variants:
                if wrong_variant in text_lower:
                    total_found += 1
                    incorrect_terms += 1
                    incorrect_list.append({
                        "found": wrong_variant,
                        "expected": correct_term
                    })
        
        # Расчет точности
        if total_found > 0:
            accuracy_score = correct_terms / total_found
        else:
            accuracy_score = 1.0  # Если терминов нет, то ошибок тоже нет
        
        return DomainAccuracy(
            domain=DomainType.ECOMMERCE,
            total_terms_found=total_found,
            correct_terms=correct_terms,
            incorrect_terms=incorrect_terms,
            missed_terms=0,  # Сложно определить без эталона
            accuracy_score=accuracy_score,
            correct_terms_list=correct_list,
            incorrect_terms_list=incorrect_list
        )
    
    async def _analyze_confidence_metrics(self, word_confidences: List[float], metrics: QualityMetrics):
        """Анализ метрик уверенности"""
        if not word_confidences:
            return
        
        # Средняя уверенность
        metrics.average_word_confidence = statistics.mean(word_confidences)
        
        # Количество слов с низкой уверенностью
        low_confidence_words = [c for c in word_confidences if c < self.confidence_threshold]
        metrics.low_confidence_words_count = len(low_confidence_words)
        
        # Процент слов с низкой уверенностью
        if word_confidences:
            metrics.low_confidence_percentage = len(low_confidence_words) / len(word_confidences)
    
    async def _analyze_content_metrics(self, text: str, metrics: QualityMetrics):
        """Анализ метрик содержания"""
        words = text.split()
        
        metrics.word_count = len(words)
        
        if words:
            # Уникальные слова
            unique_words = set(word.lower() for word in words)
            metrics.unique_words_count = len(unique_words)
            
            # Богатство словаря
            metrics.vocabulary_richness = len(unique_words) / len(words)
    
    async def _analyze_temporal_metrics(self, text: str, audio_duration: float, metrics: QualityMetrics):
        """Анализ временных метрик"""
        words = text.split()
        
        if audio_duration > 0 and words:
            # Скорость речи в словах в минуту
            metrics.speech_rate_wpm = (len(words) / audio_duration) * 60
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Нормализация текста для сравнения WER/CER"""
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление пунктуации
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбиение текста на предложения"""
        # Простое разбиение по знакам препинания
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence_length(self, length: int) -> float:
        """Оценка длины предложения"""
        if 8 <= length <= 20:
            return 1.0
        elif length < 8:
            return length / 8.0
        else:
            return max(0.3, 20.0 / length)
    
    def _score_word_repetitions(self, words: List[str]) -> float:
        """Оценка повторений слов"""
        if len(words) < 2:
            return 1.0
        
        word_counts = Counter(word.lower() for word in words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        
        # Чем меньше повторений, тем лучше
        repetition_ratio = repeated_words / len(word_counts)
        return max(0.0, 1.0 - repetition_ratio)
    
    def _score_word_transitions(self, words: List[str]) -> float:
        """Оценка переходов между словами"""
        # Упрощенная оценка - избегание повторения подряд идущих слов
        if len(words) < 2:
            return 1.0
        
        adjacent_repeats = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                adjacent_repeats += 1
        
        return max(0.0, 1.0 - (adjacent_repeats / len(words)))
    
    def _score_filler_words(self, text: str) -> float:
        """Оценка филлер-слов"""
        text_lower = text.lower()
        
        # Русские филлеры
        russian_fillers = [
            'ээ', 'мм', 'ах', 'эх', 'ну', 'вот', 'это самое', 
            'как бы', 'типа', 'короче', 'блин'
        ]
        
        filler_count = 0
        for filler in russian_fillers:
            filler_count += text_lower.count(filler)
        
        words = text.split()
        if not words:
            return 1.0
        
        filler_ratio = filler_count / len(words)
        return max(0.0, 1.0 - filler_ratio * 2)  # Штраф за филлеры
    
    def _score_vocabulary_diversity(self, text: str) -> float:
        """Оценка разнообразия словаря"""
        words = [word.lower() for word in text.split()]
        if len(words) < 2:
            return 0.5
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        # Нормализация (слишком высокое разнообразие тоже может быть плохо)
        if diversity_ratio > 0.8:
            return 0.8 + (diversity_ratio - 0.8) * 0.5
        return diversity_ratio
    
    def _score_basic_grammar(self, text: str) -> float:
        """Базовая оценка грамматики"""
        # Упрощенная эвристическая оценка
        sentences = self._split_into_sentences(text)
        if not sentences:
            return 0.0
        
        grammar_score = 0.7  # Базовая оценка
        
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            
            # Проверка наличия подлежащего и сказуемого (эвристика)
            if len(words) >= 2:
                grammar_score += 0.1
            
            # Штраф за очень короткие предложения
            if len(words) < 2:
                grammar_score -= 0.2
        
        return max(0.0, min(1.0, grammar_score))
    
    def _score_word_complexity(self, words: List[str]) -> float:
        """Оценка сложности слов"""
        if not words:
            return 0.5
        
        # Простая эвристика - средняя длина слов
        avg_length = statistics.mean(len(word) for word in words)
        
        # Оптимальная длина слов 4-8 символов
        if 4 <= avg_length <= 8:
            return 1.0
        elif avg_length < 4:
            return avg_length / 4.0
        else:
            return max(0.3, 8.0 / avg_length)
    
    def _has_reasonable_comma_usage(self, sentence: str) -> bool:
        """Проверка разумного использования запятых"""
        comma_count = sentence.count(',')
        word_count = len(sentence.split())
        
        if word_count < 5:
            return comma_count == 0
        
        # Примерно одна запятая на 8-10 слов
        expected_commas = word_count // 10
        return abs(comma_count - expected_commas) <= 1
    
    def _has_verb_like_words(self, words: List[str]) -> bool:
        """Эвристическая проверка наличия глаголов"""
        # Простая эвристика для русского языка
        verb_endings = ['ть', 'ет', 'ит', 'ют', 'ят', 'ал', 'ла', 'ло', 'ли']
        
        for word in words:
            word_lower = word.lower()
            for ending in verb_endings:
                if word_lower.endswith(ending):
                    return True
        return False
    
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
        if not self.jiwer_available:
            logger.warning("jiwer not available, using simple WER calculation")
            return self._simple_wer_calculation(hypothesis, reference)
        
        try:
            import jiwer
            
            # Нормализация текстов
            hypothesis_norm = self._normalize_text_for_comparison(hypothesis)
            reference_norm = self._normalize_text_for_comparison(reference)
            
            # Расчет WER
            wer = jiwer.wer(reference_norm, hypothesis_norm)
            return min(1.0, max(0.0, wer))
            
        except Exception as e:
            logger.error(f"WER calculation failed: {e}")
            return self._simple_wer_calculation(hypothesis, reference)
    
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
        if not self.jiwer_available:
            logger.warning("jiwer not available, using simple CER calculation")
            return self._simple_cer_calculation(hypothesis, reference)
        
        try:
            import jiwer
            
            # Нормализация текстов
            hypothesis_norm = self._normalize_text_for_comparison(hypothesis)
            reference_norm = self._normalize_text_for_comparison(reference)
            
            # Расчет CER
            cer = jiwer.cer(reference_norm, hypothesis_norm)
            return min(1.0, max(0.0, cer))
            
        except Exception as e:
            logger.error(f"CER calculation failed: {e}")
            return self._simple_cer_calculation(hypothesis, reference)
    
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
        if not text.strip():
            return 0.0
        
        try:
            # Используем уже реализованный метод расчета беглости
            fluency_score = self._calculate_fluency_score(text)
            
            # Дополнительные факторы для разных языков
            if language == "ru":
                # Для русского языка учитываем морфологию
                if self.morph_analyzer:
                    morphology_bonus = self._assess_russian_morphology(text)
                    fluency_score = (fluency_score + morphology_bonus) / 2.0
            
            return min(1.0, max(0.0, fluency_score))
            
        except Exception as e:
            logger.error(f"Fluency assessment failed: {e}")
            return 0.5  # Средняя оценка при ошибке
    
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
        if not text.strip():
            return 0.0
        
        try:
            if domain.lower() == "ecommerce":
                # Используем уже реализованный метод для e-commerce
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    domain_accuracy = loop.run_until_complete(self._analyze_ecommerce_accuracy(text))
                    return domain_accuracy.accuracy_score
                finally:
                    loop.close()
            
            elif domain.lower() == "general":
                # Общий домен - базовая оценка
                return self._assess_general_domain_accuracy(text)
            
            else:
                # Для неизвестных доменов возвращаем нейтральную оценку
                logger.warning(f"Unknown domain '{domain}', using general assessment")
                return self._assess_general_domain_accuracy(text)
                
        except Exception as e:
            logger.error(f"Domain accuracy assessment failed for domain '{domain}': {e}")
            return 0.5  # Средняя оценка при ошибке
    
    def _simple_wer_calculation(self, hypothesis: str, reference: str) -> float:
        """
        Простой расчет WER без библиотеки jiwer
        Simple WER calculation without jiwer library
        """
        try:
            # Нормализация текстов
            hyp_words = self._normalize_text_for_comparison(hypothesis).split()
            ref_words = self._normalize_text_for_comparison(reference).split()
            
            if not ref_words:
                return 1.0 if hyp_words else 0.0
            
            # Простое сравнение через пересечение множеств
            hyp_set = set(hyp_words)
            ref_set = set(ref_words)
            
            # Приблизительная оценка WER
            common_words = hyp_set & ref_set
            wer = 1.0 - (len(common_words) / len(ref_set))
            
            return min(1.0, max(0.0, wer))
            
        except Exception as e:
            logger.error(f"Simple WER calculation failed: {e}")
            return 1.0
    
    def _simple_cer_calculation(self, hypothesis: str, reference: str) -> float:
        """
        Простой расчет CER без библиотеки jiwer
        Simple CER calculation without jiwer library
        """
        try:
            # Нормализация текстов
            hyp_chars = set(self._normalize_text_for_comparison(hypothesis).replace(' ', ''))
            ref_chars = set(self._normalize_text_for_comparison(reference).replace(' ', ''))
            
            if not ref_chars:
                return 1.0 if hyp_chars else 0.0
            
            # Приблизительная оценка CER
            common_chars = hyp_chars & ref_chars
            cer = 1.0 - (len(common_chars) / len(ref_chars))
            
            return min(1.0, max(0.0, cer))
            
        except Exception as e:
            logger.error(f"Simple CER calculation failed: {e}")
            return 1.0
    
    def _assess_russian_morphology(self, text: str) -> float:
        """
        Оценка русской морфологии с pymorphy2
        Assess Russian morphology with pymorphy2
        """
        if not self.morph_analyzer:
            return 0.5
        
        try:
            words = text.split()
            if not words:
                return 0.0
            
            morphology_scores = []
            
            for word in words[:10]:  # Ограничиваем для производительности
                # Очистка слова от пунктуации
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if not clean_word:
                    continue
                
                # Анализ морфологии
                parsed = self.morph_analyzer.parse(clean_word)
                if parsed:
                    # Проверяем уверенность анализа
                    best_parse = parsed[0]
                    score = best_parse.score if hasattr(best_parse, 'score') else 0.7
                    morphology_scores.append(score)
            
            return statistics.mean(morphology_scores) if morphology_scores else 0.5
            
        except Exception as e:
            logger.error(f"Russian morphology assessment failed: {e}")
            return 0.5
    
    def _assess_general_domain_accuracy(self, text: str) -> float:
        """
        Базовая оценка для общего домена
        Basic assessment for general domain
        """
        # Простая эвристическая оценка
        factors = []
        
        # Длина текста
        words = text.split()
        if words:
            length_factor = min(1.0, len(words) / 20.0)  # Оптимально 20+ слов
            factors.append(length_factor)
        
        # Разнообразие словаря
        if words:
            unique_words = set(word.lower() for word in words)
            diversity = len(unique_words) / len(words)
            factors.append(min(1.0, diversity * 2.0))
        
        # Отсутствие явных ошибок (базовая проверка)
        error_indicators = ['ээ', 'мм', 'эээ', 'ммм', 'тест', 'проверка']
        text_lower = text.lower()
        error_count = sum(1 for indicator in error_indicators if indicator in text_lower)
        error_factor = max(0.0, 1.0 - error_count * 0.2)
        factors.append(error_factor)
        
        return statistics.mean(factors) if factors else 0.5
    
    def _create_empty_quality_metrics(self, error_message: str) -> QualityMetrics:
        """Создание пустых метрик при ошибке"""
        metrics = QualityMetrics()
        metrics.overall_score = 0.0
        metrics.quality_level = QualityLevel.VERY_POOR
        metrics.evaluation_method = "failed"
        metrics.improvement_suggestions = [error_message]
        return metrics