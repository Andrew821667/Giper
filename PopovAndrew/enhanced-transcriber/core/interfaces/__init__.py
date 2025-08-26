# Core interfaces for Enhanced Transcriber
# Абстрактные интерфейсы для всех компонентов системы

from .transcriber import ITranscriber, ITranscriberProvider
from .audio_processor import IAudioProcessor, IAudioAnalyzer
from .quality_assessor import IQualityAssessor, IQualityMetrics

__all__ = [
    'ITranscriber',
    'ITranscriberProvider', 
    'IAudioProcessor',
    'IAudioAnalyzer',
    'IQualityAssessor',
    'IQualityMetrics'
]