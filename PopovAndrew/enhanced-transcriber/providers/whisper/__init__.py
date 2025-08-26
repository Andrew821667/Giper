# Whisper providers for local and OpenAI transcription
# Провайдеры Whisper для локальной и OpenAI транскрипции

from .whisper_local import WhisperLocalTranscriber
from .whisper_openai import WhisperOpenAITranscriber
from .whisper_provider import WhisperProvider

__all__ = [
    'WhisperLocalTranscriber', 
    'WhisperOpenAITranscriber', 
    'WhisperProvider'
]