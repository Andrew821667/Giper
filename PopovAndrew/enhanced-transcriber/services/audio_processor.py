"""
Обработчик аудио для улучшения качества перед транскрипцией
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import shutil

import numpy as np
import librosa
import soundfile as sf

from core.interfaces.audio_processor import IAudioProcessor
from core.models.audio_metadata import AudioMetadata, AudioFormat, AudioQuality

logger = logging.getLogger(__name__)


class AudioProcessorService(IAudioProcessor):
    """Полная реализация аудиопроцессора"""

    def __init__(self, enable_noise_reduction=True, enable_volume_normalization=True, enable_speech_enhancement=True, target_sample_rate=16000):
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_volume_normalization = enable_volume_normalization
        self.enable_speech_enhancement = enable_speech_enhancement
        self.target_sample_rate = target_sample_rate
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        logger.info("AudioProcessorService инициализирован")

    def is_supported_format(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.supported_formats

    async def process(self, audio_file: str, **kwargs) -> Dict[str, Any]:
        """Основная обработка аудио"""
        start_time = time.time()
        temp_output = None
        
        try:
            # Загрузка
            y, sr = librosa.load(audio_file, sr=None, mono=True)
            
            # Ресемплинг
            if sr != self.target_sample_rate:
                y = await self.resample_audio(y, sr, self.target_sample_rate)
                sr = self.target_sample_rate
            
            # Нормализация
            if self.enable_volume_normalization:
                y = await self.normalize_volume(y)
            
            # Шумоподавление
            if self.enable_noise_reduction:
                y = await self.reduce_noise(y, sr)
            
            # Улучшение
            if self.enable_speech_enhancement:
                y = await self.enhance_audio(y, sr)
            
            # Сохранение
            temp_output = tempfile.mktemp(suffix=".wav")
            sf.write(temp_output, y, sr)
            
            metadata = AudioMetadata(
                format=AudioFormat(codec="pcm_s16le", sample_rate=sr, channels=1, bit_depth=16),
                quality=AudioQuality.HIGH,
                duration=len(y) / sr,
                original_size=os.path.getsize(audio_file),
                processed_size=os.path.getsize(temp_output)
            )
            
            return {
                "success": True,
                "processed_audio": temp_output,
                "metadata": metadata,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            if temp_output and os.path.exists(temp_output):
                os.unlink(temp_output)
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}

    # РЕАЛИЗАЦИЯ ВСЕХ АБСТРАКТНЫХ МЕТОДОВ

    async def enhance_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Улучшение аудио"""
        try:
            # Предварительное усиление
            enhanced = librosa.effects.preemphasis(audio_data)
            # Сглаживание
            enhanced = np.tanh(enhanced * 2) / 2
            return enhanced
        except Exception as e:
            logger.warning(f"Улучшение не удалось: {e}")
            return audio_data

    async def normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """Нормализация громкости"""
        try:
            return librosa.util.normalize(audio_data)
        except Exception as e:
            logger.warning(f"Нормализация не удалась: {e}")
            return audio_data

    async def reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Шумоподавление"""
        try:
            # Базовое спектральное шумоподавление
            S = librosa.stft(audio_data)
            magnitude = np.abs(S)
            phase = np.angle(S)
            
            # Оценка шума
            noise_frames = min(10, magnitude.shape[1] // 4)
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Вычитание шума
            clean_magnitude = magnitude - 1.5 * noise_profile
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Восстановление
            clean_S = clean_magnitude * np.exp(1j * phase)
            return librosa.istft(clean_S)
            
        except Exception as e:
            logger.warning(f"Шумоподавление не удалось: {e}")
            return audio_data

    async def resample_audio(self, audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Ресемплинг"""
        try:
            return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
        except Exception as e:
            logger.warning(f"Ресемплинг не удался: {e}")
            return audio_data

    async def cleanup(self, temp_file: str):
        """Очистка"""
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    def get_supported_formats(self) -> set:
        return self.supported_formats.copy()
