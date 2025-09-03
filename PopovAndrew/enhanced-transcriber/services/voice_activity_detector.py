"""
Модуль Voice Activity Detection (VAD) для обнаружения речевой активности
Подавление тишины и сегментация аудио на речевые участки
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import tempfile
import os

import numpy as np
import librosa
import soundfile as sf
from scipy import signal

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """
    Детектор речевой активности для предобработки аудио
    Удаляет тишину и выделяет сегменты с речью
    """
    
    def __init__(self, 
                 frame_duration: float = 0.025,  # 25ms фреймы
                 hop_duration: float = 0.010,    # 10ms шаг
                 energy_threshold: float = 0.6,  # порог энергии
                 frequency_threshold: float = 0.4,  # порог частотного содержимого
                 silence_duration: float = 0.3):   # минимальная длительность тишины
        
        self.frame_duration = frame_duration
        self.hop_duration = hop_duration
        self.energy_threshold = energy_threshold
        self.frequency_threshold = frequency_threshold
        self.silence_duration = silence_duration
        
        # Параметры для более точного детектирования
        self.min_speech_duration = 0.1  # минимальная длительность речи
        self.speech_pad_duration = 0.1  # отступы вокруг речи
        
        logger.info("VAD инициализирован")
    
    async def detect_voice_activity(self, audio_file: str, 
                                  method: str = "energy") -> Dict[str, Any]:
        """
        Основная функция детектирования речевой активности
        
        Args:
            audio_file: путь к аудиофайлу
            method: метод детектирования ("energy", "spectral", "combined", "webrtcvad")
        
        Returns:
            Dict с информацией о речевых сегментах
        """
        start_time = time.time()
        
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_file, sr=16000)  # стандартизация на 16kHz
            duration = len(y) / sr
            
            logger.info(f"Обработка файла длительностью {duration:.1f}s методом {method}")
            
            if method == "energy":
                speech_segments = await self._energy_based_vad(y, sr)
            elif method == "spectral":
                speech_segments = await self._spectral_based_vad(y, sr)
            elif method == "combined":
                speech_segments = await self._combined_vad(y, sr)
            elif method == "webrtcvad":
                speech_segments = await self._webrtc_vad(audio_file, y, sr)
            else:
                raise ValueError(f"Неподдерживаемый метод VAD: {method}")
            
            # Постобработка сегментов
            speech_segments = self._refine_segments(speech_segments, duration)
            
            # Статистика
            total_speech_time = sum(seg["end"] - seg["start"] for seg in speech_segments)
            silence_time = duration - total_speech_time
            speech_ratio = total_speech_time / duration if duration > 0 else 0
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "method": method,
                "processing_time": processing_time,
                "total_duration": duration,
                "speech_segments": speech_segments,
                "total_speech_time": total_speech_time,
                "silence_time": silence_time,
                "speech_ratio": speech_ratio,
                "segments_count": len(speech_segments)
            }
            
            logger.info(f"VAD завершен: {len(speech_segments)} сегментов, {speech_ratio:.1%} речи")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка VAD: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _energy_based_vad(self, y: np.ndarray, sr: int) -> List[Dict]:
        """VAD на основе энергетического анализа"""
        
        # Параметры фреймирования
        frame_length = int(self.frame_duration * sr)
        hop_length = int(self.hop_duration * sr)
        
        # Вычисление энергии по фреймам
        frames = librosa.util.frame(y, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        energy = np.array([np.sum(frame**2) for frame in frames])
        
        # Нормализация энергии
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        # Адаптивный порог на основе статистики энергии
        adaptive_threshold = np.percentile(energy, 30) + self.energy_threshold * (
            np.percentile(energy, 70) - np.percentile(energy, 30)
        )
        
        # Определение активных фреймов
        voice_frames = energy > adaptive_threshold
        
        # Сглаживание (медианный фильтр)
        voice_frames = signal.medfilt(voice_frames.astype(float), kernel_size=5).astype(bool)
        
        # Преобразование в временные сегменты
        segments = self._frames_to_segments(voice_frames, hop_length, sr)
        
        return segments
    
    async def _spectral_based_vad(self, y: np.ndarray, sr: int) -> List[Dict]:
        """VAD на основе спектрального анализа"""
        
        # STFT для спектрального анализа
        stft = librosa.stft(y, hop_length=int(self.hop_duration * sr))
        magnitude = np.abs(stft)
        
        # Спектральные признаки
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, 
            hop_length=int(self.hop_duration * sr))[0]
        
        # Нормализация признаков
        features = np.array([spectral_centroids, spectral_rolloff, zero_crossing_rate])
        features = (features - np.mean(features, axis=1, keepdims=True)) / (
            np.std(features, axis=1, keepdims=True) + 1e-8)
        
        # Комбинированный спектральный счет
        spectral_score = np.mean(features, axis=0)
        
        # Определение речевых участков
        adaptive_threshold = np.percentile(spectral_score, 40) + self.frequency_threshold * (
            np.percentile(spectral_score, 80) - np.percentile(spectral_score, 40)
        )
        
        voice_frames = spectral_score > adaptive_threshold
        
        # Сглаживание
        voice_frames = signal.medfilt(voice_frames.astype(float), kernel_size=7).astype(bool)
        
        # Преобразование в сегменты
        hop_length = int(self.hop_duration * sr)
        segments = self._frames_to_segments(voice_frames, hop_length, sr)
        
        return segments
    
    async def _combined_vad(self, y: np.ndarray, sr: int) -> List[Dict]:
        """Комбинированный VAD (энергия + спектральный анализ)"""
        
        # Получение результатов от обоих методов
        energy_segments = await self._energy_based_vad(y, sr)
        spectral_segments = await self._spectral_based_vad(y, sr)
        
        # Объединение сегментов
        all_segments = energy_segments + spectral_segments
        
        # Слияние пересекающихся сегментов
        if all_segments:
            merged_segments = self._merge_overlapping_segments(all_segments)
            return merged_segments
        
        return []
    
    async def _webrtc_vad(self, audio_file: str, y: np.ndarray, sr: int) -> List[Dict]:
        """VAD с использованием WebRTC (если доступен)"""
        try:
            import webrtcvad
            import wave
            
            # Создание VAD детектора
            vad = webrtcvad.Vad(2)  # агрессивность 0-3
            
            # WebRTC требует определенные форматы
            if sr != 16000:
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                y_resampled = y
            
            # Преобразование в 16-bit PCM
            y_int16 = (y_resampled * 32767).astype(np.int16)
            
            # Анализ фреймов по 10ms
            frame_duration = 10  # ms
            frame_length = int(sr * frame_duration / 1000)
            
            voice_frames = []
            for i in range(0, len(y_int16) - frame_length, frame_length):
                frame = y_int16[i:i + frame_length]
                frame_bytes = frame.tobytes()
                
                is_speech = vad.is_speech(frame_bytes, sr)
                voice_frames.append(is_speech)
            
            # Преобразование в сегменты
            segments = self._frames_to_segments(np.array(voice_frames), frame_length, sr)
            
            return segments
            
        except ImportError:
            logger.warning("webrtcvad не установлен, используется энергетический VAD")
            return await self._energy_based_vad(y, sr)
        except Exception as e:
            logger.warning(f"Ошибка WebRTC VAD: {e}, используется резервный метод")
            return await self._energy_based_vad(y, sr)
    
    def _frames_to_segments(self, voice_frames: np.ndarray, 
                          hop_length: int, sr: int) -> List[Dict]:
        """Преобразование булевого массива фреймов в временные сегменты"""
        
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_frames):
            current_time = i * hop_length / sr
            
            if is_voice and not in_speech:
                # Начало речевого сегмента
                start_time = current_time
                in_speech = True
            elif not is_voice and in_speech:
                # Конец речевого сегмента
                end_time = current_time
                if end_time - start_time >= self.min_speech_duration:
                    segments.append({
                        "start": max(0, start_time - self.speech_pad_duration),
                        "end": end_time + self.speech_pad_duration,
                        "duration": end_time - start_time,
                        "type": "speech"
                    })
                in_speech = False
        
        # Закрытие последнего сегмента если необходимо
        if in_speech:
            end_time = len(voice_frames) * hop_length / sr
            if end_time - start_time >= self.min_speech_duration:
                segments.append({
                    "start": max(0, start_time - self.speech_pad_duration),
                    "end": end_time + self.speech_pad_duration,
                    "duration": end_time - start_time,
                    "type": "speech"
                })
        
        return segments
    
    def _refine_segments(self, segments: List[Dict], total_duration: float) -> List[Dict]:
        """Постобработка и уточнение сегментов"""
        
        # Ограничение сегментов границами аудио
        refined_segments = []
        for segment in segments:
            segment["start"] = max(0, segment["start"])
            segment["end"] = min(total_duration, segment["end"])
            segment["duration"] = segment["end"] - segment["start"]
            
            if segment["duration"] > 0:
                refined_segments.append(segment)
        
        # Слияние близких сегментов
        if len(refined_segments) > 1:
            merged_segments = []
            current_segment = refined_segments[0]
            
            for next_segment in refined_segments[1:]:
                gap = next_segment["start"] - current_segment["end"]
                
                if gap <= self.silence_duration:
                    # Слияние сегментов
                    current_segment["end"] = next_segment["end"]
                    current_segment["duration"] = current_segment["end"] - current_segment["start"]
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            
            merged_segments.append(current_segment)
            refined_segments = merged_segments
        
        return refined_segments
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Слияние пересекающихся сегментов"""
        if not segments:
            return []
        
        # Сортировка по времени начала
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        merged = []
        current = sorted_segments[0]
        
        for segment in sorted_segments[1:]:
            if segment["start"] <= current["end"]:
                # Пересечение - слияние
                current["end"] = max(current["end"], segment["end"])
                current["duration"] = current["end"] - current["start"]
            else:
                merged.append(current)
                current = segment
        
        merged.append(current)
        return merged
    
    async def remove_silence(self, audio_file: str, 
                           output_file: str = None,
                           method: str = "energy") -> Dict[str, Any]:
        """
        Удаление тишины из аудиофайла
        
        Returns:
            Dict с информацией о результате обработки
        """
        try:
            # Детектирование речевых сегментов
            vad_result = await self.detect_voice_activity(audio_file, method)
            
            if not vad_result["success"]:
                return vad_result
            
            # Загрузка исходного аудио
            y, sr = librosa.load(audio_file, sr=None)
            
            # Извлечение речевых сегментов
            speech_segments = vad_result["speech_segments"]
            
            if not speech_segments:
                logger.warning("Не найдено речевых сегментов")
                return {
                    "success": False,
                    "error": "Не найдено речевых сегментов"
                }
            
            # Объединение речевых участков
            processed_audio = []
            for segment in speech_segments:
                start_sample = int(segment["start"] * sr)
                end_sample = int(segment["end"] * sr)
                processed_audio.append(y[start_sample:end_sample])
            
            final_audio = np.concatenate(processed_audio)
            
            # Определение выходного файла
            if output_file is None:
                input_path = Path(audio_file)
                output_file = str(input_path.parent / f"{input_path.stem}_no_silence{input_path.suffix}")
            
            # Сохранение обработанного аудио
            sf.write(output_file, final_audio, sr)
            
            # Статистика
            original_duration = len(y) / sr
            processed_duration = len(final_audio) / sr
            time_saved = original_duration - processed_duration
            compression_ratio = processed_duration / original_duration
            
            result = {
                "success": True,
                "original_file": audio_file,
                "processed_file": output_file,
                "original_duration": original_duration,
                "processed_duration": processed_duration,
                "time_saved": time_saved,
                "compression_ratio": compression_ratio,
                "segments_kept": len(speech_segments),
                "vad_info": vad_result
            }
            
            logger.info(f"Тишина удалена: {time_saved:.1f}s сэкономлено ({compression_ratio:.1%} от исходного)")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка удаления тишины: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_voice_statistics(self, vad_result: Dict) -> Dict[str, Any]:
        """Получение детальной статистики по речевой активности"""
        if not vad_result.get("success"):
            return {"error": "VAD не выполнен успешно"}
        
        segments = vad_result["speech_segments"]
        
        if not segments:
            return {"error": "Нет речевых сегментов"}
        
        durations = [seg["duration"] for seg in segments]
        gaps = []
        
        # Вычисление пауз между сегментами
        for i in range(len(segments) - 1):
            gap = segments[i+1]["start"] - segments[i]["end"]
            gaps.append(gap)
        
        stats = {
            "total_segments": len(segments),
            "speech_time": vad_result["total_speech_time"],
            "silence_time": vad_result["silence_time"],
            "speech_ratio": vad_result["speech_ratio"],
            "average_segment_duration": np.mean(durations),
            "longest_segment": np.max(durations),
            "shortest_segment": np.min(durations),
            "average_pause": np.mean(gaps) if gaps else 0,
            "longest_pause": np.max(gaps) if gaps else 0,
            "speaking_rate": len(segments) / vad_result["total_duration"] * 60,  # сегментов в минуту
        }
        
        return stats


# Функция для быстрого использования VAD
async def quick_vad(audio_file: str, method: str = "combined") -> Dict[str, Any]:
    """Быстрое использование VAD с дефолтными настройками"""
    vad = VoiceActivityDetector()
    return await vad.detect_voice_activity(audio_file, method)


# Функция для быстрого удаления тишины
async def remove_silence_quick(audio_file: str, output_file: str = None) -> str:
    """Быстрое удаление тишины из аудиофайла"""
    vad = VoiceActivityDetector()
    result = await vad.remove_silence(audio_file, output_file)
    
    if result["success"]:
        return result["processed_file"]
    else:
        raise Exception(result["error"])
