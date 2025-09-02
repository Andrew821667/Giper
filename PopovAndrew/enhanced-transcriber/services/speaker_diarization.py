"""
Модуль Speaker Diarization для разделения спикеров и назначения ролей K:/M:
Поддерживает pyannote-audio для точного разделения спикеров в звонках
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import tempfile
import os

import numpy as np
import librosa
import torch

logger = logging.getLogger(__name__)

class SpeakerDiarization:
    """
    Сервис для разделения спикеров и назначения ролей в телефонных разговорах
    """
    
    def __init__(self, use_pyannote: bool = True, hf_token: Optional[str] = None):
        self.use_pyannote = use_pyannote
        self.hf_token = hf_token
        self.pipeline = None
        self.fallback_mode = False
        
        # Настройки для звонков
        self.min_speaker_duration = 1.0  # минимальная длительность речи спикера
        self.min_silence_duration = 0.5  # минимальная пауза между спикерами
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Инициализация модели диаризации"""
        if not self.use_pyannote:
            logger.info("Pyannote отключен, используется fallback режим")
            self.fallback_mode = True
            return
            
        try:
            # Попытка загрузки pyannote
            from pyannote.audio import Pipeline
            
            logger.info("Загрузка pyannote speaker diarization pipeline...")
            
            # Загрузка предобученной модели
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # Настройка для GPU если доступно
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("Pyannote pipeline загружен на GPU")
            else:
                logger.info("Pyannote pipeline загружен на CPU")
                
        except ImportError:
            logger.warning("pyannote-audio не установлен, используется fallback режим")
            self.fallback_mode = True
        except Exception as e:
            logger.error(f"Ошибка загрузки pyannote pipeline: {e}")
            logger.info("Переключение на fallback режим")
            self.fallback_mode = True
    
    async def diarize_audio(self, audio_file: str, max_speakers: int = 2) -> Dict[str, Any]:
        """
        Основная функция диаризации спикеров
        
        Returns:
            Dict с информацией о спикерах и временных сегментах
        """
        start_time = time.time()
        
        try:
            if self.fallback_mode:
                return await self._fallback_diarization(audio_file, max_speakers)
            else:
                return await self._pyannote_diarization(audio_file, max_speakers)
                
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
            # Fallback к простому разделению
            return await self._fallback_diarization(audio_file, max_speakers)
    
    async def _pyannote_diarization(self, audio_file: str, max_speakers: int) -> Dict[str, Any]:
        """Диаризация с использованием pyannote-audio"""
        logger.info("Запуск pyannote диаризации...")
        
        start_time = time.time()
        
        # Загрузка аудио
        waveform, sample_rate = librosa.load(audio_file, sr=16000)
        duration = len(waveform) / sample_rate
        
        # Создание временного файла для pyannote
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, waveform, sample_rate)
            
            # Запуск диаризации
            diarization = self.pipeline(tmp_file.name, num_speakers=max_speakers)
            
            # Очистка временного файла
            os.unlink(tmp_file.name)
        
        # Обработка результатов
        speaker_segments = []
        speaker_stats = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_info = {
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start,
                "speaker": speaker
            }
            speaker_segments.append(segment_info)
            
            # Статистика по спикерам
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"total_duration": 0, "segments": 0}
            
            speaker_stats[speaker]["total_duration"] += segment_info["duration"]
            speaker_stats[speaker]["segments"] += 1
        
        # Назначение ролей на основе порядка речи и длительности
        role_mapping = self._assign_roles_smart(speaker_stats, speaker_segments)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "method": "pyannote",
            "processing_time": processing_time,
            "total_duration": duration,
            "speakers_detected": len(speaker_stats),
            "speaker_segments": speaker_segments,
            "speaker_stats": speaker_stats,
            "role_mapping": role_mapping,
            "segments_with_roles": self._apply_role_mapping(speaker_segments, role_mapping)
        }
    
    async def _fallback_diarization(self, audio_file: str, max_speakers: int) -> Dict[str, Any]:
        """Простая диаризация на основе энергетического анализа"""
        logger.info("Запуск fallback диаризации...")
        
        start_time = time.time()
        
        # Загрузка аудио
        y, sr = librosa.load(audio_file, sr=16000)
        duration = len(y) / sr
        
        # Простое разделение на основе пауз и энергии
        frame_length = int(0.5 * sr)  # 0.5 секунды
        hop_length = int(0.1 * sr)    # 0.1 секунды
        
        # Вычисление энергии по кадрам
        energy = []
        timestamps = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
            timestamps.append(i / sr)
        
        energy = np.array(energy)
        
        # Определение активных сегментов речи
        energy_threshold = np.percentile(energy, 30)  # 30-й процентиль как порог
        active_frames = energy > energy_threshold
        
        # Группировка в сегменты
        segments = []
        current_segment = None
        
        for i, (active, timestamp) in enumerate(zip(active_frames, timestamps)):
            if active and current_segment is None:
                # Начало нового сегмента
                current_segment = {"start": timestamp, "speaker": "unknown"}
            elif not active and current_segment is not None:
                # Конец сегмента
                current_segment["end"] = timestamp
                current_segment["duration"] = current_segment["end"] - current_segment["start"]
                
                if current_segment["duration"] >= self.min_speaker_duration:
                    segments.append(current_segment)
                
                current_segment = None
        
        # Закрытие последнего сегмента если необходимо
        if current_segment is not None:
            current_segment["end"] = timestamps[-1]
            current_segment["duration"] = current_segment["end"] - current_segment["start"]
            if current_segment["duration"] >= self.min_speaker_duration:
                segments.append(current_segment)
        
        # Простое назначение спикеров (чередование)
        for i, segment in enumerate(segments):
            segment["speaker"] = f"SPEAKER_{i % max_speakers}"
        
        # Создание статистики
        speaker_stats = {}
        for segment in segments:
            speaker = segment["speaker"]
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"total_duration": 0, "segments": 0}
            
            speaker_stats[speaker]["total_duration"] += segment["duration"]
            speaker_stats[speaker]["segments"] += 1
        
        # Назначение ролей
        role_mapping = self._assign_roles_simple(segments)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "method": "fallback",
            "processing_time": processing_time,
            "total_duration": duration,
            "speakers_detected": len(speaker_stats),
            "speaker_segments": segments,
            "speaker_stats": speaker_stats,
            "role_mapping": role_mapping,
            "segments_with_roles": self._apply_role_mapping(segments, role_mapping)
        }
    
    def _assign_roles_smart(self, speaker_stats: Dict, segments: List) -> Dict[str, str]:
        """Умное назначение ролей на основе анализа разговора"""
        if len(speaker_stats) < 2:
            # Один спикер - назначаем как клиента
            speakers = list(speaker_stats.keys())
            return {speakers[0]: "K"} if speakers else {}
        
        # Сортировка спикеров по времени первого появления
        first_appearance = {}
        for segment in segments:
            speaker = segment["speaker"]
            if speaker not in first_appearance:
                first_appearance[speaker] = segment["start"]
        
        sorted_speakers = sorted(first_appearance.keys(), key=lambda x: first_appearance[x])
        
        # Логика назначения:
        # 1. Кто говорит первым - обычно менеджер (отвечает на звонок)
        # 2. Кто говорит больше всего - обычно менеджер
        # Но учитываем что может быть наоборот
        
        role_mapping = {}
        
        if len(sorted_speakers) >= 2:
            # Анализ длительности речи
            speaker_durations = [(sp, speaker_stats[sp]["total_duration"]) 
                               for sp in sorted_speakers]
            speaker_durations.sort(key=lambda x: x[1], reverse=True)
            
            # Первый по времени появления = M (менеджер отвечает)
            # Но если второй говорит значительно больше, то он может быть M
            first_speaker = sorted_speakers[0]
            second_speaker = sorted_speakers[1]
            
            first_duration = speaker_stats[first_speaker]["total_duration"]
            second_duration = speaker_stats[second_speaker]["total_duration"]
            
            if second_duration > first_duration * 1.5:
                # Второй говорит намного больше - вероятно он менеджер
                role_mapping[second_speaker] = "M"
                role_mapping[first_speaker] = "K"
            else:
                # Стандартное назначение
                role_mapping[first_speaker] = "M"  
                role_mapping[second_speaker] = "K"
            
            # Остальных спикеров назначаем поочередно
            remaining_speakers = sorted_speakers[2:]
            for i, speaker in enumerate(remaining_speakers):
                role_mapping[speaker] = "K" if i % 2 == 0 else "M"
        else:
            # Один спикер
            role_mapping[sorted_speakers[0]] = "K"
        
        return role_mapping
    
    def _assign_roles_simple(self, segments: List) -> Dict[str, str]:
        """Простое назначение ролей по порядку появления"""
        speakers = []
        for segment in segments:
            if segment["speaker"] not in speakers:
                speakers.append(segment["speaker"])
        
        role_mapping = {}
        for i, speaker in enumerate(speakers):
            # Первый спикер = менеджер, второй = клиент
            role_mapping[speaker] = "M" if i == 0 else "K"
        
        return role_mapping
    
    def _apply_role_mapping(self, segments: List, role_mapping: Dict) -> List[Dict]:
        """Применение назначенных ролей к сегментам"""
        segments_with_roles = []
        
        for segment in segments:
            segment_copy = segment.copy()
            segment_copy["role"] = role_mapping.get(segment["speaker"], "U")  # U = Unknown
            segments_with_roles.append(segment_copy)
        
        return segments_with_roles
    
    def format_transcript_with_roles(self, transcript_text: str, 
                                   diarization_result: Dict, 
                                   segment_threshold: float = 5.0) -> str:
        """
        Форматирование транскрипта с ролями на основе диаризации
        
        Args:
            transcript_text: Исходный текст транскрипции
            diarization_result: Результат диаризации
            segment_threshold: Минимальная длительность сегмента для разделения
        """
        if not diarization_result.get("success"):
            # Fallback к простому форматированию
            return self._simple_role_formatting(transcript_text)
        
        segments_with_roles = diarization_result["segments_with_roles"]
        
        # Разбиение текста на части согласно временным сегментам
        # Это упрощенная версия - в реальности нужна синхронизация с временными метками
        sentences = transcript_text.split('.')
        formatted_lines = []
        
        segment_idx = 0
        current_role = segments_with_roles[0]["role"] if segments_with_roles else "K"
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Смена роли каждые несколько предложений (упрощенная логика)
                if i > 0 and i % 3 == 0 and segment_idx < len(segments_with_roles) - 1:
                    segment_idx += 1
                    current_role = segments_with_roles[segment_idx]["role"]
                
                timestamp = self._estimate_timestamp(i, len(sentences), 
                                                   diarization_result.get("total_duration", 0))
                
                formatted_lines.append(
                    f"{current_role}: [{self._format_time(timestamp)}] {sentence.strip()}"
                )
        
        return "\n".join(formatted_lines)
    
    def _simple_role_formatting(self, text: str) -> str:
        """Простое форматирование с чередованием ролей"""
        sentences = text.split('.')
        formatted = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                role = "M" if i % 2 == 0 else "K"
                formatted.append(f"{role}: {sentence.strip()}")
        
        return "\n".join(formatted)
    
    def _estimate_timestamp(self, sentence_idx: int, total_sentences: int, 
                          total_duration: float) -> float:
        """Оценка временной метки для предложения"""
        if total_sentences == 0:
            return 0.0
        return (sentence_idx / total_sentences) * total_duration
    
    def _format_time(self, seconds: float) -> str:
        """Форматирование времени в MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def get_speaker_statistics(self, diarization_result: Dict) -> Dict[str, Any]:
        """Получение статистики по спикерам"""
        if not diarization_result.get("success"):
            return {"error": "Диаризация не выполнена"}
        
        stats = diarization_result["speaker_stats"]
        role_mapping = diarization_result["role_mapping"]
        
        formatted_stats = {}
        for speaker, data in stats.items():
            role = role_mapping.get(speaker, "Unknown")
            role_name = {"M": "Менеджер", "K": "Клиент", "Unknown": "Неизвестен"}[role]
            
            formatted_stats[f"{role_name} ({speaker})"] = {
                "total_speech_time": f"{data['total_duration']:.1f}s",
                "number_of_turns": data["segments"],
                "average_turn_length": f"{data['total_duration'] / max(data['segments'], 1):.1f}s"
            }
        
        return formatted_stats
