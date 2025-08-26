"""
Обработчик аудио для улучшения качества перед транскрипцией
Audio processor service for quality enhancement before transcription
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import shutil

from core.interfaces.audio_processor import IAudioProcessor
from core.models.audio_metadata import AudioMetadata, AudioFormat, AudioQuality

logger = logging.getLogger(__name__)


class AudioProcessorService(IAudioProcessor):
    """
    Сервис обработки аудио для максимального качества транскрипции
    Audio processing service for maximum transcription quality
    """
    
    def __init__(
        self,
        enable_noise_reduction: bool = True,
        enable_volume_normalization: bool = True,
        enable_speech_enhancement: bool = True,
        target_sample_rate: int = 16000,
        convert_to_mono: bool = True,
        temp_dir: Optional[str] = None
    ):
        """
        Инициализация аудио процессора
        
        Args:
            enable_noise_reduction: Включить шумоподавление
            enable_volume_normalization: Включить нормализацию громкости
            enable_speech_enhancement: Включить улучшение речи
            target_sample_rate: Целевая частота дискретизации
            convert_to_mono: Конвертировать в моно
            temp_dir: Директория для временных файлов
        """
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_volume_normalization = enable_volume_normalization
        self.enable_speech_enhancement = enable_speech_enhancement
        self.target_sample_rate = target_sample_rate
        self.convert_to_mono = convert_to_mono
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        
        # Проверка доступности библиотек
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Проверка доступности необходимых библиотек"""
        self.librosa_available = False
        self.pydub_available = False
        self.noisereduce_available = False
        
        try:
            import librosa
            self.librosa_available = True
            logger.info("✅ librosa available for audio processing")
        except ImportError:
            logger.warning("⚠️ librosa not available - limited audio processing")
        
        try:
            from pydub import AudioSegment
            self.pydub_available = True
            logger.info("✅ pydub available for audio format conversion")
        except ImportError:
            logger.warning("⚠️ pydub not available - limited format support")
        
        try:
            import noisereduce as nr
            self.noisereduce_available = True
            logger.info("✅ noisereduce available for noise reduction")
        except ImportError:
            logger.warning("⚠️ noisereduce not available - basic noise reduction only")
    
    async def enhance_audio(self, audio_file: str, **kwargs) -> str:
        """
        Улучшение аудио файла для транскрипции
        
        Args:
            audio_file: Путь к исходному аудио файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            str: Путь к обработанному аудио файлу
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        logger.info(f"🎵 Starting audio enhancement: {Path(audio_file).name}")
        
        # Анализ исходного аудио
        metadata = await self.analyze_audio(audio_file)
        logger.info(f"📊 Original audio: {metadata.duration:.1f}s, {metadata.sample_rate}Hz, {metadata.format.value}")
        
        try:
            # Создание временного файла для результата
            enhanced_file = self._create_temp_filename(audio_file, "enhanced")
            
            # Последовательная обработка
            current_file = audio_file
            
            # 1. Конвертация формата и параметров
            if self._needs_format_conversion(metadata):
                logger.info("🔄 Converting audio format...")
                converted_file = await self._convert_audio_format(current_file, metadata)
                current_file = converted_file
            
            # 2. Нормализация громкости
            if self.enable_volume_normalization:
                logger.info("🔊 Normalizing volume...")
                normalized_file = await self._normalize_volume(current_file)
                current_file = normalized_file
            
            # 3. Шумоподавление
            if self.enable_noise_reduction and self._needs_noise_reduction(metadata):
                logger.info("🔇 Reducing noise...")
                denoised_file = await self._reduce_noise(current_file)
                current_file = denoised_file
            
            # 4. Улучшение речи
            if self.enable_speech_enhancement:
                logger.info("🗣️ Enhancing speech...")
                enhanced_speech_file = await self._enhance_speech(current_file)
                current_file = enhanced_speech_file
            
            # Финальный файл
            if current_file != audio_file:
                shutil.copy2(current_file, enhanced_file)
            else:
                # Если обработка не нужна, копируем оригинал
                shutil.copy2(audio_file, enhanced_file)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Audio enhancement completed in {processing_time:.1f}s: {enhanced_file}")
            
            return enhanced_file
            
        except Exception as e:
            logger.error(f"❌ Audio enhancement failed: {e}")
            # Возвращаем оригинальный файл при ошибке
            return audio_file
    
    async def analyze_audio(self, audio_file: str, **kwargs) -> AudioMetadata:
        """
        Анализ аудио файла
        
        Args:
            audio_file: Путь к аудио файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            AudioMetadata: Метаданные аудио
        """
        file_path = Path(audio_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Базовые метаданные файла
            file_size = file_path.stat().st_size
            audio_format = self._detect_audio_format(file_path.suffix)
            
            # Анализ аудио содержимого
            if self.librosa_available:
                metadata = await self._analyze_with_librosa(audio_file, file_size, audio_format)
            else:
                metadata = await self._analyze_basic(audio_file, file_size, audio_format)
            
            analysis_duration = time.time() - start_time
            metadata.analysis_duration = analysis_duration
            
            logger.info(f"📊 Audio analysis completed: {metadata.quality_score:.2f} quality score")
            return metadata
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            # Возвращаем базовые метаданные
            return AudioMetadata(
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                duration=0.0,
                sample_rate=16000,
                channels=1,
                format=self._detect_audio_format(file_path.suffix),
                analysis_duration=time.time() - start_time
            )
    
    async def _analyze_with_librosa(self, audio_file: str, file_size: int, audio_format: AudioFormat) -> AudioMetadata:
        """Детальный анализ с librosa"""
        import librosa
        import numpy as np
        
        # Загрузка аудио
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Анализ качества
        quality_metrics = self._calculate_quality_metrics(y, sr)
        
        # Детекция речи
        speech_segments = self._detect_speech_segments(y, sr)
        
        # Анализ шума
        noise_level = self._estimate_noise_level(y)
        
        # Определение качества
        overall_quality = self._determine_audio_quality(quality_metrics, noise_level)
        
        return AudioMetadata(
            file_path=audio_file,
            file_size=file_size,
            duration=duration,
            sample_rate=sr,
            channels=1 if len(y.shape) == 1 else y.shape[0],
            format=audio_format,
            overall_quality=overall_quality,
            quality_score=quality_metrics['overall_score'],
            snr_db=quality_metrics.get('snr_db'),
            speech_segments=speech_segments,
            total_speech_duration=sum(seg.duration for seg in speech_segments),
            background_noise_level=noise_level,
            needs_noise_reduction=noise_level > 0.3,
            needs_volume_normalization=quality_metrics['needs_volume_norm'],
            average_volume_db=quality_metrics.get('avg_volume_db'),
            peak_volume_db=quality_metrics.get('peak_volume_db')
        )
    
    async def _analyze_basic(self, audio_file: str, file_size: int, audio_format: AudioFormat) -> AudioMetadata:
        """Базовый анализ без librosa"""
        # Простой анализ через pydub если доступен
        if self.pydub_available:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            
            return AudioMetadata(
                file_path=audio_file,
                file_size=file_size,
                duration=len(audio) / 1000.0,  # в секундах
                sample_rate=audio.frame_rate,
                channels=audio.channels,
                format=audio_format,
                quality_score=0.7  # средняя оценка
            )
        else:
            # Минимальные метаданные
            return AudioMetadata(
                file_path=audio_file,
                file_size=file_size,
                duration=0.0,
                sample_rate=16000,
                channels=1,
                format=audio_format,
                quality_score=0.5
            )
    
    def _calculate_quality_metrics(self, y, sr) -> Dict[str, Any]:
        """Расчет метрик качества аудио"""
        import numpy as np
        
        # RMS энергия
        rms_energy = np.sqrt(np.mean(y**2))
        
        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(y))))
        
        # Спектральный центроид
        try:
            import librosa
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        except:
            spectral_centroid = 0.0
        
        # SNR оценка
        signal_power = np.mean(y**2)
        noise_power = np.var(y - np.mean(y))
        snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-10))
        
        # Громкость в dB
        avg_volume_db = 20 * np.log10(max(rms_energy, 1e-10))
        peak_volume_db = 20 * np.log10(max(np.max(np.abs(y)), 1e-10))
        
        # Общая оценка качества
        quality_factors = [
            min(snr_db / 20.0, 1.0),  # SNR фактор
            min(rms_energy * 10, 1.0),  # Энергия фактор  
            1.0 - min(zcr, 1.0)  # Стабильность
        ]
        overall_score = np.mean(quality_factors)
        
        return {
            'overall_score': float(overall_score),
            'rms_energy': float(rms_energy),
            'zcr': float(zcr),
            'spectral_centroid': float(spectral_centroid),
            'snr_db': float(snr_db),
            'avg_volume_db': float(avg_volume_db),
            'peak_volume_db': float(peak_volume_db),
            'needs_volume_norm': abs(avg_volume_db + 20) > 6  # Отклонение от -20dB
        }
    
    def _detect_speech_segments(self, y, sr):
        """Простая детекция сегментов речи"""
        from ..core.models.audio_metadata import SpeechSegment
        
        # Простой VAD на основе энергии
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        # Разбивка на фреймы
        frames = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            energy = np.sum(frame ** 2)
            frames.append(energy)
        
        # Пороговая детекция
        energy_threshold = np.percentile(frames, 75)  # 75-й процентиль
        
        segments = []
        in_speech = False
        start_time = 0
        
        for i, energy in enumerate(frames):
            time_point = i * hop_length / sr
            
            if energy > energy_threshold and not in_speech:
                # Начало речи
                in_speech = True
                start_time = time_point
            elif energy <= energy_threshold and in_speech:
                # Конец речи
                in_speech = False
                if time_point - start_time > 0.3:  # Минимум 300ms
                    segments.append(SpeechSegment(
                        start_time=start_time,
                        end_time=time_point,
                        confidence=0.8
                    ))
        
        # Закрытие последнего сегмента
        if in_speech:
            segments.append(SpeechSegment(
                start_time=start_time,
                end_time=len(y) / sr,
                confidence=0.8
            ))
        
        return segments
    
    def _estimate_noise_level(self, y) -> float:
        """Оценка уровня шума"""
        import numpy as np
        
        # Простая оценка через квантили
        rms = np.sqrt(np.mean(y**2))
        noise_floor = np.percentile(np.abs(y), 10)  # 10-й процентиль как шум
        
        if rms > 0:
            noise_ratio = noise_floor / rms
        else:
            noise_ratio = 1.0
        
        return min(noise_ratio, 1.0)
    
    def _determine_audio_quality(self, quality_metrics: Dict, noise_level: float) -> AudioQuality:
        """Определение общего уровня качества аудио"""
        score = quality_metrics['overall_score']
        snr = quality_metrics.get('snr_db', 0)
        
        # Корректировка на основе шума и SNR
        adjusted_score = score * (1 - noise_level) * min(snr / 20.0, 1.0)
        
        if adjusted_score >= 0.9:
            return AudioQuality.EXCELLENT
        elif adjusted_score >= 0.7:
            return AudioQuality.GOOD
        elif adjusted_score >= 0.5:
            return AudioQuality.FAIR
        elif adjusted_score >= 0.3:
            return AudioQuality.POOR
        else:
            return AudioQuality.VERY_POOR
    
    def _detect_audio_format(self, suffix: str) -> AudioFormat:
        """Определение формата аудио по расширению"""
        suffix_lower = suffix.lower().lstrip('.')
        
        format_map = {
            'wav': AudioFormat.WAV,
            'mp3': AudioFormat.MP3,
            'm4a': AudioFormat.M4A,
            'flac': AudioFormat.FLAC,
            'ogg': AudioFormat.OGG,
            'webm': AudioFormat.WEBM
        }
        
        return format_map.get(suffix_lower, AudioFormat.UNKNOWN)
    
    def _needs_format_conversion(self, metadata: AudioMetadata) -> bool:
        """Проверка необходимости конвертации формата"""
        needs_conversion = (
            metadata.sample_rate != self.target_sample_rate or
            (self.convert_to_mono and metadata.channels > 1) or
            metadata.format not in [AudioFormat.WAV, AudioFormat.FLAC]
        )
        
        return needs_conversion
    
    def _needs_noise_reduction(self, metadata: AudioMetadata) -> bool:
        """Проверка необходимости шумоподавления"""
        return (
            metadata.background_noise_level > 0.2 or
            metadata.overall_quality in [AudioQuality.POOR, AudioQuality.VERY_POOR]
        )
    
    def _create_temp_filename(self, original_file: str, suffix: str) -> str:
        """Создание имени временного файла"""
        original_path = Path(original_file)
        temp_name = f"{original_path.stem}_{suffix}_{int(time.time())}.wav"
        return str(self.temp_dir / temp_name)
    
    async def _convert_audio_format(self, audio_file: str, metadata: AudioMetadata) -> str:
        """Конвертация аудио формата"""
        output_file = self._create_temp_filename(audio_file, "converted")
        
        if self.librosa_available:
            # Конвертация через librosa
            import librosa
            import soundfile as sf
            
            y, sr = librosa.load(audio_file, sr=self.target_sample_rate, mono=self.convert_to_mono)
            sf.write(output_file, y, sr)
            
        elif self.pydub_available:
            # Конвертация через pydub
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(audio_file)
            
            # Изменение частоты дискретизации
            if audio.frame_rate != self.target_sample_rate:
                audio = audio.set_frame_rate(self.target_sample_rate)
            
            # Конвертация в моно
            if self.convert_to_mono and audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Экспорт в WAV
            audio.export(output_file, format="wav")
        
        else:
            # Без библиотек - копируем оригинал
            shutil.copy2(audio_file, output_file)
        
        return output_file
    
    async def _normalize_volume(self, audio_file: str) -> str:
        """Нормализация громкости"""
        output_file = self._create_temp_filename(audio_file, "normalized")
        
        if self.librosa_available:
            import librosa
            import soundfile as sf
            import numpy as np
            
            y, sr = librosa.load(audio_file, sr=None)
            
            # RMS нормализация до -20dB
            current_rms = np.sqrt(np.mean(y**2))
            if current_rms > 0:
                target_rms = 0.1  # Примерно -20dB
                y_normalized = y * (target_rms / current_rms)
                
                # Ограничение пиков
                y_normalized = np.clip(y_normalized, -1.0, 1.0)
            else:
                y_normalized = y
            
            sf.write(output_file, y_normalized, sr)
            
        elif self.pydub_available:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(audio_file)
            
            # Нормализация через pydub
            normalized_audio = audio.normalize()
            
            # Уменьшение громкости до комфортного уровня
            normalized_audio = normalized_audio - 6  # -6dB от максимума
            
            normalized_audio.export(output_file, format="wav")
        
        else:
            shutil.copy2(audio_file, output_file)
        
        return output_file
    
    async def _reduce_noise(self, audio_file: str) -> str:
        """Шумоподавление"""
        output_file = self._create_temp_filename(audio_file, "denoised")
        
        if self.noisereduce_available and self.librosa_available:
            import noisereduce as nr
            import librosa
            import soundfile as sf
            
            # Загрузка аудио
            y, sr = librosa.load(audio_file, sr=None)
            
            # Шумоподавление
            y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False)
            
            sf.write(output_file, y_denoised, sr)
            
        else:
            # Простое шумоподавление через фильтрацию
            await self._simple_noise_reduction(audio_file, output_file)
        
        return output_file
    
    async def _simple_noise_reduction(self, input_file: str, output_file: str):
        """Простое шумоподавление без noisereduce"""
        if self.librosa_available:
            import librosa
            import soundfile as sf
            import numpy as np
            
            y, sr = librosa.load(input_file, sr=None)
            
            # Простой high-pass фильтр для удаления низкочастотного шума
            from scipy import signal
            
            # High-pass фильтр с частотой среза 80Hz
            sos = signal.butter(4, 80, btype='high', fs=sr, output='sos')
            y_filtered = signal.sosfilt(sos, y)
            
            sf.write(output_file, y_filtered, sr)
        else:
            # Без обработки
            shutil.copy2(input_file, output_file)
    
    async def _enhance_speech(self, audio_file: str) -> str:
        """Улучшение речи"""
        output_file = self._create_temp_filename(audio_file, "enhanced")
        
        if self.librosa_available:
            import librosa
            import soundfile as sf
            import numpy as np
            
            y, sr = librosa.load(audio_file, sr=None)
            
            # Подчеркивание речевых частот (300Hz - 3400Hz)
            # Применение мягкого эквалайзера
            y_enhanced = self._apply_speech_eq(y, sr)
            
            sf.write(output_file, y_enhanced, sr)
        else:
            shutil.copy2(audio_file, output_file)
        
        return output_file
    
    def _apply_speech_eq(self, y, sr):
        """Применение эквалайзера для улучшения речи"""
        try:
            from scipy import signal
            import numpy as np
            
            # Легкое усиление средних частот (речевой диапазон)
            # Band-pass фильтр 300Hz - 3400Hz с небольшим усилением
            
            # Дизайн фильтра
            nyquist = sr / 2
            low_freq = 300 / nyquist
            high_freq = min(3400 / nyquist, 0.99)
            
            b, a = signal.butter(2, [low_freq, high_freq], btype='band')
            
            # Применение фильтра с небольшим весом
            y_filtered = signal.filtfilt(b, a, y)
            
            # Смешивание с оригиналом (80% оригинал, 20% фильтрованный)
            y_enhanced = 0.8 * y + 0.2 * y_filtered
            
            return y_enhanced
            
        except Exception as e:
            logger.warning(f"Speech EQ failed: {e}")
            return y
    
    def reduce_noise(
        self, 
        audio_data: Any, 
        sample_rate: int,
        reduction_strength: float = 0.8
    ) -> Any:
        """
        Шумоподавление
        Noise reduction
        
        Args:
            audio_data: Аудио данные (numpy array)
            sample_rate: Частота дискретизации
            reduction_strength: Сила подавления шума (0-1)
            
        Returns:
            Any: Обработанные аудио данные
        """
        try:
            import numpy as np
            
            # Проверка входных данных
            if audio_data is None:
                raise ValueError("Audio data is None")
            
            # Преобразование в numpy array если нужно
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Использование noisereduce если доступно
            if self.noisereduce_available:
                import noisereduce as nr
                return nr.reduce_noise(
                    y=audio_data, 
                    sr=sample_rate, 
                    stationary=False,
                    prop_decrease=reduction_strength
                )
            
            # Простое шумоподавление через фильтрацию
            return self._apply_simple_noise_reduction(audio_data, sample_rate, reduction_strength)
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio_data  # Возвращаем оригинальные данные при ошибке
    
    def normalize_volume(self, audio_data: Any, target_db: float = -20.0) -> Any:
        """
        Нормализация громкости
        Volume normalization
        
        Args:
            audio_data: Аудио данные
            target_db: Целевой уровень в дБ
            
        Returns:
            Any: Нормализованные аудио данные
        """
        try:
            import numpy as np
            
            # Проверка входных данных
            if audio_data is None:
                raise ValueError("Audio data is None")
            
            # Преобразование в numpy array если нужно
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Расчет текущего RMS
            current_rms = np.sqrt(np.mean(audio_data**2))
            
            if current_rms == 0:
                return audio_data  # Ничего не делаем с тишиной
            
            # Преобразование target_db в линейное значение
            target_linear = 10**(target_db / 20.0)
            
            # Вычисление коэффициента нормализации
            normalization_factor = target_linear / current_rms
            
            # Применение нормализации
            normalized_audio = audio_data * normalization_factor
            
            # Ограничение пиков (избежание клиппинга)
            max_val = np.max(np.abs(normalized_audio))
            if max_val > 1.0:
                normalized_audio = normalized_audio / max_val * 0.99
            
            return normalized_audio
            
        except Exception as e:
            logger.error(f"Volume normalization failed: {e}")
            return audio_data  # Возвращаем оригинальные данные при ошибке
    
    def resample_audio(
        self, 
        audio_data: Any, 
        original_sr: int, 
        target_sr: int
    ) -> Any:
        """
        Ресэмплинг аудио
        Resample audio to target sample rate
        
        Args:
            audio_data: Аудио данные
            original_sr: Исходная частота
            target_sr: Целевая частота
            
        Returns:
            Any: Ресэмплированные данные
        """
        try:
            import numpy as np
            
            # Проверка входных данных
            if audio_data is None:
                raise ValueError("Audio data is None")
            
            if original_sr == target_sr:
                return audio_data  # Ничего не делаем
            
            # Преобразование в numpy array если нужно
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Использование librosa для высококачественного ресэмплинга
            if self.librosa_available:
                import librosa
                return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
            
            # Простой ресэмплинг через scipy
            try:
                from scipy import signal
                
                # Вычисление нового количества сэмплов
                num_samples = int(len(audio_data) * target_sr / original_sr)
                
                # Ресэмплинг
                resampled = signal.resample(audio_data, num_samples)
                
                return resampled
                
            except ImportError:
                # Простейший линейный ресэмплинг
                return self._simple_resample(audio_data, original_sr, target_sr)
                
        except Exception as e:
            logger.error(f"Audio resampling failed: {e}")
            return audio_data  # Возвращаем оригинальные данные при ошибке
    
    def _apply_simple_noise_reduction(self, audio_data, sample_rate: int, strength: float):
        """
        Простое шумоподавление без noisereduce
        Simple noise reduction without noisereduce library
        """
        try:
            import numpy as np
            
            # High-pass фильтр для удаления низкочастотного шума
            try:
                from scipy import signal
                
                # Пороговая частота в зависимости от силы
                cutoff_freq = 50 + (strength * 80)  # от 50Гц до 130Гц
                
                # High-pass фильтр
                sos = signal.butter(4, cutoff_freq, btype='high', fs=sample_rate, output='sos')
                filtered = signal.sosfilt(sos, audio_data)
                
                # Смешивание с оригиналом
                mix_factor = 0.3 + (strength * 0.5)  # от 30% до 80% фильтра
                result = (1 - mix_factor) * audio_data + mix_factor * filtered
                
                return result
                
            except ImportError:
                # Простейшая фильтрация через скользящее среднее
                window_size = max(3, int(sample_rate * 0.001))  # 1ms окно
                smoothed = np.convolve(audio_data, np.ones(window_size)/window_size, mode='same')
                return (1 - strength) * audio_data + strength * smoothed
                
        except Exception as e:
            logger.error(f"Simple noise reduction failed: {e}")
            return audio_data
    
    def _simple_resample(self, audio_data, original_sr: int, target_sr: int):
        """
        Простейший линейный ресэмплинг
        Simplest linear resampling
        """
        try:
            import numpy as np
            
            # Вычисление новой длины
            original_length = len(audio_data)
            new_length = int(original_length * target_sr / original_sr)
            
            # Линейная интерполяция
            original_indices = np.arange(original_length)
            new_indices = np.linspace(0, original_length - 1, new_length)
            
            # Интерполяция
            resampled = np.interp(new_indices, original_indices, audio_data)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Simple resampling failed: {e}")
            return audio_data