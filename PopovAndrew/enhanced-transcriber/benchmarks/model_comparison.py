"""
Модуль для сравнения и тестирования различных ASR моделей
Поддерживает автоматическое тестирование точности, скорости и ресурсов
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import csv
from datetime import datetime
import asyncio

import numpy as np
import pandas as pd
import psutil
import GPUtil

# Метрики качества
try:
    import jiwer
except ImportError:
    jiwer = None

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Результат тестирования одной модели на одном файле"""
    model_name: str
    audio_file: str
    transcription: str
    reference_text: str = ""
    
    # Метрики точности
    wer: float = 0.0
    cer: float = 0.0
    bleu_score: float = 0.0
    
    # Метрики производительности
    processing_time: float = 0.0
    real_time_factor: float = 0.0
    cpu_usage_avg: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_avg: float = 0.0
    
    # Метаданные
    audio_duration: float = 0.0
    confidence: float = 0.0
    status: str = "unknown"
    error_message: str = ""
    timestamp: str = ""

class ModelBenchmark:
    """Система для бенчмаркинга ASR моделей"""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_models = {
            'enhanced_transcriber': self._test_enhanced_transcriber,
            'faster_whisper': self._test_faster_whisper,
            'wav2vec2': self._test_wav2vec2,
            'vosk': self._test_vosk,
            'speechbrain': self._test_speechbrain,
            'nemo_asr': self._test_nemo_asr
        }
        
        self.system_monitor = SystemMonitor()
    
    async def run_comprehensive_benchmark(self, 
                                        audio_files: List[str],
                                        reference_texts: List[str],
                                        models_to_test: List[str],
                                        benchmark_name: str = "comprehensive_test") -> Dict[str, Any]:
        """
        Запуск полного бенчмарка всех моделей на всех файлах
        """
        logger.info(f"Запуск бенчмарка '{benchmark_name}' для {len(models_to_test)} моделей и {len(audio_files)} файлов")
        
        if len(audio_files) != len(reference_texts):
            raise ValueError("Количество аудиофайлов должно совпадать с количеством эталонных текстов")
        
        benchmark_start = time.time()
        all_results = []
        
        # Тестирование каждой модели
        for model_name in models_to_test:
            if model_name not in self.supported_models:
                logger.warning(f"Модель '{model_name}' не поддерживается, пропуск")
                continue
            
            logger.info(f"Тестирование модели: {model_name}")
            model_results = []
            
            # Тестирование на каждом файле
            for i, (audio_file, reference_text) in enumerate(zip(audio_files, reference_texts)):
                try:
                    logger.info(f"  Файл {i+1}/{len(audio_files)}: {Path(audio_file).name}")
                    
                    result = await self._test_single_model_file(
                        model_name, audio_file, reference_text
                    )
                    model_results.append(result)
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Ошибка тестирования {model_name} на {audio_file}: {e}")
                    error_result = BenchmarkResult(
                        model_name=model_name,
                        audio_file=audio_file,
                        transcription="",
                        reference_text=reference_text,
                        status="error",
                        error_message=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    model_results.append(error_result)
                    all_results.append(error_result)
            
            # Сохранение промежуточных результатов модели
            await self._save_model_results(model_name, model_results, benchmark_name)
        
        # Создание сводного отчета
        benchmark_time = time.time() - benchmark_start
        summary = self._create_benchmark_summary(all_results, benchmark_name, benchmark_time)
        
        # Сохранение полного отчета
        await self._save_comprehensive_report(all_results, summary, benchmark_name)
        
        logger.info(f"Бенчмарк '{benchmark_name}' завершен за {benchmark_time:.1f}s")
        return summary
    
    async def _test_single_model_file(self, model_name: str, audio_file: str, reference_text: str) -> BenchmarkResult:
        """Тестирование одной модели на одном файле"""
        
        # Получение длительности аудио
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_file)
        except:
            audio_duration = 0.0
        
        # Запуск мониторинга системы
        self.system_monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Вызов соответствующей функции тестирования
            test_func = self.supported_models[model_name]
            transcription, confidence, additional_info = await test_func(audio_file)
            
            processing_time = time.time() - start_time
            
            # Остановка мониторинга
            system_stats = self.system_monitor.stop_monitoring()
            
            # Вычисление метрик точности
            wer, cer, bleu = self._calculate_accuracy_metrics(transcription, reference_text)
            
            # Создание результата
            result = BenchmarkResult(
                model_name=model_name,
                audio_file=audio_file,
                transcription=transcription,
                reference_text=reference_text,
                wer=wer,
                cer=cer,
                bleu_score=bleu,
                processing_time=processing_time,
                real_time_factor=processing_time / max(audio_duration, 0.1),
                cpu_usage_avg=system_stats.get('cpu_avg', 0),
                memory_usage_mb=system_stats.get('memory_mb', 0),
                gpu_usage_avg=system_stats.get('gpu_avg', 0),
                audio_duration=audio_duration,
                confidence=confidence,
                status="completed",
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.system_monitor.stop_monitoring()
            
            logger.error(f"Ошибка в модели {model_name}: {e}")
            
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_file,
                transcription="",
                reference_text=reference_text,
                processing_time=processing_time,
                audio_duration=audio_duration,
                status="error",
                error_message=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    # Функции тестирования отдельных моделей
    
    async def _test_enhanced_transcriber(self, audio_file: str) -> Tuple[str, float, Dict]:
        """Тестирование Enhanced Transcriber (наша основная система)"""
        try:
            # Импорт в функции для избежания циклических зависимостей
            from enhanced_transcriber import EnhancedTranscriber
            
            system = EnhancedTranscriber(
                target_quality=0.95,
                enable_audio_enhancement=True,
                enable_quality_assessment=True
            )
            
            await system.initialize()
            result = await system.transcribe(audio_file, language="ru")
            
            if result and result.text:
                return result.text, result.confidence, {"models_used": len(system.models)}
            else:
                return "", 0.0, {"error": "Empty result"}
                
        except Exception as e:
            raise Exception(f"Enhanced Transcriber error: {e}")
    
    async def _test_faster_whisper(self, audio_file: str) -> Tuple[str, float, Dict]:
        """Тестирование Faster-Whisper"""
        try:
            from faster_whisper import WhisperModel
            
            model = WhisperModel("medium", device="cuda" if self._gpu_available() else "cpu")
            segments, info = model.transcribe(audio_file, language="ru")
            
            text = " ".join([segment.text for segment in segments])
            confidence = 0.85  # Faster-whisper не предоставляет точную уверенность
            
            return text, confidence, {"detected_language": info.language}
            
        except ImportError:
            raise Exception("faster-whisper не установлен")
        except Exception as e:
            raise Exception(f"Faster-Whisper error: {e}")
    
    async def _test_wav2vec2(self, audio_file: str) -> Tuple[str, float, Dict]:
        """Тестирование Wav2Vec 2.0"""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
            import torch
            import librosa
            
            model_name = "facebook/wav2vec2-large-960h"
            tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            # Загрузка аудио
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # Предсказание
            input_values = tokenizer(audio, return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])
            
            return transcription, 0.80, {"model": model_name}
            
        except ImportError:
            raise Exception("transformers не установлен")
        except Exception as e:
            raise Exception(f"Wav2Vec2 error: {e}")
    
    async def _test_vosk(self, audio_file: str) -> Tuple[str, float, Dict]:
        """Тестирование Vosk"""
        try:
            import vosk
            import json
            import wave
            
            # Загрузка модели (требуется предварительная загрузка)
            model_path = "vosk-model-ru"  # Нужно скачать модель
            if not Path(model_path).exists():
                raise Exception(f"Модель Vosk не найдена в {model_path}")
            
            model = vosk.Model(model_path)
            rec = vosk.KaldiRecognizer(model, 16000)
            
            # Обработка аудио
            wf = wave.open(audio_file, 'rb')
            text_parts = []
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text_parts.append(result.get('text', ''))
            
            final_result = json.loads(rec.FinalResult())
            text_parts.append(final_result.get('text', ''))
            
            full_text = ' '.join(text_parts)
            return full_text, 0.75, {"model": "vosk-ru"}
            
        except ImportError:
            raise Exception("vosk не установлен")
        except Exception as e:
            raise Exception(f"Vosk error: {e}")
    
    async def _test_speechbrain(self, audio_file: str) -> Tuple[str, float, Dict]:
        """Тестирование SpeechBrain"""
        try:
            from speechbrain.pretrained import EncoderDecoderASR
            
            asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-wav2vec2-commonvoice-ru",
                savedir="tmp_speechbrain"
            )
            
            transcription = asr_model.transcribe_file(audio_file)
            return transcription, 0.80, {"model": "speechbrain-wav2vec2-ru"}
            
        except ImportError:
            raise Exception("speechbrain не установлен")
        except Exception as e:
            raise Exception(f"SpeechBrain error: {e}")
    
    async def _test_nemo_asr(self, audio_file: str) -> Tuple[str, float, Dict]:
        """Тестирование NVIDIA NeMo ASR"""
        try:
            import nemo.collections.asr as nemo_asr
            
            # Загрузка предобученной модели
            asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name="nvidia/stt_ru_conformer_ctc_medium"
            )
            
            transcription = asr_model.transcribe(paths2audio_files=[audio_file])[0]
            return transcription, 0.85, {"model": "nemo-conformer-ru"}
            
        except ImportError:
            raise Exception("nemo_toolkit не установлен")
        except Exception as e:
            raise Exception(f"NeMo ASR error: {e}")
    
    def _calculate_accuracy_metrics(self, hypothesis: str, reference: str) -> Tuple[float, float, float]:
        """Вычисление метрик точности"""
        if not reference.strip():
            return 0.0, 0.0, 0.0
        
        # Очистка текстов
        hypothesis = hypothesis.lower().strip()
        reference = reference.lower().strip()
        
        # WER (Word Error Rate)
        wer = 0.0
        if jiwer:
            try:
                wer = jiwer.wer(reference, hypothesis)
            except:
                wer = 1.0  # Максимальная ошибка если не удалось вычислить
        
        # CER (Character Error Rate)
        cer = 0.0
        if jiwer:
            try:
                cer = jiwer.cer(reference, hypothesis)
            except:
                cer = 1.0
        
        # Простая BLEU-подобная метрика
        bleu = self._simple_bleu(hypothesis, reference)
        
        return wer, cer, bleu
    
    def _simple_bleu(self, hypothesis: str, reference: str) -> float:
        """Упрощенная BLEU метрика"""
        hyp_words = set(hypothesis.split())
        ref_words = set(reference.split())
        
        if not ref_words:
            return 0.0
        
        intersection = len(hyp_words & ref_words)
        union = len(ref_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _gpu_available(self) -> bool:
        """Проверка доступности GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _create_benchmark_summary(self, results: List[BenchmarkResult], 
                                benchmark_name: str, total_time: float) -> Dict[str, Any]:
        """Создание сводки по результатам бенчмарка"""
        
        # Группировка по моделям
        by_model = {}
        for result in results:
            model = result.model_name
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        # Агрегация метрик по моделям
        model_stats = {}
        for model, model_results in by_model.items():
            successful = [r for r in model_results if r.status == "completed"]
            
            if successful:
                model_stats[model] = {
                    "total_files": len(model_results),
                    "successful": len(successful),
                    "success_rate": len(successful) / len(model_results) * 100,
                    "avg_wer": np.mean([r.wer for r in successful]),
                    "avg_cer": np.mean([r.cer for r in successful]),
                    "avg_bleu": np.mean([r.bleu_score for r in successful]),
                    "avg_processing_time": np.mean([r.processing_time for r in successful]),
                    "avg_rtf": np.mean([r.real_time_factor for r in successful]),
                    "avg_cpu_usage": np.mean([r.cpu_usage_avg for r in successful]),
                    "avg_memory_mb": np.mean([r.memory_usage_mb for r in successful]),
                    "avg_confidence": np.mean([r.confidence for r in successful])
                }
            else:
                model_stats[model] = {
                    "total_files": len(model_results),
                    "successful": 0,
                    "success_rate": 0.0,
                    "error": "Все тесты неудачны"
                }
        
        return {
            "benchmark_name": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_time,
            "total_tests": len(results),
            "models_tested": list(by_model.keys()),
            "model_statistics": model_stats,
            "best_accuracy": self._find_best_model(model_stats, "avg_wer", lower_is_better=True),
            "fastest_model": self._find_best_model(model_stats, "avg_rtf", lower_is_better=True),
            "most_confident": self._find_best_model(model_stats, "avg_confidence", lower_is_better=False)
        }
    
    def _find_best_model(self, model_stats: Dict, metric: str, lower_is_better: bool) -> Dict:
        """Поиск лучшей модели по метрике"""
        valid_models = {k: v for k, v in model_stats.items() 
                       if "error" not in v and metric in v}
        
        if not valid_models:
            return {"model": "none", "value": None}
        
        if lower_is_better:
            best_model = min(valid_models.items(), key=lambda x: x[1][metric])
        else:
            best_model = max(valid_models.items(), key=lambda x: x[1][metric])
        
        return {"model": best_model[0], "value": best_model[1][metric]}
    
    async def _save_model_results(self, model_name: str, results: List[BenchmarkResult], 
                                benchmark_name: str):
        """Сохранение результатов модели"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_{model_name}_{timestamp}.json"
        
        results_data = []
        for result in results:
            results_data.append({
                "model_name": result.model_name,
                "audio_file": result.audio_file,
                "transcription": result.transcription,
                "reference_text": result.reference_text,
                "wer": result.wer,
                "cer": result.cer,
                "bleu_score": result.bleu_score,
                "processing_time": result.processing_time,
                "real_time_factor": result.real_time_factor,
                "cpu_usage_avg": result.cpu_usage_avg,
                "memory_usage_mb": result.memory_usage_mb,
                "gpu_usage_avg": result.gpu_usage_avg,
                "audio_duration": result.audio_duration,
                "confidence": result.confidence,
                "status": result.status,
                "error_message": result.error_message,
                "timestamp": result.timestamp
            })
        
        with open(self.results_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    async def _save_comprehensive_report(self, results: List[BenchmarkResult], 
                                       summary: Dict, benchmark_name: str):
        """Сохранение полного отчета"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON отчет
        json_filename = f"{benchmark_name}_full_report_{timestamp}.json"
        full_report = {
            "summary": summary,
            "detailed_results": [
                {
                    "model_name": r.model_name,
                    "audio_file": r.audio_file,
                    "wer": r.wer,
                    "cer": r.cer,
                    "processing_time": r.processing_time,
                    "confidence": r.confidence,
                    "status": r.status
                } for r in results
            ]
        }
        
        with open(self.results_dir / json_filename, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        
        # CSV отчет
        csv_filename = f"{benchmark_name}_results_{timestamp}.csv"
        with open(self.results_dir / csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Audio File', 'WER', 'CER', 'BLEU', 'Processing Time', 
                'RTF', 'Confidence', 'Status', 'CPU %', 'Memory MB'
            ])
            
            for r in results:
                writer.writerow([
                    r.model_name, Path(r.audio_file).name, 
                    f"{r.wer:.3f}", f"{r.cer:.3f}", f"{r.bleu_score:.3f}",
                    f"{r.processing_time:.2f}", f"{r.real_time_factor:.2f}",
                    f"{r.confidence:.3f}", r.status,
                    f"{r.cpu_usage_avg:.1f}", f"{r.memory_usage_mb:.1f}"
                ])
        
        # Текстовый сводный отчет
        txt_filename = f"{benchmark_name}_summary_{timestamp}.txt"
        self._create_text_summary(summary, self.results_dir / txt_filename)
    
    def _create_text_summary(self, summary: Dict, output_path: Path):
        """Создание текстового сводного отчета"""
        lines = [
            f"СВОДНЫЙ ОТЧЕТ БЕНЧМАРКА",
            f"Название: {summary['benchmark_name']}",
            f"Дата: {summary['timestamp']}",
            f"Общее время: {summary['total_duration']:.1f}s",
            f"Всего тестов: {summary['total_tests']}",
            f"",
            f"РЕЗУЛЬТАТЫ ПО МОДЕЛЯМ:",
        ]
        
        for model, stats in summary["model_statistics"].items():
            if "error" not in stats:
                lines.extend([
                    f"",
                    f"  {model}:",
                    f"    Успешность: {stats['success_rate']:.1f}% ({stats['successful']}/{stats['total_files']})",
                    f"    WER: {stats['avg_wer']:.3f}",
                    f"    CER: {stats['avg_cer']:.3f}",
                    f"    BLEU: {stats['avg_bleu']:.3f}",
                    f"    Среднее время: {stats['avg_processing_time']:.2f}s",
                    f"    RTF: {stats['avg_rtf']:.2f}x",
                    f"    Уверенность: {stats['avg_confidence']:.3f}"
                ])
        
        lines.extend([
            f"",
            f"ЛУЧШИЕ РЕЗУЛЬТАТЫ:",
            f"  Точность (низкий WER): {summary['best_accuracy']['model']} ({summary['best_accuracy']['value']:.3f})",
            f"  Скорость (низкий RTF): {summary['fastest_model']['model']} ({summary['fastest_model']['value']:.2f}x)",
            f"  Уверенность: {summary['most_confident']['model']} ({summary['most_confident']['value']:.3f})"
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class SystemMonitor:
    """Мониторинг системных ресурсов во время тестирования"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
    
    def start_monitoring(self):
        """Начать мониторинг"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Остановить мониторинг и получить статистику"""
        self.monitoring = False
        
        # Получение текущих значений
        cpu_percent = psutil.cpu_percent()
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        
        gpu_usage = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except:
            pass
        
        return {
            "cpu_avg": cpu_percent,
            "memory_mb": memory_mb,
            "gpu_avg": gpu_usage
        }


# Функция для быстрого запуска бенчмарка
async def quick_benchmark(audio_files: List[str], reference_texts: List[str],
                         models: List[str] = None) -> Dict[str, Any]:
    """Быстрый запуск бенчмарка с дефолтными настройками"""
    if models is None:
        models = ["enhanced_transcriber", "faster_whisper"]
    
    benchmark = ModelBenchmark()
    return await benchmark.run_comprehensive_benchmark(
        audio_files, reference_texts, models, "quick_test"
    )
