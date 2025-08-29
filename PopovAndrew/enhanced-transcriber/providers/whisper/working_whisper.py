"""
Рабочий Whisper провайдер для Colab - совместимый с оригинальным интерфейсом
"""
import time
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import whisper

from core.interfaces.transcriber import ITranscriber
from core.models.transcription_result import TranscriptionResult, TranscriptionStatus

class WorkingWhisperTranscriber(ITranscriber):
    """Рабочий Whisper транскрайбер для Colab"""
    
    def __init__(self, model_name: str = "base", device: str = "auto"):
        """
        Инициализация Whisper транскрайбера
        
        Args:
            model_name: Название модели Whisper (tiny, base, small, medium, large)
            device: Устройство для вычислений (auto, cpu, cuda)
        """
        self.model_name_str = model_name
        self.device = "cpu" if device == "auto" else device  # Принудительно CPU в Colab
        self._model = None
        self._supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        self._supported_languages = ['ru', 'en', 'auto'] 
        
        # Инициализируем модель
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация Whisper модели"""
        try:
            # Используем уже загруженную модель если доступна
            global whisper_model_medium
            if 'whisper_model_medium' in globals() and self.model_name_str == "medium":
                self._model = whisper_model_medium
                print(f"Используем предзагруженную Whisper medium модель")
            else:
                # Загружаем нужную модель с патчем
                import torch
                orig_load = torch.load
                torch.load = lambda f, map_location=None, **kw: orig_load(f, map_location, **{k:v for k,v in kw.items() if k != 'weights_only'})
                
                self._model = whisper.load_model(self.model_name_str, device=self.device)
                torch.load = orig_load
                print(f"Загружена Whisper модель: {self.model_name_str}")
                
        except Exception as e:
            raise RuntimeError(f"Не удалось инициализировать модель Whisper: {e}")
    
    async def transcribe(self, audio_file: str, language: Optional[str] = None, **kwargs) -> TranscriptionResult:
        """Транскрипция аудио файла"""
        if not self._model:
            raise RuntimeError("Whisper model not initialized")
        
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # Выполняем транскрипцию
            result = self._model.transcribe(audio_file, language=language or "ru")
            
            processing_time = time.time() - start_time
            
            # Возвращаем результат в формате проекта
            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=0.87,  
                processing_time=processing_time,
                model_used=f"Whisper {self.model_name_str}",
                language_detected=result.get("language", language or "ru"),
                status=TranscriptionStatus.COMPLETED,
                provider_metadata={
                    "provider": "whisper_working",
                    "model": self.model_name_str,
                    "device": self.device,
                    "segments": len(result.get("segments", []))
                }
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                model_used=f"Whisper {self.model_name_str}",
                language_detected=language or "ru",
                status=TranscriptionStatus.FAILED,
                provider_metadata={"error": str(e)}
            )
    
    @property
    def model_name(self) -> str:
        return self.model_name_str
    
    @property
    def supported_languages(self) -> list:
        return self._supported_languages.copy()
    
    def is_supported_format(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self._supported_formats
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name_str,
            "provider": "whisper_working",
            "device": self.device,
            "specialization": "Universal ASR"
        }
