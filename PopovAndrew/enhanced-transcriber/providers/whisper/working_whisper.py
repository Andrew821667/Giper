"""
Рабочий Whisper провайдер для Google Colab с GPU поддержкой
"""
import time
import torch
import whisper
from typing import Optional, Dict, Any
from pathlib import Path

from core.interfaces.transcriber import ITranscriber
from core.models.transcription_result import TranscriptionResult, TranscriptionStatus

class WorkingWhisperTranscriber(ITranscriber):
    def __init__(self, model_name: str = "medium", device: str = "auto"):
        self.model_name_str = model_name
        # GPU поддержка
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        self._supported_languages = ['ru', 'en', 'auto']
        
        print(f"Whisper будет использовать устройство: {self.device}")
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация с патчем torch.load"""
        try:
            # Патч для Colab совместимости
            orig_load = torch.load
            torch.load = lambda f, map_location=None, **kw: orig_load(f, map_location, **{k:v for k,v in kw.items() if k != 'weights_only'})
            
            self._model = whisper.load_model(self.model_name_str, device=self.device)
            torch.load = orig_load
            print(f"Whisper {self.model_name_str} загружен на {self.device}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки Whisper: {e}")
    
    async def transcribe(self, audio_file: str, language: Optional[str] = None, **kwargs) -> TranscriptionResult:
        if not self._model or not Path(audio_file).exists():
            raise RuntimeError("Модель не готова или файл не найден")
        
        start_time = time.time()
        try:
            result = self._model.transcribe(audio_file, language=language or "ru")
            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=0.87,
                processing_time=time.time() - start_time,
                model_used=f"Whisper {self.model_name_str}",
                language_detected=result.get("language", language or "ru"),
                status=TranscriptionStatus.COMPLETED,
                provider_metadata={"provider": "whisper_working", "model": self.model_name_str, "device": self.device}
            )
        except Exception as e:
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=time.time() - start_time,
                model_used=f"Whisper {self.model_name_str}", language_detected=language or "ru",
                status=TranscriptionStatus.FAILED, provider_metadata={"error": str(e)}
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
        return {"name": self.model_name_str, "provider": "whisper_working", "device": self.device}
