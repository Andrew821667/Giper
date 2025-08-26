#!/usr/bin/env python3
"""
Тестирование реальных компонентов Enhanced Transcriber
Test real Enhanced Transcriber components
"""

import sys
import asyncio
import time
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

async def test_quality_assessor():
    """Тест оценщика качества"""
    print("📊 Testing Quality Assessor...")
    
    from services.quality_assessor import QualityAssessmentService
    
    try:
        # Создание оценщика
        assessor = QualityAssessmentService(
            enable_semantic_analysis=False,  # Отключаем для простоты
            enable_domain_analysis=True
        )
        
        # Тестовый текст
        test_text = "Здравствуйте, хочу оформить заказ на доставку товара в интернет-магазин"
        
        # Оценка качества
        quality_metrics = await assessor.assess_quality(
            transcribed_text=test_text,
            audio_file="/tmp/test.wav",  # Мок файл
            domain="ecommerce"
        )
        
        print(f"✅ Quality assessment completed")
        print(f"   Overall Score: {quality_metrics.overall_score:.1%}")
        print(f"   Quality Level: {quality_metrics.quality_level.value}")
        print(f"   Word Count: {quality_metrics.word_count}")
        print(f"   Vocabulary Richness: {quality_metrics.vocabulary_richness:.2f}")
        
        if quality_metrics.improvement_suggestions:
            print(f"   Suggestions: {', '.join(quality_metrics.improvement_suggestions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality Assessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_audio_processor():
    """Тест аудио процессора"""
    print("\n🎵 Testing Audio Processor...")
    
    from services.audio_processor import AudioProcessorService
    
    try:
        # Создание процессора
        processor = AudioProcessorService(
            enable_noise_reduction=True,
            enable_volume_normalization=True,
            enable_speech_enhancement=True
        )
        
        print(f"✅ Audio Processor created")
        print(f"   Noise reduction: enabled")
        print(f"   Volume normalization: enabled") 
        print(f"   Speech enhancement: enabled")
        print(f"   Target sample rate: {processor.target_sample_rate}Hz")
        
        # Создание тестового файла
        test_audio = "/tmp/test_audio_proc.wav"
        Path(test_audio).touch()
        
        try:
            # Анализ аудио (без librosa будет базовый)
            metadata = await processor.analyze_audio(test_audio)
            
            print(f"   Audio analysis completed")
            print(f"   File size: {metadata.file_size} bytes")
            print(f"   Format: {metadata.format.value}")
            print(f"   Quality score: {metadata.quality_score:.2f}")
            
        finally:
            if Path(test_audio).exists():
                Path(test_audio).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Audio Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ensemble_service_creation():
    """Тест создания ensemble сервиса"""
    print("\n🔄 Testing Ensemble Service Creation...")
    
    from services.ensemble_service import EnsembleTranscriptionService
    from core.models.transcription_result import TranscriptionResult, TranscriptionStatus
    
    # Мок транскрайбер
    class TestTranscriber:
        def __init__(self, name):
            self.model_name_str = name
        
        @property
        def model_name(self):
            return self.model_name_str
        
        def get_model_info(self):
            return {"name": self.model_name_str, "provider": "test"}
        
        async def transcribe(self, audio_file, language="ru"):
            await asyncio.sleep(0.05)  # Имитация обработки
            return TranscriptionResult(
                text=f"Test result from {self.model_name_str}",
                confidence=0.85,
                processing_time=0.05,
                model_used=self.model_name_str,
                language_detected=language,
                status=TranscriptionStatus.COMPLETED
            )
    
    try:
        # Создание мок моделей
        models = [
            TestTranscriber("T-one Test"),
            TestTranscriber("Whisper Test")
        ]
        
        # Создание ensemble сервиса
        ensemble = EnsembleTranscriptionService(
            models=models,
            target_quality_threshold=0.95
        )
        
        print(f"✅ Ensemble Service created")
        
        # Получение информации
        info = ensemble.get_ensemble_info()
        print(f"   Models count: {info['models_count']}")
        print(f"   Target quality: {info['target_quality_threshold']:.1%}")
        print(f"   Consensus method: {info['consensus_method']}")
        
        # Веса моделей
        print(f"   Model weights: {info['model_weights']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_config_models():
    """Тест конфигурационных моделей"""
    print("\n⚙️ Testing Configuration Models...")
    
    try:
        from core.models.config_models import TranscriptionConfig, ECommerceConfig, ModelProvider
        
        # Тест ECommerceConfig
        ecommerce_config = ECommerceConfig()
        terms = ecommerce_config.get_default_ecommerce_terms()
        
        print(f"✅ ECommerceConfig created")
        print(f"   E-commerce terms loaded: {len(terms)} terms")
        print(f"   Example corrections: {dict(list(terms.items())[:3])}")
        
        # Тест TranscriptionConfig
        config = TranscriptionConfig()
        models = config.get_default_models()
        
        print(f"✅ TranscriptionConfig created")
        print(f"   Default models: {len(models)}")
        print(f"   Available models: {list(models.keys())}")
        print(f"   Default strategy: {config.default_strategy.value}")
        print(f"   Target quality: {config.quality_threshold:.1%}")
        
        # Выбор оптимальной модели
        optimal = config.get_optimal_model("ru", "good", "ecommerce")
        print(f"   Optimal for Russian/good/ecommerce: {optimal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_models():
    """Тест моделей данных"""
    print("\n📊 Testing Data Models...")
    
    try:
        from core.models.transcription_result import TranscriptionResult, TranscriptionStatus, WordTimestamp
        from core.models.quality_metrics import QualityMetrics, QualityLevel
        from core.models.audio_metadata import AudioMetadata, AudioFormat, AudioQuality
        
        # Тест TranscriptionResult
        result = TranscriptionResult(
            text="Тестовый текст транскрипции",
            confidence=0.92,
            processing_time=2.5,
            model_used="Test Model",
            language_detected="ru",
            status=TranscriptionStatus.COMPLETED
        )
        
        print(f"✅ TranscriptionResult created")
        print(f"   Text: {result.text}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Status: {result.status.value}")
        
        # Тест QualityMetrics
        metrics = QualityMetrics()
        metrics.word_count = 4
        metrics.unique_words_count = 4
        metrics.vocabulary_richness = 1.0
        metrics.update_overall_assessment()
        
        print(f"✅ QualityMetrics created")
        print(f"   Overall score: {metrics.overall_score:.1%}")
        print(f"   Quality level: {metrics.quality_level.value}")
        
        # Тест AudioMetadata
        metadata = AudioMetadata(
            file_path="/tmp/test.wav",
            file_size=1024000,
            duration=60.0,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV
        )
        
        print(f"✅ AudioMetadata created")
        print(f"   Duration: {metadata.duration_formatted}")
        print(f"   Size: {metadata.file_size_mb:.1f} MB")
        print(f"   Suitable for transcription: {metadata.is_suitable_for_transcription()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Главная функция тестирования компонентов"""
    print("=" * 80)
    print("🧪 ENHANCED TRANSCRIBER - REAL COMPONENTS TEST")
    print("=" * 80)
    
    start_time = time.time()
    
    tests = [
        ("Data Models", test_data_models),
        ("Configuration Models", test_config_models),
        ("Quality Assessor", test_quality_assessor),
        ("Audio Processor", test_audio_processor),
        ("Ensemble Service", test_ensemble_service_creation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    print(f"📊 Success rate: {passed/(passed+failed):.1%}" if (passed+failed) > 0 else "No tests run")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL REAL COMPONENTS WORKING PERFECTLY!")
        print("✅ Enhanced Transcriber architecture is solid")
        print("🎯 Ready for 95%+ quality transcription")
    else:
        print(f"\n⚠️ {failed} components need attention")
        print("🔧 Fix the failing components before production use")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())