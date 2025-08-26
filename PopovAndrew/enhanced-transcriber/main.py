#!/usr/bin/env python3
"""
Enhanced Transcriber - Production Main Entry Point
Точка входа для продакшн использования Enhanced Transcriber

Команды:
python main.py transcribe audio.wav                    # Одиночная транскрипция
python main.py batch folder/                           # Пакетная обработка
python main.py demo                                     # Демонстрация системы
python main.py status                                   # Статус системы
"""

import asyncio
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import List, Optional
import os
from datetime import datetime

# Добавляем текущую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_transcriber import EnhancedTranscriber
from core.models.transcription_result import TranscriptionResult

# Настройка логирования
def setup_logging(verbose: bool = False):
    """Настройка системы логирования"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Создание директории для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Файловый хендлер
    log_file = log_dir / f"enhanced_transcriber_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Настройка root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Отключение излишне подробных логов от библиотек
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


class TranscriptionCLI:
    """CLI интерфейс для Enhanced Transcriber"""
    
    def __init__(self, logger):
        self.logger = logger
        self.transcriber: Optional[EnhancedTranscriber] = None
    
    async def initialize_transcriber(
        self, 
        openai_api_key: Optional[str] = None,
        target_quality: float = 0.95,
        domain: str = "ecommerce"
    ) -> bool:
        """Инициализация Enhanced Transcriber"""
        self.logger.info("🚀 Initializing Enhanced Transcriber...")
        
        try:
            # API ключ из переменной окружения
            if not openai_api_key:
                openai_api_key = os.getenv('OPENAI_API_KEY')
            
            # Создание transcriber
            self.transcriber = EnhancedTranscriber(
                openai_api_key=openai_api_key,
                target_quality=target_quality,
                domain=domain
            )
            
            # Инициализация
            init_results = await self.transcriber.initialize()
            
            if not init_results["ready"]:
                self.logger.error("❌ Enhanced Transcriber initialization failed")
                return False
            
            # Вывод статуса инициализации
            self._print_initialization_status(init_results)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced Transcriber: {e}")
            return False
    
    def _print_initialization_status(self, init_results: dict):
        """Красивый вывод статуса инициализации"""
        print("\\n" + "="*60)
        print("🎯 ENHANCED TRANSCRIBER - READY FOR 95%+ QUALITY")
        print("="*60)
        
        print("\\n🤖 MODELS STATUS:")
        for model in init_results["models"]:
            status_emoji = "✅" if model["status"] == "success" else "❌"
            priority = model.get("priority", "unknown")
            print(f"   {status_emoji} {model['name']} ({priority} priority)")
            if model["status"] == "failed":
                print(f"      Error: {model.get('error', 'Unknown')}")
        
        print("\\n🔧 SERVICES STATUS:")
        services = [
            ("Audio Processor", init_results["audio_processor"]),
            ("Quality Assessor", init_results["quality_assessor"]), 
            ("Ensemble Service", init_results["ensemble_service"])
        ]
        
        for service_name, available in services:
            status_emoji = "✅" if available else "❌"
            print(f"   {status_emoji} {service_name}")
        
        print("\\n🎯 TARGET CONFIGURATION:")
        print(f"   Target Quality: 95%+")
        print(f"   Domain: E-commerce (Гипер Онлайн)")
        print(f"   Language: Russian (primary)")
        
        if init_results["ready"]:
            print("\\n🚀 SYSTEM STATUS: READY FOR PRODUCTION")
        else:
            print("\\n⚠️ SYSTEM STATUS: INITIALIZATION ISSUES")
        
        print("="*60 + "\\n")
    
    async def transcribe_single(
        self, 
        audio_file: str,
        language: str = "ru",
        output_format: str = "text",
        output_file: Optional[str] = None
    ) -> bool:
        """Транскрипция одного файла"""
        if not self.transcriber:
            self.logger.error("Transcriber not initialized")
            return False
        
        if not Path(audio_file).exists():
            self.logger.error(f"Audio file not found: {audio_file}")
            return False
        
        try:
            self.logger.info(f"🎵 Starting transcription: {audio_file}")
            
            # Транскрипция
            result = await self.transcriber.transcribe(
                audio_file=audio_file,
                language=language,
                force_target_quality=True
            )
            
            # Вывод результата
            self._print_transcription_result(result, audio_file)
            
            # Сохранение результата
            if output_file:
                await self._save_result(result, output_file, output_format)
            
            return result.text and result.status != "failed"
            
        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {e}")
            return False
    
    async def transcribe_batch(
        self,
        input_path: str,
        language: str = "ru",
        output_dir: Optional[str] = None,
        max_concurrent: int = 2
    ) -> bool:
        """Пакетная транскрипция"""
        if not self.transcriber:
            self.logger.error("Transcriber not initialized")
            return False
        
        # Поиск аудио файлов
        audio_files = self._find_audio_files(input_path)
        if not audio_files:
            self.logger.error(f"No audio files found in: {input_path}")
            return False
        
        self.logger.info(f"📦 Found {len(audio_files)} audio files for batch processing")
        
        try:
            # Пакетная транскрипция
            results = await self.transcriber.transcribe_batch(
                audio_files=audio_files,
                language=language,
                max_concurrent=max_concurrent
            )
            
            # Обработка результатов
            successful = 0
            for i, result in enumerate(results):
                if result.text and result.status != "failed":
                    successful += 1
                    
                    # Сохранение результата
                    if output_dir:
                        output_file = Path(output_dir) / f"{Path(audio_files[i]).stem}_transcription.txt"
                        await self._save_result(result, str(output_file), "text")
            
            # Итоговая статистика
            print(f"\\n📊 BATCH RESULTS: {successful}/{len(audio_files)} successful")
            
            # Статистика качества
            quality_stats = self.transcriber.get_quality_stats()
            if "target_achievement_rate" in quality_stats:
                print(f"🎯 Target Quality Achievement: {quality_stats['target_achievement_rate']:.1%}")
            
            return successful > 0
            
        except Exception as e:
            self.logger.error(f"❌ Batch transcription failed: {e}")
            return False
    
    def _find_audio_files(self, input_path: str) -> List[str]:
        """Поиск аудио файлов в директории"""
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
        audio_files = []
        
        path = Path(input_path)
        
        if path.is_file():
            if path.suffix.lower() in audio_extensions:
                audio_files.append(str(path))
        elif path.is_dir():
            for ext in audio_extensions:
                audio_files.extend([str(f) for f in path.glob(f"*{ext}")])
                audio_files.extend([str(f) for f in path.glob(f"**/*{ext}")])
        
        return sorted(set(audio_files))
    
    def _print_transcription_result(self, result: TranscriptionResult, audio_file: str):
        """Красивый вывод результата транскрипции"""
        print("\\n" + "="*80)
        print("📝 TRANSCRIPTION RESULT")
        print("="*80)
        
        # Основная информация
        print(f"📁 File: {Path(audio_file).name}")
        print(f"🎯 Model: {result.model_used}")
        print(f"🌍 Language: {result.language_detected}")
        print(f"⏱️ Processing Time: {result.processing_time:.2f}s")
        print(f"📊 Confidence: {result.confidence:.1%}")
        
        if result.audio_duration:
            print(f"🕐 Audio Duration: {result.audio_duration:.1f}s")
        
        # Информация о качестве
        if result.quality_metrics:
            quality = result.quality_metrics.overall_score
            level = result.quality_metrics.quality_level.value
            target_achieved = "🎯 TARGET ACHIEVED" if quality >= 0.95 else "⚠️ Below Target"
            
            print(f"\\n🏆 QUALITY ASSESSMENT:")
            print(f"   Overall Score: {quality:.1%} ({level})")
            print(f"   Status: {target_achieved}")
            
            if result.quality_metrics.word_error_rate is not None:
                print(f"   Word Error Rate: {result.quality_metrics.word_error_rate:.1%}")
            
            if result.quality_metrics.domain_accuracy:
                domain_acc = result.quality_metrics.domain_accuracy.accuracy_score
                print(f"   E-commerce Terms: {domain_acc:.1%}")
            
            if result.quality_metrics.improvement_suggestions:
                print(f"   Suggestions: {', '.join(result.quality_metrics.improvement_suggestions[:2])}")
        
        # Ensemble информация
        if result.provider_metadata and 'ensemble_size' in result.provider_metadata:
            print(f"\\n🤖 ENSEMBLE INFO:")
            print(f"   Models Used: {result.provider_metadata['ensemble_size']}")
            print(f"   Consensus Method: Weighted Voting")
        
        # Текст транскрипции
        print(f"\\n💬 TRANSCRIPTION ({len(result.text.split())} words):")
        print("-" * 80)
        print(result.text)
        print("-" * 80)
        
        # Статус
        status_emoji = "✅" if result.status != "failed" else "❌"
        print(f"\\n{status_emoji} Status: {result.status}")
        
        if result.error_message:
            print(f"⚠️ Error: {result.error_message}")
        
        print("=" * 80 + "\\n")
    
    async def _save_result(self, result: TranscriptionResult, output_file: str, format_type: str):
        """Сохранение результата в файл"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type == "json":
                # JSON формат с полными метаданными
                data = {
                    "text": result.text,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "model_used": result.model_used,
                    "language_detected": result.language_detected,
                    "timestamp": datetime.now().isoformat(),
                    "quality_metrics": result.quality_metrics.to_dict() if result.quality_metrics else None,
                    "provider_metadata": result.provider_metadata
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
            else:
                # Текстовый формат
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.text)
            
            self.logger.info(f"💾 Result saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
    
    async def show_status(self):
        """Показать статус системы"""
        if not self.transcriber:
            print("❌ Enhanced Transcriber not initialized")
            return
        
        system_status = self.transcriber.get_system_status()
        quality_stats = self.transcriber.get_quality_stats()
        
        print("\\n" + "="*60)
        print("📊 ENHANCED TRANSCRIBER STATUS")
        print("="*60)
        
        print("\\n🔧 SYSTEM STATUS:")
        print(f"   Ready: {'✅' if system_status['system_ready'] else '❌'}")
        print(f"   Models: {system_status['models_count']}")
        print(f"   Ensemble: {'✅' if system_status['ensemble_available'] else '❌'}")
        print(f"   Audio Processor: {'✅' if system_status['audio_processor_available'] else '❌'}")
        print(f"   Quality Assessor: {'✅' if system_status['quality_assessor_available'] else '❌'}")
        
        print(f"\\n🎯 CONFIGURATION:")
        print(f"   Target Quality: {system_status['target_quality']:.1%}")
        print(f"   Domain: {system_status['domain']}")
        
        if quality_stats.get("total_transcriptions", 0) > 0:
            print(f"\\n📈 PERFORMANCE STATISTICS:")
            print(f"   Total Transcriptions: {quality_stats['total_transcriptions']}")
            print(f"   Success Rate: {quality_stats['success_rate']:.1%}")
            print(f"   Average Quality: {quality_stats['average_quality']:.1%}")
            print(f"   Target Achievements: {quality_stats['target_achievement_count']}")
            print(f"   Target Achievement Rate: {quality_stats['target_achievement_rate']:.1%}")
            print(f"   Quality Level: {quality_stats['quality_level']}")
        
        print("="*60 + "\\n")
    
    async def run_demo(self):
        """Демонстрация возможностей системы"""
        print("\\n🎬 ENHANCED TRANSCRIBER DEMO")
        print("="*50)
        
        # Поиск демо файлов
        demo_files = [
            "demo_audio.wav",
            "test_audio.mp3", 
            "sample_audio.wav",
            "../audio/sample.wav"
        ]
        
        demo_file = None
        for audio_file in demo_files:
            if Path(audio_file).exists():
                demo_file = audio_file
                break
        
        if demo_file:
            print(f"🎵 Using demo file: {demo_file}")
            success = await self.transcribe_single(demo_file)
            
            if success:
                print("✅ Demo completed successfully!")
            else:
                print("❌ Demo failed")
        else:
            print("⚠️ No demo audio file found.")
            print("   Place an audio file (demo_audio.wav, test_audio.mp3, etc.) in the current directory")
            print("   Supported formats: WAV, MP3, M4A, FLAC, OGG, WEBM")
        
        # Показать статистику
        await self.show_status()


async def main():
    """Главная функция CLI"""
    parser = argparse.ArgumentParser(
        description="Enhanced Transcriber - High Quality Speech Recognition (95%+)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py transcribe audio.wav
  python main.py transcribe call.mp3 --output result.txt
  python main.py batch audio_folder/ --output-dir results/
  python main.py demo
  python main.py status
        """
    )
    
    # Основные команды
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Команда transcribe
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe single audio file')
    transcribe_parser.add_argument('audio_file', help='Path to audio file')
    transcribe_parser.add_argument('--language', '-l', default='ru', help='Language (default: ru)')
    transcribe_parser.add_argument('--output', '-o', help='Output file path')
    transcribe_parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    # Команда batch
    batch_parser = subparsers.add_parser('batch', help='Batch transcribe multiple files')
    batch_parser.add_argument('input_path', help='Path to directory or file pattern')
    batch_parser.add_argument('--language', '-l', default='ru', help='Language (default: ru)')
    batch_parser.add_argument('--output-dir', '-o', help='Output directory')
    batch_parser.add_argument('--concurrent', '-c', type=int, default=2, help='Max concurrent jobs')
    
    # Команда demo
    subparsers.add_parser('demo', help='Run demonstration')
    
    # Команда status
    subparsers.add_parser('status', help='Show system status')
    
    # Общие опции
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--target-quality', type=float, default=0.95, help='Target quality (default: 0.95)')
    parser.add_argument('--domain', default='ecommerce', help='Domain specialization (default: ecommerce)')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(args.verbose)
    
    # Проверка команды
    if not args.command:
        parser.print_help()
        return
    
    # Создание CLI
    cli = TranscriptionCLI(logger)
    
    # Инициализация (кроме команды status без транскрибера)
    if args.command != 'status' or args.command == 'demo':
        success = await cli.initialize_transcriber(
            openai_api_key=args.openai_key,
            target_quality=args.target_quality,
            domain=args.domain
        )
        
        if not success:
            logger.error("❌ Failed to initialize Enhanced Transcriber")
            sys.exit(1)
    
    # Выполнение команды
    try:
        if args.command == 'transcribe':
            success = await cli.transcribe_single(
                audio_file=args.audio_file,
                language=args.language,
                output_format=args.format,
                output_file=args.output
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'batch':
            success = await cli.transcribe_batch(
                input_path=args.input_path,
                language=args.language,
                output_dir=args.output_dir,
                max_concurrent=args.concurrent
            )
            sys.exit(0 if success else 1)
            
        elif args.command == 'demo':
            await cli.run_demo()
            
        elif args.command == 'status':
            await cli.show_status()
    
    except KeyboardInterrupt:
        logger.info("\\n👋 Interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)
    
    finally:
        # Очистка ресурсов
        if cli.transcriber:
            await cli.transcriber.cleanup()


if __name__ == "__main__":
    asyncio.run(main())