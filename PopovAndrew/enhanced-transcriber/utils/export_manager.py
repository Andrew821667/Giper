"""
Модуль экспорта результатов транскрипции в различных форматах
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import csv

class TranscriptionExporter:
    """Экспорт результатов транскрипции в разные форматы"""
    
    def __init__(self, output_dir: str = "data/transcripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_txt_with_roles(self, result, audio_filename: str, speaker_segments=None) -> str:
        """Экспорт в TXT формат с ролями K:/M:"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Path(audio_filename).stem}_{timestamp}.txt"
        output_path = self.output_dir / filename
        
        # Простое разделение на роли (улучшить когда добавим speaker diarization)
        lines = result.text.split('.')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            if line.strip():
                role = "K" if i % 2 == 0 else "M"
                formatted_lines.append(f"{role}: {line.strip()}")
        
        # Формирование итогового файла
        content = [
            f"# Транскрипция: {Path(audio_filename).name}",
            f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Модель: {result.model_used}",
            f"# Уверенность: {result.confidence:.1%}",
            f"# Время обработки: {result.processing_time:.1f}s",
            f"# Количество слов: {result.word_count}",
            "",
            "\n".join(formatted_lines)
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
        
        return str(output_path)
    
    def export_to_json(self, result, audio_filename: str, metadata: Dict = None) -> str:
        """Экспорт в JSON с полными метаданными"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Path(audio_filename).stem}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        export_data = {
            "source_file": Path(audio_filename).name,
            "transcription": {
                "text": result.text,
                "confidence": result.confidence,
                "word_count": result.word_count,
                "language_detected": getattr(result, 'language_detected', 'ru')
            },
            "model_info": {
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status)
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "quality_metrics": getattr(result, 'quality_metrics', None),
                "provider_metadata": getattr(result, 'provider_metadata', {}),
                **(metadata or {})
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
    
    def export_to_csv(self, results_batch: List[Dict], batch_name: str = "batch") -> str:
        """Экспорт пакета результатов в CSV для анализа"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_name}_{timestamp}.csv"
        output_path = self.output_dir / filename
        
        fieldnames = [
            'filename', 'model_used', 'confidence', 'processing_time', 
            'word_count', 'text_preview', 'status', 'export_date'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in results_batch:
                writer.writerow({
                    'filename': item.get('filename', 'unknown'),
                    'model_used': item.get('model_used', 'unknown'),
                    'confidence': item.get('confidence', 0),
                    'processing_time': item.get('processing_time', 0),
                    'word_count': item.get('word_count', 0),
                    'text_preview': item.get('text', '')[:100],
                    'status': item.get('status', 'unknown'),
                    'export_date': datetime.now().isoformat()
                })
        
        return str(output_path)
    
    def create_summary_report(self, results_batch: List[Dict]) -> str:
        """Создание сводного отчета по пакету обработки"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_report_{timestamp}.txt"
        output_path = self.output_dir / filename
        
        total_files = len(results_batch)
        successful = len([r for r in results_batch if r.get('status') == 'completed'])
        avg_confidence = sum(r.get('confidence', 0) for r in results_batch) / max(total_files, 1)
        total_processing_time = sum(r.get('processing_time', 0) for r in results_batch)
        
        report_lines = [
            f"СВОДНЫЙ ОТЧЕТ ПО ТРАНСКРИПЦИИ",
            f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"СТАТИСТИКА:",
            f"  Всего файлов: {total_files}",
            f"  Успешно обработано: {successful}",
            f"  Процент успеха: {(successful/max(total_files,1)*100):.1f}%",
            f"  Средняя уверенность: {avg_confidence:.1%}",
            f"  Общее время обработки: {total_processing_time:.1f}s",
            f"",
            f"ДЕТАЛИ ПО ФАЙЛАМ:"
        ]
        
        for result in results_batch:
            report_lines.extend([
                f"  {result.get('filename', 'unknown')}:",
                f"    Статус: {result.get('status', 'unknown')}",
                f"    Уверенность: {result.get('confidence', 0):.1%}",
                f"    Время: {result.get('processing_time', 0):.1f}s",
                f""
            ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        return str(output_path)

# Функция для интеграции с основной системой
async def export_transcription_result(system, audio_file, formats=['txt', 'json']):
    """Транскрипция файла с экспортом в указанных форматах"""
    exporter = TranscriptionExporter()
    result = await system.transcribe(audio_file, language='ru')
    
    exported_files = []
    
    if result and result.text:
        if 'txt' in formats:
            txt_file = exporter.export_to_txt_with_roles(result, audio_file)
            exported_files.append(txt_file)
            
        if 'json' in formats:
            json_file = exporter.export_to_json(result, audio_file)
            exported_files.append(json_file)
    
    return result, exported_files
