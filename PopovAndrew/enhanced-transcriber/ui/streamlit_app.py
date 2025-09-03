"""
Streamlit интерфейс для Enhanced Transcriber
Веб-приложение для загрузки и транскрипции аудиофайлов
"""

import streamlit as st
import asyncio
import os
from pathlib import Path
import time
import librosa

# Импорт нашей системы
import sys
sys.path.append(".")
from enhanced_transcriber import EnhancedTranscriber

st.set_page_config(
    page_title="Enhanced Transcriber",
    page_icon="🎙️",
    layout="wide"
)

@st.cache_resource
def load_transcriber():
    """Загрузка и кэширование системы транскрипции"""
    system = EnhancedTranscriber(
        openai_api_key=None,
        target_quality=0.98,
        enable_audio_enhancement=True,
        enable_quality_assessment=True,
        domain="ecommerce"
    )
    # Синхронная инициализация для Streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(system.initialize())
    return system

def format_with_roles(text, timestamps=None):
    """Форматирование с ролями K:/M:"""
    lines = text.split('.')
    formatted = []
    
    for i, line in enumerate(lines):
        if line.strip():
            # Простая эвристика: четные - клиент, нечетные - менеджер
            role = "K" if i % 2 == 0 else "M"
            timestamp = f"[{i*10:02d}:{(i*10)%60:02d}]" if timestamps else ""
            formatted.append(f"{role}: {timestamp} {line.strip()}")
    
    return "\n".join(formatted)

def main():
    st.title("🎙️ Enhanced Transcriber")
    st.markdown("**Система максимального качества: T-one + Whisper Large-v3 + GPU**")
    
    # Загрузка системы
    with st.spinner("Загрузка моделей ИИ..."):
        system = load_transcriber()
    
    st.success(f"Система готова! Модели: {len(system.models)}")
    
    # Боковая панель с настройками
    st.sidebar.header("Настройки")
    language = st.sidebar.selectbox("Язык", ["ru", "en", "auto"], index=0)
    output_format = st.sidebar.radio("Формат вывода", ["Простой текст", "С ролями K:/M:", "С временными метками"])
    confidence_threshold = st.sidebar.slider("Минимальная уверенность", 0.0, 1.0, 0.5)
    
    # Загрузка файла
    st.header("📁 Загрузка аудиофайла")
    uploaded_file = st.file_uploader(
        "Выберите аудиофайл", 
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        help="Поддерживаемые форматы: WAV, MP3, M4A, FLAC, OGG"
    )
    
    if uploaded_file is not None:
        # Информация о файле
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Размер файла", f"{uploaded_file.size / (1024*1024):.1f} MB")
        
        # Сохранение временного файла
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            duration = librosa.get_duration(path=temp_path)
            with col2:
                st.metric("Длительность", f"{duration:.1f}s")
            with col3:
                estimated_time = duration * 0.3  # примерная оценка
                st.metric("~Время обработки", f"{estimated_time:.1f}s")
        except:
            pass
        
        # Кнопка транскрипции
        if st.button("🚀 Запустить транскрипцию", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            try:
                status_text.text("Обработка аудио...")
                progress_bar.progress(25)
                
                # Запуск транскрипции
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    system.transcribe(temp_path, language=language)
                )
                
                progress_bar.progress(100)
                processing_time = time.time() - start_time
                
                if result and result.text and result.confidence >= confidence_threshold:
                    status_text.text("Транскрипция завершена успешно!")
                    
                    # Результаты
                    st.header("📊 Результаты")
                    
                    # Метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Уверенность", f"{result.confidence:.1%}")
                    with col2:
                        st.metric("Модель", result.model_used)
                    with col3:
                        st.metric("Время обработки", f"{result.processing_time:.1f}s")
                    with col4:
                        st.metric("Количество слов", result.word_count)
                    
                    # Текст транскрипции
                    st.header("📝 Транскрипция")
                    
                    if output_format == "Простой текст":
                        formatted_text = result.text
                    elif output_format == "С ролями K:/M:":
                        formatted_text = format_with_roles(result.text)
                    else:  # С временными метками
                        formatted_text = format_with_roles(result.text, timestamps=True)
                    
                    st.text_area("Результат", formatted_text, height=300)
                    
                    # Кнопка скачивания
                    st.download_button(
                        label="💾 Скачать транскрипцию",
                        data=formatted_text,
                        file_name=f"transcript_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                    
                else:
                    st.error(f"Транскрипция неудачная или низкая уверенность ({result.confidence:.1%})")
                    
            except Exception as e:
                st.error(f"Ошибка транскрипции: {e}")
            
            finally:
                # Очистка временного файла
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Информация о системе
    st.sidebar.header("ℹ️ О системе")
    st.sidebar.info(
        "Enhanced Transcriber использует:\n"
        "• T-one ASR (русский язык)\n" 
        "• Whisper Large-v3 (универсальный)\n"
        "• GPU ускорение Tesla P100\n"
        "• Ансамблевое объединение результатов"
    )

if __name__ == "__main__":
    main()
