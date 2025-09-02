#!/bin/bash
# Запуск Enhanced Transcriber в Streamlit
cd /path/to/enhanced-transcriber
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
