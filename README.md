**CS_6001_P2_Questions_For_Export.xlsx**: Excel sheet ready for exporting as a CSV file

**history_qa_csv.csv**: Exported CSV file for model inferencing. It uses "|" (pipe symbol) as column separator

**Ollama Setup.txt**: How to setup Ollama service on Mill cluster. This step is **required** for LLM inferencing.

**history_qa_llm_inferencing.py**: Python inferencing code for history MCQ answering

How to export Excel sheet to CSV with custom deliminator (such as "|"): https://stackoverflow.com/a/47155279

LLM models supported by Ollama: https://ollama.com/library

Ollama LLM model name is formatted as **model name:model size**. For example, the correct name for Qwen3 model with 30-billion parameters is **qwen3:30b**
