# 📚 Resumidor de Documentos y Generador de Exámenes 

Aplicación web construida en **Python + Streamlit** que analiza documentos académicos (`.docx`, `.pdf`, `.pptx`, `.txt`), genera resúmenes estructurados y crea bancos de preguntas de examen **sin depender de APIs externas ni modelos de IA pagos**.

## ✨ ¿Cómo logra resultados cercanos a una IA?
1. **Resumen extractivo híbrido:** Combina **TextRank** (grafo de similitud TF-IDF + PageRank) con **heurísticas estructurales** (peso extra a ideas relacionadas con títulos, inicio/cierre de secciones y marcadores académicos como *"se define como"*, *"en conclusión"*) y **MMR (Maximal Marginal Relevance)** para eliminar oraciones repetitivas.
2. **Generador de exámenes multiformato:**
   - **Opción Múltiple** con distractores inteligentes sacados del mismo documento.
   - **Completar espacios (Cloze / Flashcards)**.
   - **Preguntas de Desarrollo** (detecta definiciones, funciones, clasificaciones y causas).
   - **Verdadero / Falso** (crea afirmaciones falsas plausibles sustituyendo conceptos clave).
   - **Modo Simulacro interactivo** con puntaje instantáneo.
3. **Soporte de imágenes y código (OCR):** Usa **Tesseract OCR** para extraer texto de capturas de pantalla (por ejemplo, bloques de código puestos como imagen en diapositivas o PDFs) y separa automáticamente el código de la teoría para no contaminar el resumen.

## 🚀 Despliegue en Streamlit Community Cloud (vía GitHub)
1. Crea un repositorio en GitHub y sube todos los archivos de esta carpeta.
2. Entra a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **New app**, selecciona tu repositorio, rama `main` y archivo principal `app.py`.
4. Streamlit instalará automáticamente `requirements.txt` (librerías Python) y `packages.txt` (Tesseract OCR en español).

## 💻 Ejecución local
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download es_core_news_sm
streamlit run app.py
```

*(Opcional en Windows para OCR local: instala [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) y agrégalo al PATH).*
