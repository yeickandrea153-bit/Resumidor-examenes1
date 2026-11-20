# 📚 Resumidor de Documentos y Generador de Exámenes v2.0 (Sin IA)

Aplicación web construida en **Python + Streamlit** que analiza documentos académicos (`.docx`, `.pdf`, `.pptx`, `.txt`), genera resúmenes estructurados proporcionales y crea bancos de preguntas de examen **sin depender de APIs externas**.

## ✨ Novedades en v2.0
- **Reconstrucción de párrafos en PDF:** Une renglones cortados y descarta fragmentos incompletos (se acabaron las frases cortadas a la mitad).
- **Eliminación automática de cabeceras, pies y marcas de agua:** Detecta líneas repetidas en el 35%+ de las páginas (como el nombre de tu institución o números de página) y las borra antes de resumir.
- **Filtro de Índices y Bibliografías:** Excluye automáticamente tablas de contenido (`.... 12`), URLs/DOIs y secciones de *Referencias / Bibliografía*.
- **Soporte para PDFs a dos columnas:** Ordena los bloques de texto respetando la lectura columna por columna.
- **Resumen proporcional por capítulos (RAM-safe):** Divide libros extensos por capítulos o bloques de hasta 220 oraciones para evitar caídas de memoria en Streamlit Cloud.
- **Viñetas en PowerPoint y filtro de calidad OCR:** Normaliza listas sin punto final en `.pptx` y descarta ruido visual de Tesseract OCR.
- **`requirements.txt` listo para Streamlit Cloud:** Incluye el enlace directo del modelo `es_core_news_sm-3.8.0` de spaCy para evitar errores de permisos.

## 🚀 Despliegue en Streamlit Cloud
1. Sube estos archivos a tu repositorio de GitHub (rama `main`).
2. En [share.streamlit.io](https://share.streamlit.io/), selecciona tu repositorio, rama `main` y archivo principal `app.py`.
