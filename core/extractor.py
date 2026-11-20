"""Extractor unificado para DOCX, PDF y PPTX con soporte de OCR para imágenes y separación de código."""
import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.nlp_utils import clean_text, looks_like_code


@dataclass
class ExtractedSection:
    title: str
    paragraphs: List[str] = field(default_factory=list)


@dataclass
class ExtractedImageText:
    source_location: str
    text: str
    is_code: bool


@dataclass
class ExtractionResult:
    filename: str
    file_type: str
    sections: List[ExtractedSection] = field(default_factory=list)
    code_blocks: List[Tuple[str, str]] = field(default_factory=list)  # (ubicacion, codigo)
    image_ocr_results: List[ExtractedImageText] = field(default_factory=list)
    total_images_found: int = 0
    ocr_available: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def full_theory_text(self) -> str:
        """Devuelve todo el texto teórico limpio (excluyendo bloques de código puro)."""
        chunks = []
        for sec in self.sections:
            for p in sec.paragraphs:
                chunks.append(p)
        return "\n\n".join(chunks)

    @property
    def headings(self) -> List[str]:
        return [s.title for s in self.sections if s.title and s.title != "Contenido General"]


def check_ocr_available() -> bool:
    """Verifica si pytesseract y el binario de Tesseract están disponibles en el sistema."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def run_ocr_on_bytes(image_bytes: bytes, lang: str = "spa+eng") -> str:
    """Ejecuta OCR sobre los bytes de una imagen aplicando preprocesamiento básico."""
    try:
        from PIL import Image, ImageOps, ImageFilter
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        # Ignorar iconos diminutos o líneas decorativas
        if img.width < 60 or img.height < 30:
            return ""

        # Preprocesamiento suave para mejorar lectura de capturas y diapositivas
        gray = ImageOps.grayscale(img)
        if gray.width < 800:
            scale = 2
            gray = gray.resize((gray.width * scale, gray.height * scale))
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.SHARPEN)

        try:
            text = pytesseract.image_to_string(gray, lang=lang)
        except Exception:
            # Si falla spa+eng (p. ej. falta idioma spa), reintentar con idioma por defecto
            text = pytesseract.image_to_string(gray)
        return clean_text(text)
    except Exception:
        return ""


def _add_paragraph_or_code(
    text: str,
    current_section: ExtractedSection,
    code_blocks: List[Tuple[str, str]],
    location: str,
):
    cleaned = clean_text(text)
    if not cleaned:
        return
    if looks_like_code(cleaned):
        code_blocks.append((location, cleaned))
    else:
        current_section.paragraphs.append(cleaned)


def extract_from_docx(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    import docx

    result = ExtractionResult(
        filename=filename,
        file_type="DOCX",
        ocr_available=check_ocr_available(),
    )
    document = docx.Document(io.BytesIO(file_bytes))
    current_section = ExtractedSection(title="Contenido General")
    result.sections.append(current_section)

    for idx, para in enumerate(document.paragraphs):
        txt = para.text.strip()
        if not txt:
            continue
        style_name = (para.style.name or "").lower() if para.style else ""
        if "heading" in style_name or "título" in style_name or "titulo" in style_name or style_name == "title":
            # Nueva sección lógica
            if current_section.paragraphs or current_section.title != "Contenido General":
                current_section = ExtractedSection(title=txt)
                result.sections.append(current_section)
            else:
                current_section.title = txt
        else:
            _add_paragraph_or_code(txt, current_section, result.code_blocks, f"Párrafo {idx + 1}")

    # Extraer texto útil de tablas
    for t_idx, table in enumerate(document.tables):
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells if c.text.strip()]
            if cells:
                rows_text.append(" | ".join(cells))
        if rows_text:
            table_summary = ". ".join(rows_text)
            current_section.paragraphs.append(table_summary)

    # Extraer imágenes incrustadas en el paquete DOCX
    image_rels = [
        rel for rel in document.part.rels.values()
        if "image" in rel.target_ref
    ]
    result.total_images_found = len(image_rels)

    if use_ocr and image_rels:
        if not result.ocr_available:
            result.warnings.append(
                f"Se detectaron {len(image_rels)} imagen(es), pero Tesseract OCR no está instalado en este entorno."
            )
        else:
            for i, rel in enumerate(image_rels, start=1):
                try:
                    img_bytes = rel.target_part.blob
                    ocr_text = run_ocr_on_bytes(img_bytes)
                    if ocr_text and len(ocr_text) >= 15:
                        is_code = looks_like_code(ocr_text)
                        loc = f"Imagen {i} (Word)"
                        result.image_ocr_results.append(ExtractedImageText(loc, ocr_text, is_code))
                        if is_code:
                            result.code_blocks.append((loc, ocr_text))
                        else:
                            current_section.paragraphs.append(ocr_text)
                except Exception:
                    continue

    return result


def extract_from_pdf(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    import fitz  # PyMuPDF

    result = ExtractionResult(
        filename=filename,
        file_type="PDF",
        ocr_available=check_ocr_available(),
    )
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    current_section = ExtractedSection(title="Contenido General")
    result.sections.append(current_section)

    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"Página {page_num + 1}"
        page_text = clean_text(page.get_text("text"))

        # Si la página casi no tiene texto, podría ser escaneada
        if len(page_text) < 30 and use_ocr and result.ocr_available:
            pix = page.get_pixmap(dpi=200)
            ocr_page_text = run_ocr_on_bytes(pix.tobytes("png"))
            if ocr_page_text:
                is_code = looks_like_code(ocr_page_text)
                result.image_ocr_results.append(
                    ExtractedImageText(f"{page_label} (Escaneada)", ocr_page_text, is_code)
                )
                if is_code:
                    result.code_blocks.append((f"{page_label} (Escaneada)", ocr_page_text))
                else:
                    current_section.paragraphs.append(ocr_page_text)
        else:
            # Detectar posibles subtítulos por línea corta en mayúsculas o numerada
            for block in page_text.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                if len(block) < 80 and "\n" not in block and (block.isupper() or block[0].isdigit() and "." in block[:4]):
                    current_section = ExtractedSection(title=block.title() if block.isupper() else block)
                    result.sections.append(current_section)
                else:
                    _add_paragraph_or_code(block, current_section, result.code_blocks, page_label)

        # Extraer imágenes incrustadas en la página
        images = page.get_images(full=True)
        for img_idx, img_info in enumerate(images, start=1):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            result.total_images_found += 1

            if use_ocr and result.ocr_available:
                try:
                    base_img = doc.extract_image(xref)
                    ocr_text = run_ocr_on_bytes(base_img["image"])
                    if ocr_text and len(ocr_text) >= 15:
                        is_code = looks_like_code(ocr_text)
                        loc = f"{page_label} - Imagen {img_idx}"
                        result.image_ocr_results.append(ExtractedImageText(loc, ocr_text, is_code))
                        if is_code:
                            result.code_blocks.append((loc, ocr_text))
                        else:
                            current_section.paragraphs.append(ocr_text)
                except Exception:
                    continue

    if use_ocr and result.total_images_found > 0 and not result.ocr_available:
        result.warnings.append(
            f"Se detectaron {result.total_images_found} imagen(es) en el PDF, pero Tesseract OCR no está instalado."
        )

    doc.close()
    return result


def extract_from_pptx(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    result = ExtractionResult(
        filename=filename,
        file_type="PPTX",
        ocr_available=check_ocr_available(),
    )
    prs = Presentation(io.BytesIO(file_bytes))

    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_label = f"Diapositiva {slide_idx}"
        slide_title = slide_label
        if slide.shapes.title and slide.shapes.title.text.strip():
            slide_title = f"{slide_label}: {slide.shapes.title.text.strip()}"

        section = ExtractedSection(title=slide_title)
        result.sections.append(section)

        img_counter = 0
        for shape in slide.shapes:
            # Saltar el título ya registrado
            if shape == slide.shapes.title:
                continue

            if shape.has_text_frame:
                text = "\n".join(p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip())
                if text:
                    _add_paragraph_or_code(text, section, result.code_blocks, slide_label)

            if shape.has_table:
                rows_text = []
                for row in shape.table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells if c.text.strip()]
                    if cells:
                        rows_text.append(" | ".join(cells))
                if rows_text:
                    section.paragraphs.append(". ".join(rows_text))

            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_counter += 1
                result.total_images_found += 1
                if use_ocr and result.ocr_available:
                    try:
                        img_bytes = shape.image.blob
                        ocr_text = run_ocr_on_bytes(img_bytes)
                        if ocr_text and len(ocr_text) >= 15:
                            is_code = looks_like_code(ocr_text)
                            loc = f"{slide_label} - Imagen {img_counter}"
                            result.image_ocr_results.append(ExtractedImageText(loc, ocr_text, is_code))
                            if is_code:
                                result.code_blocks.append((loc, ocr_text))
                            else:
                                section.paragraphs.append(ocr_text)
                    except Exception:
                        continue

        # Notas del presentador (muy valiosas en clases universitarias)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                _add_paragraph_or_code(notes, section, result.code_blocks, f"{slide_label} (Notas)")

    if use_ocr and result.total_images_found > 0 and not result.ocr_available:
        result.warnings.append(
            f"Se detectaron {result.total_images_found} imagen(es) en el PPTX, pero Tesseract OCR no está instalado."
        )

    return result


def extract_from_raw_text(raw_text: str) -> ExtractionResult:
    result = ExtractionResult(filename="Texto pegado", file_type="TXT", ocr_available=False)
    section = ExtractedSection(title="Texto Ingresado")
    result.sections.append(section)
    for block in clean_text(raw_text).split("\n\n"):
        _add_paragraph_or_code(block, section, result.code_blocks, "Texto pegado")
    return result


def extract_document(file_bytes: bytes, filename: str, use_ocr: bool = True) -> ExtractionResult:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".docx":
        return extract_from_docx(file_bytes, filename, use_ocr=use_ocr)
    elif ext == ".pdf":
        return extract_from_pdf(file_bytes, filename, use_ocr=use_ocr)
    elif ext == ".pptx":
        return extract_from_pptx(file_bytes, filename, use_ocr=use_ocr)
    elif ext == ".txt":
        return extract_from_raw_text(file_bytes.decode("utf-8", errors="ignore"))
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .docx, .pdf, .pptx o .txt")
