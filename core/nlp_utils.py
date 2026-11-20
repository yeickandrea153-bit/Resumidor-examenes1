"""Utilidades de Procesamiento de Lenguaje Natural (NLP) en español sin APIs externas."""
import re
import unicodedata
from typing import List, Set, Tuple, Optional

# Lista amplia de palabras vacías (stopwords) en español
SPANISH_STOPWORDS: Set[str] = {
    "a", "acá", "ahí", "al", "algo", "algún", "alguna", "algunas", "alguno", "algunos",
    "allá", "allí", "ambos", "ante", "antes", "aquel", "aquella", "aquellas", "aquello",
    "aquellos", "aquí", "arriba", "así", "atrás", "aun", "aunque", "bajo", "bastante",
    "bien", "cabe", "cada", "casi", "cierto", "cierta", "ciertas", "ciertos", "como",
    "cómo", "con", "conmigo", "conseguimos", "conseguir", "consigo", "consigue",
    "consiguen", "consigues", "contigo", "contra", "cual", "cuál", "cuales", "cuáles",
    "cualquier", "cualquiera", "cuando", "cuándo", "cuanto", "cuánto", "cuanta",
    "cuánta", "cuantas", "cuántas", "cuantos", "cuántos", "de", "debajo", "debe",
    "deben", "deber", "debido", "decir", "del", "delante", "demás", "demasiado",
    "dentro", "desde", "después", "detrás", "día", "días", "dice", "dicen", "dicho",
    "diferente", "diferentes", "donde", "dónde", "dos", "durante", "e", "ejemplo",
    "el", "él", "ella", "ellas", "ello", "ellos", "embargo", "emplean", "emplear",
    "empleas", "en", "encima", "entonces", "entre", "era", "erais", "éramos", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "está", "estaba",
    "estabais", "estábamos", "estaban", "estabas", "estad", "estada", "estadas",
    "estado", "estados", "estamos", "están", "estando", "estar", "estará", "estarán",
    "estarás", "estaré", "estaréis", "estaremos", "estaría", "estarían", "estas",
    "estás", "este", "esto", "estos", "estoy", "estuve", "estuviera", "estuvieron",
    "estuvo", "etc", "fin", "fue", "fueron", "fui", "fuimos", "gran", "grande", "grandes",
    "ha", "había", "habían", "haber", "habrá", "hace", "hacen", "hacer", "hacia",
    "haciendo", "han", "hasta", "hay", "haya", "he", "hecho", "hemos", "hicieron",
    "hizo", "hoy", "igual", "incluso", "ir", "la", "las", "le", "les", "lo", "los",
    "luego", "manera", "más", "mayor", "me", "mediante", "mejor", "menos", "mi", "mí",
    "mía", "mías", "mientras", "mío", "míos", "mis", "misma", "mismas", "mismo",
    "mismos", "modo", "mucha", "muchas", "mucho", "muchos", "muy", "nada", "nadie",
    "ni", "ningún", "ninguna", "ningunas", "ninguno", "ningunos", "no", "nos",
    "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva",
    "nuevas", "nuevo", "nuevos", "nunca", "o", "os", "otra", "otras", "otro", "otros",
    "para", "parece", "parte", "pero", "poca", "pocas", "poco", "pocos", "podemos",
    "poder", "podría", "podrían", "poner", "por", "porque", "porqué", "posible",
    "primer", "primera", "primero", "primeros", "propia", "propias", "propio",
    "propios", "puede", "pueden", "puedo", "pues", "que", "qué", "quien", "quién",
    "quienes", "quiénes", "quiere", "quizá", "quizás", "realizar", "realizó",
    "respecto", "sabe", "saben", "saber", "se", "sé", "sea", "sean", "según", "ser",
    "será", "serán", "sería", "serían", "si", "sí", "sido", "siempre", "siendo",
    "sigue", "siguiente", "sin", "sino", "sobre", "sola", "solamente", "solas", "solo",
    "sólo", "solos", "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo",
    "suyos", "tal", "tales", "también", "tampoco", "tan", "tanta", "tantas", "tanto",
    "tantos", "te", "tendrá", "tendrán", "tenemos", "tener", "tenga", "tengan", "tengo",
    "tenía", "tenían", "tercer", "ti", "tiempo", "tiene", "tienen", "toda", "todas",
    "todavía", "todo", "todos", "total", "tras", "través", "tres", "tu", "tú", "tus",
    "tuyo", "último", "un", "una", "unas", "uno", "unos", "usa", "usan", "usar", "uso",
    "usted", "ustedes", "va", "vais", "valor", "vamos", "van", "varias", "varios",
    "vaya", "veces", "ver", "vez", "vosotras", "vosotros", "vuestra", "vuestras",
    "vuestro", "vuestros", "y", "ya", "yo"
}

# Abreviaturas comunes que no deben cortar una oración
ABBREVIATIONS = (
    r"Sr|Sra|Srta|Dr|Dra|Ing|Lic|Mg|Prof|Fig|Cap|Art|Pág|Pag|Vol|Núm|Num|No|Ej|Etc|vs|p|pp|ed|cfr"
)


def load_spacy_model():
    """Intenta cargar el modelo español de spaCy; si no está, intenta descargarlo o retorna None."""
    try:
        import spacy
        try:
            return spacy.load("es_core_news_sm")
        except OSError:
            try:
                from spacy.cli import download
                download("es_core_news_sm")
                return spacy.load("es_core_news_sm")
            except Exception:
                return None
    except ImportError:
        return None


def clean_text(text: str) -> str:
    """Limpia ruido típico de PDFs/Word/OCR manteniendo estructura de párrafos."""
    if not text:
        return ""
    # Unir palabras cortadas con guion al final de línea (común en PDFs)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Normalizar saltos de línea múltiples
    text = re.sub(r"\r\n?", "\n", text)
    # Quitar caracteres de control invisibles
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalizar espacios horizontales repetidos
    text = re.sub(r"[ \t]+", " ", text)
    # Normalizar líneas vacías
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_word(word: str) -> str:
    """Minúsculas y sin tildes para comparaciones robustas."""
    w = word.lower().strip()
    w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
    return w


def tokenize_words(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extrae tokens alfabéticos limpios."""
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}", text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in SPANISH_STOPWORDS]
    return tokens


def split_sentences(text: str, nlp=None) -> List[str]:
    """Divide un texto en oraciones limpias usando spaCy si existe, o un segmentador regex robusto."""
    text = clean_text(text)
    if not text:
        return []

    if nlp is not None:
        # Limitar longitud por seguridad de memoria en documentos enormes
        sentences = []
        for paragraph in text.split("\n"):
            p = paragraph.strip()
            if not p:
                continue
            doc = nlp(p)
            for sent in doc.sents:
                s = sent.text.strip()
                if len(s) >= 20:
                    sentences.append(s)
        return sentences

    # Fallback sin spaCy: proteger abreviaturas y decimales antes de dividir
    sentences = []
    abbr_pattern = re.compile(rf"\b({ABBREVIATIONS})\.", re.IGNORECASE)
    for paragraph in text.split("\n"):
        p = paragraph.strip()
        if not p:
            continue
        protected = abbr_pattern.sub(r"\1<DOT>", p)
        protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9])", protected)
        for part in parts:
            s = part.replace("<DOT>", ".").strip()
            if len(s) >= 20:
                sentences.append(s)
    return sentences


def looks_like_code(text: str) -> bool:
    """Heurística para distinguir bloques de código (de OCR o texto) de prosa teórica."""
    if not text or len(text.strip()) < 15:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    code_patterns = [
        r"\b(def|class|import|from|return|public|private|static|void|int|float|string|var|let|const|function|SELECT|INSERT|UPDATE|DELETE|CREATE TABLE|WHERE|for|while|if|else|elif|switch|case|try|catch|print|console\.log|System\.out|echo|printf)\b",
        r"[{};\[\]]",
        r"==|!=|<=|>=|=>|->|\+=|-=",
        r"//.*$|/\*|\*/|#include|#!/",
        r"\w+\s*\([^)]*\)\s*[:{]?",
    ]

    code_line_hits = 0
    for line in lines:
        hits = sum(1 for pat in code_patterns if re.search(pat, line))
        # Muchos símbolos típicos de programación en relación con el largo de la línea
        symbol_ratio = len(re.findall(r"[{}()\[\];=<>+\-*/_\\|&]", line)) / max(len(line), 1)
        if hits >= 2 or symbol_ratio > 0.14:
            code_line_hits += 1

    return (code_line_hits / len(lines)) >= 0.35
