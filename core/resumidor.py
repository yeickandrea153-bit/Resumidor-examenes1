"""Motor de resumen extractivo híbrido (TextRank + Heurísticas estructurales + MMR) y Glosario."""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.extractor import ExtractionResult
from core.nlp_utils import (
    SPANISH_STOPWORDS,
    normalize_word,
    split_sentences,
    tokenize_words,
)

# Marcadores discursivos que suelen indicar definiciones, conclusiones o ideas clave
DISCOURSE_BOOSTERS = [
    r"\bse define como\b",
    r"\bconsiste en\b",
    r"\ben conclusi[oó]n\b",
    r"\ben resumen\b",
    r"\bpor lo tanto\b",
    r"\bes fundamental\b",
    r"\bes importante\b",
    r"\bse caracteriza por\b",
    r"\bsu objetivo\b",
    r"\btiene como finalidad\b",
    r"\bse clasifica(n)? en\b",
    r"\bse divide(n)? en\b",
    r"\bpermite(n)?\b",
    r"\bventaja(s)?\b",
    r"\bdiferencia(s)?\b",
    r"\bprincipio(s)?\b",
    r"\bconcepto\b",
]


@dataclass
class ScoredSentence:
    index: int
    section_title: str
    text: str
    score: float


@dataclass
class SummaryResult:
    selected_sentences: List[ScoredSentence] = field(default_factory=list)
    by_section: Dict[str, List[str]] = field(default_factory=dict)
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    glossary: List[Tuple[str, str]] = field(default_factory=list)  # (concepto, oracion_definicion)
    total_sentences: int = 0
    total_words: int = 0


def _pagerank(sim_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Implementación limpia de PageRank/TextRank sobre matriz de similitud."""
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Quitar autosimilitud en la diagonal
    mat = sim_matrix.copy()
    np.fill_diagonal(mat, 0.0)

    row_sums = mat.sum(axis=1, keepdims=True)
    # Evitar división por cero en oraciones aisladas
    transition = np.divide(mat, row_sums, out=np.full_like(mat, 1.0 / n), where=row_sums != 0)

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1.0 - damping) / n + damping * transition.T.dot(scores)
        if np.linalg.norm(new_scores - scores, 1) < tol:
            scores = new_scores
            break
        scores = new_scores

    # Normalizar a [0, 1]
    max_s = scores.max()
    if max_s > 0:
        scores = scores / max_s
    return scores


def _heuristic_score(
    sentence: str,
    pos_in_section: int,
    section_len: int,
    heading_tokens: set,
) -> float:
    """Calcula puntaje heurístico combinando posición, coincidencia con títulos y marcadores académicos."""
    score = 0.0

    # 1. Posición en la sección (primera y última oración suelen sintetizar)
    if pos_in_section == 0:
        score += 0.30
    elif pos_in_section == section_len - 1 and section_len > 2:
        score += 0.15

    # 2. Coincidencia con palabras de títulos/encabezados
    if heading_tokens:
        sent_tokens = {normalize_word(t) for t in tokenize_words(sentence)}
        overlap = len(sent_tokens & heading_tokens)
        if overlap > 0:
            score += min(0.35, 0.12 * overlap)

    # 3. Marcadores discursivos académicos
    lower = sentence.lower()
    marker_hits = sum(1 for pat in DISCOURSE_BOOSTERS if re.search(pat, lower))
    if marker_hits > 0:
        score += min(0.35, 0.18 * marker_hits)

    # 4. Penalizar oraciones demasiado cortas o excesivamente largas (listas crudas)
    word_count = len(sentence.split())
    if word_count < 8:
        score -= 0.25
    elif word_count > 65:
        score -= 0.15

    return max(0.0, score)


def extract_keywords_and_glossary(
    sentences: List[Tuple[str, str]],
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    top_k: int = 12,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str]]]:
    """Extrae conceptos clave (unigramas y bigramas) y busca su mejor oración explicativa."""
    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        return [], []

    # Importancia global por término (media de TF-IDF)
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    top_indices = mean_scores.argsort()[::-1]

    keywords: List[Tuple[str, float]] = []
    seen_norm = set()

    for idx in top_indices:
        term = feature_names[idx]
        norm = normalize_word(term)
        # Evitar términos puramente numéricos o redundantes
        if len(norm) < 3 or norm.isdigit():
            continue
        if any(norm in s or s in norm for s in seen_norm):
            continue
        seen_norm.add(norm)
        keywords.append((term, float(mean_scores[idx])))
        if len(keywords) >= top_k:
            break

    # Armar mini-glosario buscando oraciones definitorias para los primeros conceptos
    glossary: List[Tuple[str, str]] = []
    used_sentences = set()
    def_pattern = re.compile(r"\b(es|son|se define|consiste|significa|representa|refiere|permite)\b", re.IGNORECASE)

    for term, _ in keywords[:8]:
        best_sent = None
        best_rank = -1
        term_norm = normalize_word(term)
        for _, s_text in sentences:
            if s_text in used_sentences:
                continue
            if term_norm in normalize_word(s_text):
                rank = 1
                if def_pattern.search(s_text):
                    rank += 2
                if 10 <= len(s_text.split()) <= 45:
                    rank += 1
                if rank > best_rank:
                    best_rank = rank
                    best_sent = s_text
        if best_sent:
            used_sentences.add(best_sent)
            glossary.append((term.capitalize(), best_sent))

    return keywords, glossary


def summarize_document(
    extraction: ExtractionResult,
    ratio: float = 0.30,
    max_sentences: int = 15,
    nlp=None,
) -> SummaryResult:
    """Genera un resumen extractivo híbrido (TextRank + Heurísticas + MMR anti-redundancia)."""
    # Recolectar oraciones con su sección y posición
    flat_sentences: List[Tuple[str, str, int, int]] = []  # (section_title, text, pos, sec_len)
    for sec in extraction.sections:
        sec_text = "\n".join(sec.paragraphs)
        sents = split_sentences(sec_text, nlp=nlp)
        for pos, s in enumerate(sents):
            flat_sentences.append((sec.title, s, pos, len(sents)))

    total_sents = len(flat_sentences)
    total_words = sum(len(s[1].split()) for s in flat_sentences)

    if total_sents == 0:
        return SummaryResult(total_sentences=0, total_words=0)

    texts = [s[1] for s in flat_sentences]
    heading_tokens = {
        normalize_word(t)
        for h in extraction.headings
        for t in tokenize_words(h)
    }

    # Vectorización TF-IDF con unigramas y bigramas
    vectorizer = TfidfVectorizer(
        stop_words=list(SPANISH_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        max_features=3000,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Si todas las palabras fueron stopwords
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(tfidf_matrix)
    tr_scores = _pagerank(sim_matrix)

    # Combinar TextRank (65%) + Heurística estructural (35%)
    combined_scores = np.zeros(total_sents)
    for i, (sec_title, s_text, pos, sec_len) in enumerate(flat_sentences):
        h_score = _heuristic_score(s_text, pos, sec_len, heading_tokens)
        combined_scores[i] = 0.65 * tr_scores[i] + 0.35 * h_score

    # Determinar número objetivo de oraciones
    target_n = max(3, min(max_sentences, int(round(total_sents * ratio))))
    target_n = min(target_n, total_sents)

    # Selección con MMR (Maximal Marginal Relevance) para evitar frases repetitivas
    selected_indices: List[int] = []
    candidates = list(range(total_sents))
    lambda_param = 0.72  # Balance entre relevancia (0.72) y diversidad (0.28)

    while len(selected_indices) < target_n and candidates:
        best_idx = None
        best_mmr = -1e9
        for c in candidates:
            relevance = combined_scores[c]
            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(sim_matrix[c, s] for s in selected_indices)
            mmr = lambda_param * relevance - (1.0 - lambda_param) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = c
        selected_indices.append(best_idx)
        candidates.remove(best_idx)

    # Reordenar cronológicamente para mantener cohesión y flujo de lectura natural
    selected_indices.sort()

    selected_sentences = [
        ScoredSentence(
            index=i,
            section_title=flat_sentences[i][0],
            text=flat_sentences[i][1],
            score=float(combined_scores[i]),
        )
        for i in selected_indices
    ]

    by_section: Dict[str, List[str]] = {}
    for item in selected_sentences:
        by_section.setdefault(item.section_title, []).append(item.text)

    keywords, glossary = extract_keywords_and_glossary(
        [(s[0], s[1]) for s in flat_sentences],
        tfidf_matrix,
        vectorizer,
    )

    return SummaryResult(
        selected_sentences=selected_sentences,
        by_section=by_section,
        keywords=keywords,
        glossary=glossary,
        total_sentences=total_sents,
        total_words=total_words,
    )
