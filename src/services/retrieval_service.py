from collections import defaultdict
from typing import Optional, List, Dict, Set, NamedTuple

from src.database.config import SessionLocal
from src.models.embedding import ProgramEmbedding
from src.models.program import Program
from src.services.embedding_service import LocalEmbeddings
from src.nlp.domain_guardrail import DOMAIN_CONFIDENCE_THRESHOLD


LOOKUP_SECTIONS: Set[str] = {"program_name", "info_general", "division"}
NARRATIVE_SECTIONS: Set[str] = {
    "perfil_ingreso", "perfil_egresado", "perfil_ocupacional",
    "diferencial", "requisitos", "descripcion",
}
TABULAR_SECTIONS: Set[str] = {"course_row",
                              "elective_row", "degree_option_row"}

ALL_SECTIONS: Set[str] = LOOKUP_SECTIONS | NARRATIVE_SECTIONS | TABULAR_SECTIONS

# ─────────────────────────────────────────────────────────────────────────────
# FIX [1/10] — PROGRAM_RESOLVE_MIN_SIM movido aquí desde retrieval_service_patch.py
#
# Centralizar los umbrales junto al servicio que los consume mejora la cohesión:
# quien toca resolve_program() ve inmediatamente los umbrales disponibles.
# rag_pipeline.py importa este dict desde aquí → no necesita retrieval_service_patch.py.
# ─────────────────────────────────────────────────────────────────────────────
PROGRAM_RESOLVE_MIN_SIM: Dict[str, float] = {
    # Bloque 4b — pendientes resueltos por embedding (el que causaba el bug "pollo asado")
    "block_4b":      0.78,
    # _try_set_active_program_from_title
    "title_select":  0.75,
    # Rutas NARRATIVA y TABULAR
    "narrative":     0.72,
    "tabular":       0.72,
    # Ruta INSCRIPCIÓN
    "inscription":   0.70,
    # DEFAULT_RAG y GENERAL_CONTROLLED (llamadas genéricas a _ensure_program)
    "default":       0.72,
}


class ResolvedProgram(NamedTuple):
    id: int
    snies: str
    program_name: str
    division: Optional[str]
    modality: Optional[str]


def _distance_threshold(
    max_distance: Optional[float],
    min_similarity: Optional[float],
) -> float:
    """
    Unifica max_distance y min_similarity en un único umbral de distancia coseno.

    Prioridad: min_similarity > max_distance > default (0.7).
    """
    if min_similarity is not None:
        try:
            s = max(0.0, min(1.0, float(min_similarity)))
            return 1.0 - s
        except (TypeError, ValueError):
            pass

    if max_distance is not None:
        try:
            return max(0.0, min(2.0, float(max_distance)))
        except (TypeError, ValueError):
            pass

    return 0.7


class RetrievalService:
    def __init__(self):
        self.embedding_model = LocalEmbeddings()

    def semantic_search(
        self,
        query: str,
        limit: int = 8,
        sections: Optional[List[str]] = None,
        max_distance: Optional[float] = 0.7,
        min_similarity: Optional[float] = None,
        program_id: Optional[int] = None,
        include_program_meta: bool = False,
    ) -> List[Dict]:
        """
        Búsqueda semántica por coseno sobre program_embeddings.

        Args:
            query: Texto de consulta.
            limit: Número máximo de resultados.
            sections: Filtrar por secciones específicas. None = todas.
            max_distance: Umbral máximo de distancia coseno (menor = más similar).
            min_similarity: Alternativa a max_distance (0–1, mayor = más similar).
                            Si se proveen ambos, min_similarity tiene prioridad.
            program_id: Filtrar por programa específico.
            include_program_meta: Agregar snies y program_name al resultado.

        Returns:
            Lista de dicts con keys: program_id, section, content, distance, similarity.

        Nota: max_distance=0.7 (similitud ≥ 0.30) es intencionalmente permisivo.
        El pipeline aplica sus propios filtros downstream (best_sim < 0.60 → LOW_CONFIDENCE).
        """
        session = SessionLocal()
        try:
            qv = self.embedding_model.embed_query(query)
            thr = _distance_threshold(
                max_distance=max_distance, min_similarity=min_similarity)

            dist_expr = ProgramEmbedding.embedding.cosine_distance(
                qv).label("distance")

            q = session.query(ProgramEmbedding, dist_expr)

            if program_id is not None:
                q = q.filter(ProgramEmbedding.program_id == int(program_id))

            if sections is None:
                allowed = list(ALL_SECTIONS)
            else:
                allowed = [s for s in sections if s in ALL_SECTIONS]
                if not allowed:
                    return []

            results = (
                q.filter(ProgramEmbedding.section.in_(allowed))
                 .filter(dist_expr < thr)
                 .order_by(dist_expr.asc())
                 .limit(int(limit))
                 .all()
            )

            if not results:
                return []

            grouped: Dict = defaultdict(list)
            for emb, dist in results:
                grouped[(emb.program_id, emb.section)
                        ].append((emb, float(dist)))

            contexts: List[Dict] = []
            for (pid, section), items in grouped.items():
                items_sorted = sorted(
                    items, key=lambda t: getattr(t[0], "chunk_index", 0))
                content = "\n".join(
                    e.content.strip() for e, _ in items_sorted if e.content)
                best_dist = min(d for _, d in items_sorted)
                contexts.append({
                    "program_id": pid,
                    "section": section,
                    "content": content,
                    "distance": best_dist,
                    "similarity": round(1.0 - best_dist, 4),
                })

            if include_program_meta:
                pids = sorted({c["program_id"] for c in contexts})
                rows = (
                    session.query(Program.id, Program.snies,
                                  Program.program_name)
                    .filter(Program.id.in_(pids))
                    .all()
                )
                meta = {
                    r.id: {"snies": str(r.snies),
                           "program_name": r.program_name}
                    for r in rows
                }
                for c in contexts:
                    c.update(meta.get(c["program_id"], {}))

            contexts.sort(key=lambda x: x.get("distance", 999.0))
            return contexts

        finally:
            session.close()

    def is_in_domain(
        self,
        query: str,
        threshold: float = DOMAIN_CONFIDENCE_THRESHOLD,
    ) -> tuple[bool, float]:
        """
        Usa el índice de embeddings como juez de dominio.

        Si la query no encuentra nada con similitud >= threshold en todo
        el índice, está fuera del dominio académico — por definición,
        porque el índice ES el dominio.

        Ventaja sobre reglas: generaliza a cualquier query fuera de dominio
        sin necesidad de listas de exclusión.

        Args:
            query:     Texto de la query del usuario.
            threshold: Similitud mínima para considerar in-domain.
                       Comparte valor con DOMAIN_CONFIDENCE_THRESHOLD de
                       domain_guardrail.py (actualmente 0.25). Ajusta ese
                       valor para cambiar el comportamiento en ambos módulos.

        Returns:
            (is_in_domain: bool, best_similarity: float)
        """
        results = self.semantic_search(
            query=query,
            limit=3,
            min_similarity=0.0,   # sin filtro — queremos el mejor resultado absoluto
            max_distance=None,
            sections=["program_name", "info_general", "perfil_egresado",
                      "perfil_ocupacional", "descripcion"],
        )

        if not results:
            return False, 0.0

        best_sim = max(r.get("similarity", 0.0) for r in results)
        return best_sim >= threshold, round(best_sim, 4)

    def resolve_program(
        self,
        query: str,
        limit: int = 3,
        max_distance: Optional[float] = None,
        # 0.72
        min_similarity: Optional[float] = PROGRAM_RESOLVE_MIN_SIM["default"],
    ) -> Optional[ResolvedProgram]:
        """
        Resuelve el programa más relevante para una query.

        El umbral por defecto (min_similarity=0.72) es seguro para uso general.
        Cada punto de llamada en rag_pipeline.py puede sobrescribirlo usando
        PROGRAM_RESOLVE_MIN_SIM[contexto] para ajustar según el riesgo.

        Ejemplo: bloque 4b usa 0.78 (más estricto), inscripción usa 0.70 (algo
        más permisivo porque el candidato ya pasó extract_program_candidate).

        Returns:
            ResolvedProgram (NamedTuple de escalares) o None si no supera el umbral.
        """
        session = SessionLocal()
        try:
            qv = self.embedding_model.embed_query(query)
            thr = _distance_threshold(
                max_distance=max_distance, min_similarity=min_similarity)

            dist_expr = ProgramEmbedding.embedding.cosine_distance(
                qv).label("distance")

            hits = (
                session.query(ProgramEmbedding, dist_expr)
                .filter(ProgramEmbedding.section.in_(list(LOOKUP_SECTIONS)))
                .filter(dist_expr < thr)
                .order_by(dist_expr.asc())
                .limit(int(limit))
                .all()
            )

            if not hits:
                return None

            best_embedding = hits[0][0]

            program = (
                session.query(Program)
                .filter(Program.id == best_embedding.program_id)
                .first()
            )

            if not program:
                return None

            return ResolvedProgram(
                id=program.id,
                snies=str(program.snies),
                program_name=program.program_name or "",
                division=program.division,
                modality=program.modality,
            )

        finally:
            session.close()
