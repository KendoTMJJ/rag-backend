from __future__ import annotations

import re
from typing import Optional, Dict, Any, List

from sqlalchemy import asc, desc, func, or_

from src.database.config import SessionLocal
from src.models.program import Program
from src.models.course import Course
from src.models.elective import Elective
from src.models.degree_option import DegreeOption
from src.models.narrative import ProgramNarrative


def _money_co(v):
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    if any(ch in s for ch in ["$", "COP", ".", ","]):
        return s
    try:
        n = int(float(s))
        return f"${n:,}".replace(",", ".")
    except Exception:
        return s


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _strip_accents(text: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


# FIX: mapa de tópicos de usuario → fragmento de división en BD.
# Permite que "programas de salud" encuentre "División de Ciencias de la Salud"
# aunque el topic no coincida literalmente con el nombre del programa.
_TOPIC_TO_DIVISION: Dict[str, str] = {
    "salud": "Ciencias de la Salud",
    "medicina": "Ciencias de la Salud",
    "enfermeria": "Ciencias de la Salud",
    "epidemiologia": "Ciencias de la Salud",
    "bioetica": "Ciencias de la Salud",
    "humanidades medicas": "Ciencias de la Salud",
    "derecho": "Ciencias Jurídicas",
    "juridic": "Ciencias Jurídicas",
    "constitucional": "Ciencias Jurídicas",
    "procesal": "Ciencias Jurídicas",
    "ingenieria": "Ingeniería y Tecnología",
    "tecnologia": "Ingeniería y Tecnología",
    "sistemas": "Ingeniería y Tecnología",
    "inteligencia artificial": "Ingeniería y Tecnología",
    "ciberseguridad": "Ingeniería y Tecnología",
    "economia": "Ciencias Económicas",
    "administracion": "Ciencias Económicas",
    "contaduria": "Ciencias Económicas",
    "finanzas": "Ciencias Económicas",
    "educacion": "Ciencias de la Educación",
    "pedagogia": "Ciencias de la Educación",
}


def _resolve_division_hint(topic: str) -> Optional[str]:
    """
    Dado un topic (sin tildes, minúsculas), retorna el fragmento de división
    para usar como filtro SQL, o None si no hay mapeo conocido.
    """
    t = _strip_accents(topic).lower().strip()
    for key, division in _TOPIC_TO_DIVISION.items():
        if key in t:
            return division
    return None


class SQLRetrievalService:
    FIELD_MAP = {
        "duration": ("duration_semesters", "Duración", lambda v: f"{v} semestres"),
        "credits": ("total_credits", "Créditos", lambda v: f"{v} créditos"),
        "cost": ("investment_per_semester", "Inversión por semestre", _money_co),
        "modality": ("modality", "Modalidad", lambda v: str(v)),
        "location": ("location", "Ubicación", lambda v: str(v)),
        "schedule": ("schedule", "Horarios", lambda v: str(v)),
        "degree": ("degree_awarded", "Título otorgado", lambda v: str(v)),
        "registry": ("qualified_registry", "Registro calificado", lambda v: str(v)),
        "year_update": ("year_update", "Año de actualización", lambda v: str(v)),
        "division": ("division", "División", lambda v: str(v)),
        "start_date": ("start_date", "Fecha de inicio", lambda v: str(v)),
        "link": ("link", "Enlace", lambda v: str(v)),
    }

    NARRATIVE_FIELD_MAP = {
        "admission_profile": ("admission_profile", "Perfil de ingreso"),
        "graduated_profile": ("graduated_profile", "Perfil de egresado"),
        "occupational_profile": ("occupational_profile", "Perfil ocupacional"),
        "differential": ("differential", "Diferencial"),
        "specific_requirements": ("specific_requirements", "Requisitos específicos"),
        "description": ("description", "Descripción del programa"),
    }

    def _get_program_by_snies(self, session, snies: str) -> Optional[Program]:
        return session.query(Program).filter(Program.snies == str(snies).strip()).first()

    def get_program_overview_context(self, snies: str) -> str:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""

            parts: List[str] = []
            info_lines: List[str] = []

            if (p.program_name or "").strip():
                info_lines.append(
                    f"Programa: {p.program_name} (SNIES {p.snies}).")
            if (p.division or "").strip():
                info_lines.append(f"División: {p.division}.")
            if (p.modality or "").strip():
                info_lines.append(f"Modalidad: {p.modality}.")
            if p.duration_semesters:
                info_lines.append(
                    f"Duración: {p.duration_semesters} semestres.")
            if p.total_credits:
                info_lines.append(f"Créditos: {p.total_credits}.")
            if (p.investment_per_semester or "").strip():
                info_lines.append(
                    f"Inversión por semestre: {p.investment_per_semester}.")
            if (p.location or "").strip():
                info_lines.append(f"Ubicación: {p.location}.")
            if (p.schedule or "").strip():
                info_lines.append(f"Horarios: {p.schedule}.")
            if (p.start_date or "").strip():
                info_lines.append(f"Fecha de inicio: {p.start_date}.")
            if (p.degree_awarded or "").strip():
                info_lines.append(f"Título otorgado: {p.degree_awarded}.")
            if (p.qualified_registry or "").strip():
                info_lines.append(
                    f"Registro calificado: {p.qualified_registry}.")
            if (p.link or "").strip():
                info_lines.append(f"Enlace: {p.link}.")

            if info_lines:
                parts.append("INFO_GENERAL:\n" +
                             "\n".join(f"- {x}" for x in info_lines))

            n = session.query(ProgramNarrative).filter(
                ProgramNarrative.program_id == p.id).first()
            if n:
                narr_blocks: List[str] = []
                if (n.description or "").strip():
                    narr_blocks.append("Descripción:\n" +
                                       n.description.strip())
                if (n.admission_profile or "").strip():
                    narr_blocks.append(
                        "Perfil de ingreso:\n" + n.admission_profile.strip())
                if (n.graduated_profile or "").strip():
                    narr_blocks.append(
                        "Perfil de egresado:\n" + n.graduated_profile.strip())
                if (n.occupational_profile or "").strip():
                    narr_blocks.append(
                        "Perfil ocupacional:\n" + n.occupational_profile.strip())
                if (n.differential or "").strip():
                    narr_blocks.append("Diferencial:\n" +
                                       n.differential.strip())
                if (n.specific_requirements or "").strip():
                    narr_blocks.append(
                        "Requisitos específicos:\n" + n.specific_requirements.strip())

                if narr_blocks:
                    parts.append("NARRATIVA:\n" + "\n\n".join(narr_blocks))

            return "\n\n".join(parts).strip()
        finally:
            session.close()

    def get_program_narrative_field(self, snies: str, field: str) -> str:
        field = (field or "").strip()
        if field not in self.NARRATIVE_FIELD_MAP:
            return ""

        attr, label = self.NARRATIVE_FIELD_MAP[field]
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""

            n = session.query(ProgramNarrative).filter(
                ProgramNarrative.program_id == p.id).first()
            if not n:
                return ""

            value = getattr(n, attr, None)
            if value is None or not str(value).strip():
                return ""

            return f"{label} — {p.program_name} (SNIES {p.snies}):\n{str(value).strip()}"
        finally:
            session.close()

    def get_inscription_info(self, snies: str) -> Dict[str, Any]:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return {}
            url = getattr(p, "link", None)
            if not url or not str(url).strip():
                return {}
            return {"programName": p.program_name, "snies": str(p.snies), "url": str(url).strip()}
        finally:
            session.close()

    def get_program_field(self, snies: str, field: str) -> str:
        field = (field or "").strip().lower()
        if field not in self.FIELD_MAP:
            return ""

        attr, label, fmt = self.FIELD_MAP[field]
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""
            value = getattr(p, attr, None)
            if value is None or str(value).strip() == "" or str(value).strip().lower() in ["n/a", "na", "none"]:
                return ""
            pretty = fmt(value)
            if not str(pretty).strip():
                return ""
            return f"{label} del programa **{p.program_name}** (SNIES {p.snies}): **{pretty}**."
        finally:
            session.close()

    def get_program_minmax(self, field: str, mode: str = "min") -> Dict[str, Any]:
        field = (field or "").strip().lower()
        mode = (mode or "min").strip().lower()

        mapping = {
            "duration": Program.duration_semesters,
            "credits": Program.total_credits,
            "cost": Program.investment_per_semester,
        }
        col = mapping.get(field)
        if col is None:
            return {}

        order_fn = asc if mode == "min" else desc
        session = SessionLocal()
        try:
            q = session.query(Program).filter(col.isnot(None))
            p = q.order_by(order_fn(col)).first()
            if not p:
                return {}

            value = getattr(p, col.key, None)
            pretty_value = _money_co(value) if field == "cost" else value

            return {
                "programName": p.program_name,
                "snies": str(p.snies),
                "field": field,
                "mode": mode,
                "value": value,
                "prettyValue": pretty_value,
            }
        finally:
            session.close()

    def get_curriculum(self, snies: str) -> str:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""

            courses = (
                session.query(Course)
                .filter(Course.program_id == p.id)
                .order_by(Course.semester, Course.course_name)
                .all()
            )
            if not courses:
                return "No hay malla curricular cargada para este programa."

            out = [f"MALLA CURRICULAR — {p.program_name} (SNIES {p.snies})"]
            current = None
            for c in courses:
                if c.semester != current:
                    current = c.semester
                    out.append(f"\nSemestre {current}:")
                out.append(f"- {c.course_name} ({c.credits} cr)")
            return "\n".join(out)
        finally:
            session.close()

    def get_curriculum_semester(self, snies: str, semester: int) -> str:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""

            rows = (
                session.query(Course)
                .filter(Course.program_id == p.id, Course.semester == int(semester))
                .order_by(Course.course_name)
                .all()
            )
            if not rows:
                return f"No hay materias cargadas para el semestre {semester}."

            out = [f"SEMESTRE {semester} — {p.program_name} (SNIES {p.snies})"]
            out += [f"- {c.course_name} ({c.credits} cr)" for c in rows]
            return "\n".join(out)
        finally:
            session.close()

    def get_electives(self, snies: str) -> str:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""

            rows = (
                session.query(Elective)
                .filter(Elective.program_id == p.id)
                .order_by(Elective.type, Elective.course_name)
                .all()
            )
            if not rows:
                return "No hay electivas cargadas para este programa."

            out = [f"ELECTIVAS — {p.program_name} (SNIES {p.snies})"]
            current = None
            for e in rows:
                if e.type != current:
                    current = e.type
                    out.append(f"\nTipo: {current or 'N/A'}")
                out.append(f"- {e.course_name}")
            return "\n".join(out)
        finally:
            session.close()

    def get_degree_options(self, snies: str) -> str:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""

            rows = (
                session.query(DegreeOption)
                .filter(DegreeOption.program_id == p.id)
                .order_by(DegreeOption.option_name)
                .all()
            )
            if not rows:
                return "No hay opciones de grado cargadas para este programa."

            out = [f"OPCIONES DE GRADO — {p.program_name} (SNIES {p.snies})"]
            out += [f"- {r.option_name}" for r in rows]
            return "\n".join(out)
        finally:
            session.close()

    @staticmethod
    def _strip_accents(text: str) -> str:
        import unicodedata
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def resolve_program_by_name(self, program_text: str) -> Dict[str, Any]:
        q = (program_text or "").strip()
        if not q:
            return {}

        q = re.sub(r'["""\'`]', "", q)
        q = re.sub(r"\s+", " ", q).strip()
        q_plain = self._strip_accents(q)

        session = SessionLocal()
        try:
            p = (
                session.query(Program)
                .filter(Program.program_name.ilike(f"%{q}%"))
                .order_by(func.length(Program.program_name).asc())
                .first()
            )
            if p:
                return {"snies": str(p.snies).strip(), "programName": p.program_name}

            if q_plain.lower() != q.lower():
                p2 = (
                    session.query(Program)
                    .filter(func.lower(Program.program_name).contains(q_plain.lower()))
                    .order_by(func.length(Program.program_name).asc())
                    .first()
                )
                if p2:
                    return {"snies": str(p2.snies).strip(), "programName": p2.program_name}

            tokens = [t for t in q_plain.lower().split(" ") if len(t) >= 4]
            if not tokens:
                return {}

            qry = session.query(Program)
            for t in tokens[:6]:
                qry = qry.filter(func.lower(Program.program_name).contains(t))

            p3 = qry.order_by(func.length(Program.program_name).asc()).first()
            if p3:
                return {"snies": str(p3.snies).strip(), "programName": p3.program_name}

            return {}
        finally:
            session.close()

    def get_program_brief_context(self, snies: str) -> str:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            if not p:
                return ""
            return (
                f"Programa: {p.program_name} (SNIES {p.snies}).\n"
                f"División: {p.division or 'N/A'}.\n"
                f"Modalidad: {p.modality or 'N/A'}.\n"
                f"Duración: {p.duration_semesters or 'N/A'} semestres.\n"
                f"Créditos: {p.total_credits or 'N/A'}.\n"
                f"Ubicación: {p.location or 'N/A'}.\n"
            )
        finally:
            session.close()

    def list_programs_filtered(
        self,
        *,
        division_like: Optional[str] = None,
        modality_like: Optional[str] = None,
        location_like: Optional[str] = None,
        name_like: Optional[str] = None,
        type_like: Optional[str] = None,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        session = SessionLocal()
        try:
            q = session.query(Program)

            dl = _norm(division_like)
            ml = _norm(modality_like)
            ll = _norm(location_like)
            nl = _norm(name_like)
            tl = _norm(type_like).lower()

            if dl:
                q = q.filter(Program.division.ilike(f"%{dl}%"))
            if ml:
                q = q.filter(Program.modality.ilike(f"%{ml}%"))
            if ll:
                q = q.filter(Program.location.ilike(f"%{ll}%"))

            if nl:
                # FIX: verificar primero si el topic mapea a una división conocida.
                # "salud" → filtra por división "Ciencias de la Salud" en lugar de
                # buscar "salud" en program_name (donde no aparece).
                division_hint = _resolve_division_hint(nl)

                if division_hint and not dl:
                    q = q.filter(Program.division.ilike(f"%{division_hint}%"))
                else:
                    tokens = [t for t in re.split(
                        r"\s+", _strip_accents(nl)) if len(t) >= 3]
                    if tokens:
                        for t in tokens[:7]:
                            q = q.filter(or_(
                                Program.program_name.ilike(f"%{t}%"),
                                Program.division.ilike(f"%{t}%")
                            ))
                    else:
                        q = q.filter(or_(
                            Program.program_name.ilike(f"%{nl}%"),
                            Program.division.ilike(f"%{nl}%")
                        ))

            if tl:
                if "doctor" in tl:
                    q = q.filter(Program.program_name.ilike("%Doctorado%"))
                elif "maestr" in tl:
                    q = q.filter(Program.program_name.ilike("%Maestr%"))
                elif "especial" in tl:
                    q = q.filter(Program.program_name.ilike("%Especializ%"))

            rows = q.order_by(Program.program_name.asc()
                              ).limit(int(limit)).all()

            return [
                {
                    "programName": r.program_name,
                    "snies": str(r.snies),
                    "division": r.division,
                    "modality": r.modality,
                    "location": r.location,
                }
                for r in rows
            ]
        finally:
            session.close()

    def _get_program_id_by_snies(self, snies: str) -> Optional[int]:
        session = SessionLocal()
        try:
            p = self._get_program_by_snies(session, snies)
            return p.id if p else None
        finally:
            session.close()

    def get_snies_for_program(self, program_id: int) -> Optional[str]:
        session = SessionLocal()
        try:
            p = session.query(Program).filter(
                Program.id == int(program_id)).first()
            return str(p.snies).strip() if p and p.snies else None
        except Exception:
            return None
        finally:
            session.close()
