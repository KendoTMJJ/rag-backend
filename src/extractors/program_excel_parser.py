import re
import pandas as pd
from typing import Dict, List, Optional, Set

from src.database.config import SessionLocal
from src.models.program import Program
from src.models.course import Course
from src.models.embedding import ProgramEmbedding
from src.models.elective import Elective
from src.models.degree_option import DegreeOption
from src.models.narrative import ProgramNarrative
from src.services.embedding_service import LocalEmbeddings


LOOKUP_SECTIONS = {"program_name", "info_general", "division"}
NARRATIVE_SECTIONS = {"perfil_ingreso", "perfil_egresado",
                      "perfil_ocupacional", "diferencial", "requisitos", "descripcion"}
TABULAR_SECTIONS = {"course_row", "elective_row", "degree_option_row"}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_sheet(xls: pd.ExcelFile, target: str) -> Optional[str]:
    t = (target or "").strip().lower()
    for s in xls.sheet_names:
        if (s or "").strip().lower() == t:
            return s
    return None


def _get_snies_from_row(row) -> str:
    sn = _get_str_any(row, ["código snies", "codigo snies",
                      "snies", "codigo_snies", "código_snies"])
    if sn:
        return _clean_snies(sn)

    for k in list(row.index):
        kk = str(k).strip().lower()
        if "snies" in kk:
            v = row.get(k, "")
            if pd.isna(v):
                continue
            return _clean_snies(str(v).strip())

    return ""


def _get_str(row, key: str, default: str = "") -> str:
    v = row.get(key, default)
    if pd.isna(v):
        return default
    return str(v).strip()


def _get_str_any(row, keys: List[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k, None)
        if v is None:
            continue
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return default


def _get_int(row, key: str, default: int = 0) -> int:
    v = row.get(key, default)
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def _clean_snies(snies: str) -> str:
    return re.sub(r"\D+", "", snies or "")


def simple_chunk(text: str, max_chars: int = 900) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += max_chars
    return chunks


class ProgramExcelParser:
    """
    Hojas:
      - INFO_GENERAL
      - MALLA_CURRICULAR
      - ELECTIVAS
      - OPCIONES_GRADO
      - NARRATIVA

    SQL = fuente de verdad

    Embeddings:
      - LOOKUP siempre: program_name, info_general, division
      - TABULAR siempre: course_row, elective_row, degree_option_row
      - NARRATIVA: perfil_ingreso, perfil_egresado, perfil_ocupacional, diferencial, requisitos

    FIX: todos los embeddings de indexado usan embed_document() (prefijo "passage:")
         en lugar de embed_query() (prefijo "query:"). El modelo multilingual-e5-base
         requiere prefijos distintos para documentos y consultas. Usar el prefijo
         incorrecto al indexar degrada la similitud coseno en búsqueda.
    """

    def __init__(self, file_path: str, narrative_embedder: Optional[LocalEmbeddings] = None):
        self.file_path = file_path
        self.embedder = LocalEmbeddings()
        self.narrative_embedder = narrative_embedder or self.embedder

    def load_and_sync(self):
        xls = pd.ExcelFile(self.file_path)

        self._sync_info_general_and_lookup(xls)
        self._sync_curriculum_sql(xls)
        self._sync_electives_sql(xls)
        self._sync_degree_options_sql(xls)
        self._sync_narrative_sql(xls)

        self._sync_tabular_embeddings(xls)
        self._sync_narrative_embeddings(xls)

        print("✅ Sincronización completa (SQL + Embeddings).")

    # ---------------------------------------------------------
    # 1) INFO_GENERAL -> SQL + LOOKUP EMBEDDINGS
    # ---------------------------------------------------------
    def _sync_info_general_and_lookup(self, xls: pd.ExcelFile):
        if "INFO_GENERAL" not in xls.sheet_names:
            print("⚠️ No existe la hoja INFO_GENERAL.")
            return

        df = _norm_cols(pd.read_excel(xls, sheet_name="INFO_GENERAL"))

        with SessionLocal() as db:
            for _, row in df.iterrows():
                snies = _clean_snies(_get_str_any(
                    row, ["código snies", "codigo snies"]))
                if not snies:
                    continue

                program_name = _get_str(row, "nombre del programa")
                division = _get_str_any(row, ["división", "division"])
                modality = _get_str(row, "modalidad")
                duration = _get_int(row, "duración (semestres)") or _get_int(
                    row, "duracion (semestres)")
                total_credits = _get_int(row, "total créditos") or _get_int(
                    row, "total creditos")
                investment = _get_str_any(
                    row, ["inversión por semestre", "inversion por semestre"])
                year_update = _get_int(row, "año actualización") or _get_int(
                    row, "ano actualizacion")
                location = _get_str_any(row, ["ubicación", "ubicacion"])
                schedule = _get_str(row, "horarios")
                start_date = _get_str(row, "fecha de inicio")
                degree_awarded = _get_str_any(
                    row, ["titulo otorgado", "título otorgado"])
                qualified_registry = _get_str(row, "registro calificado")
                link = _get_str(row, "enlace")

                program = db.query(Program).filter(
                    Program.snies == snies).first()

                if not program:
                    program = Program(
                        snies=snies,
                        program_name=program_name,
                        division=division,
                        modality=modality,
                        duration_semesters=duration,
                        total_credits=total_credits,
                        investment_per_semester=investment,
                        year_update=year_update,
                        location=location,
                        schedule=schedule,
                        start_date=start_date,
                        degree_awarded=degree_awarded,
                        qualified_registry=qualified_registry,
                        link=link,
                    )
                    db.add(program)
                    db.flush()
                    print(f"✨ Programa creado SNIES {snies}: {program_name}")
                else:
                    program.program_name = program_name or program.program_name
                    program.division = division or program.division
                    program.modality = modality or program.modality
                    program.duration_semesters = duration or program.duration_semesters
                    program.total_credits = total_credits or program.total_credits
                    program.investment_per_semester = investment or program.investment_per_semester
                    program.year_update = year_update or program.year_update
                    program.location = location or program.location
                    program.schedule = schedule or program.schedule
                    program.start_date = start_date or program.start_date
                    program.degree_awarded = degree_awarded or program.degree_awarded
                    program.qualified_registry = qualified_registry or program.qualified_registry
                    program.link = link or program.link
                    db.flush()
                    print(
                        f"✔️ Programa actualizado SNIES {snies}: {program.program_name}")

                # Limpia lookup anterior
                db.query(ProgramEmbedding).filter(
                    ProgramEmbedding.program_id == program.id,
                    ProgramEmbedding.section.in_(list(LOOKUP_SECTIONS)),
                ).delete(synchronize_session=False)

                # program_name
                name_text = f"{program.program_name} (SNIES {program.snies})"
                db.add(ProgramEmbedding(
                    program_id=program.id,
                    section="program_name",
                    content=name_text,
                    chunk_index=0,
                    # FIX: embed_document usa prefijo "passage:" (indexado)
                    embedding=self.embedder.embed_document(name_text),
                ))

                # division (lookup)
                if (program.division or "").strip():
                    div_text = f"División del programa {program.program_name} (SNIES {program.snies}): {program.division}"
                    db.add(ProgramEmbedding(
                        program_id=program.id,
                        section="division",
                        content=div_text,
                        chunk_index=0,
                        # FIX: embed_document usa prefijo "passage:" (indexado)
                        embedding=self.embedder.embed_document(div_text),
                    ))

                # info_general
                info_text = (
                    f"Programa: {program.program_name}. SNIES: {program.snies}. "
                    f"División: {program.division or 'N/A'}. "
                    f"Modalidad: {program.modality or 'N/A'}. "
                    f"Duración: {program.duration_semesters or 'N/A'} semestres. "
                    f"Créditos: {program.total_credits or 'N/A'}. "
                    f"Inversión por semestre: {program.investment_per_semester or 'N/A'}. "
                    f"Ubicación: {program.location or 'N/A'}. "
                    f"Horarios: {program.schedule or 'N/A'}. "
                    f"Fecha de inicio: {program.start_date or 'N/A'}. "
                    f"Título otorgado: {program.degree_awarded or 'N/A'}. "
                    f"Registro calificado: {program.qualified_registry or 'N/A'}. "
                    f"Enlace: {program.link or 'N/A'}."
                )
                db.add(ProgramEmbedding(
                    program_id=program.id,
                    section="info_general",
                    content=info_text,
                    chunk_index=0,
                    # FIX: embed_document usa prefijo "passage:" (indexado)
                    embedding=self.embedder.embed_document(info_text),
                ))

            db.commit()

    # ---------------------------------------------------------
    # 2) MALLA_CURRICULAR -> SQL
    # ---------------------------------------------------------
    def _sync_curriculum_sql(self, xls: pd.ExcelFile):
        if "MALLA_CURRICULAR" not in xls.sheet_names:
            print("⚠️ No existe la hoja MALLA_CURRICULAR.")
            return

        df = _norm_cols(pd.read_excel(xls, sheet_name="MALLA_CURRICULAR"))

        with SessionLocal() as db:
            cleaned: Set[int] = set()

            for _, row in df.iterrows():
                snies = _clean_snies(_get_str_any(
                    row, ["código snies", "codigo snies"]))
                if not snies:
                    continue

                program = db.query(Program).filter(
                    Program.snies == snies).first()
                if not program:
                    continue

                if program.id not in cleaned:
                    db.query(Course).filter(Course.program_id ==
                                            program.id).delete(synchronize_session=False)
                    cleaned.add(program.id)

                semester = _get_int(
                    row, "semestre #") or _get_int(row, "semestre")
                course_name = _get_str(row, "nombre de la asignatura")
                credits = _get_int(row, "créditos asignatura") or _get_int(
                    row, "creditos asignatura")

                if not course_name:
                    continue

                db.add(Course(program_id=program.id, semester=semester,
                       course_name=course_name, credits=credits))

            db.commit()
        print("📚 Malla curricular sincronizada.")

    # ---------------------------------------------------------
    # 3) ELECTIVAS -> SQL
    # ---------------------------------------------------------
    def _sync_electives_sql(self, xls: pd.ExcelFile):
        if "ELECTIVAS" not in xls.sheet_names:
            print("⚠️ No existe la hoja ELECTIVAS.")
            return

        df = _norm_cols(pd.read_excel(xls, sheet_name="ELECTIVAS"))

        with SessionLocal() as db:
            cleaned: Set[int] = set()

            for _, row in df.iterrows():
                snies = _clean_snies(_get_str_any(
                    row, ["código snies", "codigo snies"]))
                if not snies:
                    continue

                program = db.query(Program).filter(
                    Program.snies == snies).first()
                if not program:
                    continue

                if program.id not in cleaned:
                    db.query(Elective).filter(Elective.program_id ==
                                              program.id).delete(synchronize_session=False)
                    cleaned.add(program.id)

                name = _get_str(row, "nombre de la asignatura")
                tipo = _get_str(row, "tipo")
                if not name:
                    continue

                db.add(Elective(program_id=program.id,
                       course_name=name, type=tipo))

            db.commit()
        print("🧩 Electivas sincronizadas.")

    # ---------------------------------------------------------
    # 4) OPCIONES_GRADO -> SQL
    # ---------------------------------------------------------
    def _sync_degree_options_sql(self, xls: pd.ExcelFile):
        if "OPCIONES_GRADO" not in xls.sheet_names:
            print("⚠️ No existe la hoja OPCIONES_GRADO.")
            return

        df = _norm_cols(pd.read_excel(xls, sheet_name="OPCIONES_GRADO"))

        with SessionLocal() as db:
            cleaned: Set[int] = set()

            for _, row in df.iterrows():
                snies = _clean_snies(_get_str_any(
                    row, ["código snies", "codigo snies"]))
                if not snies:
                    continue

                program = db.query(Program).filter(
                    Program.snies == snies).first()
                if not program:
                    continue

                if program.id not in cleaned:
                    db.query(DegreeOption).filter(
                        DegreeOption.program_id == program.id).delete(synchronize_session=False)
                    cleaned.add(program.id)

                opt = _get_str_any(row, ["opción de grado", "opcion de grado"])
                if not opt:
                    continue

                db.add(DegreeOption(program_id=program.id, option_name=opt))

            db.commit()
        print("🎓 Opciones de grado sincronizadas.")

    # ---------------------------------------------------------
    # 4.5) NARRATIVA -> SQL (program_narratives)
    # ---------------------------------------------------------
    def _sync_narrative_sql(self, xls: pd.ExcelFile):
        sheet = _find_sheet(xls, "NARRATIVA")
        if not sheet:
            print(
                f"⚠️ No existe la hoja NARRATIVA. Hojas disponibles: {xls.sheet_names}")
            return

        df = _norm_cols(pd.read_excel(xls, sheet_name=sheet))

        total = len(df)
        skipped_no_snies = 0
        skipped_no_program = 0
        upserts = 0

        with SessionLocal() as db:
            for _, row in df.iterrows():
                snies = _get_snies_from_row(row)
                if not snies:
                    skipped_no_snies += 1
                    continue

                program = db.query(Program).filter(
                    Program.snies == snies).first()
                if not program:
                    skipped_no_program += 1
                    continue

                perfil_ingreso = _get_str_any(
                    row, ["perfil del ingreso", "perfil de ingreso", "perfil ingreso"])
                perfil_egresado = _get_str_any(
                    row, ["perfil de egresado", "perfil del egresado"])
                perfil_ocupacional = _get_str_any(
                    row, ["perfil ocupacional", "perfil ocupacional.1"])
                diferencial = _get_str_any(row, [
                    "diferencial (¿por qué estudiar aquí?)",
                    "diferencial (por qué estudiar aquí)",
                    "diferencial"
                ])
                requisitos = _get_str_any(
                    row, ["requisitos específicos", "requisitos especificos", "requisitos"])
                descripcion = _get_str_any(
                    row, ["descipción", "descripcion", "descripción"])

                n = db.query(ProgramNarrative).filter(
                    ProgramNarrative.program_id == program.id).first()
                if not n:
                    n = ProgramNarrative(program_id=program.id)
                    db.add(n)
                    db.flush()

                n.admission_profile = perfil_ingreso or n.admission_profile
                n.graduated_profile = perfil_egresado or n.graduated_profile
                n.occupational_profile = perfil_ocupacional or n.occupational_profile
                n.differential = diferencial or n.differential
                n.specific_requirements = requisitos or n.specific_requirements
                n.description = descripcion or n.description

                upserts += 1
                print(f"[NARRATIVA] snies leído='{snies}'")

            db.commit()

        print(
            "📝 NARRATIVA -> SQL OK | "
            f"sheet='{sheet}' rows={total} upserts={upserts} "
            f"skipped_no_snies={skipped_no_snies} skipped_no_program={skipped_no_program}"
        )

    # ---------------------------------------------------------
    # 5) TABULAR EMBEDDINGS
    # ---------------------------------------------------------
    def _sync_tabular_embeddings(self, xls: pd.ExcelFile):
        with SessionLocal() as db:
            db.query(ProgramEmbedding).filter(
                ProgramEmbedding.section.in_(list(TABULAR_SECTIONS))
            ).delete(synchronize_session=False)

            # course_row
            if "MALLA_CURRICULAR" in xls.sheet_names:
                df = _norm_cols(pd.read_excel(
                    xls, sheet_name="MALLA_CURRICULAR"))
                for _, row in df.iterrows():
                    snies = _clean_snies(_get_str_any(
                        row, ["código snies", "codigo snies"]))
                    if not snies:
                        continue
                    program = db.query(Program).filter(
                        Program.snies == snies).first()
                    if not program:
                        continue

                    semester = _get_int(
                        row, "semestre #") or _get_int(row, "semestre")
                    course_name = _get_str(row, "nombre de la asignatura")
                    credits = _get_int(row, "créditos asignatura") or _get_int(
                        row, "creditos asignatura")
                    if not course_name:
                        continue

                    content = (
                        f"Programa: {program.program_name} (SNIES {program.snies}). "
                        f"Malla curricular. Semestre {semester}. "
                        f"Asignatura: {course_name}. Créditos: {credits}."
                    )
                    db.add(ProgramEmbedding(
                        program_id=program.id,
                        section="course_row",
                        content=content,
                        chunk_index=0,
                        # FIX: embed_document usa prefijo "passage:" (indexado)
                        embedding=self.embedder.embed_document(content),
                    ))

            # elective_row
            if "ELECTIVAS" in xls.sheet_names:
                df = _norm_cols(pd.read_excel(xls, sheet_name="ELECTIVAS"))
                for _, row in df.iterrows():
                    snies = _clean_snies(_get_str_any(
                        row, ["código snies", "codigo snies"]))
                    if not snies:
                        continue
                    program = db.query(Program).filter(
                        Program.snies == snies).first()
                    if not program:
                        continue

                    name = _get_str(row, "nombre de la asignatura")
                    tipo = _get_str(row, "tipo")
                    if not name:
                        continue

                    content = f"Programa: {program.program_name} (SNIES {program.snies}). Electiva ({tipo or 'N/A'}): {name}."
                    db.add(ProgramEmbedding(
                        program_id=program.id,
                        section="elective_row",
                        content=content,
                        chunk_index=0,
                        # FIX: embed_document usa prefijo "passage:" (indexado)
                        embedding=self.embedder.embed_document(content),
                    ))

            # degree_option_row
            if "OPCIONES_GRADO" in xls.sheet_names:
                df = _norm_cols(pd.read_excel(
                    xls, sheet_name="OPCIONES_GRADO"))
                for _, row in df.iterrows():
                    snies = _clean_snies(_get_str_any(
                        row, ["código snies", "codigo snies"]))
                    if not snies:
                        continue
                    program = db.query(Program).filter(
                        Program.snies == snies).first()
                    if not program:
                        continue

                    opt = _get_str_any(
                        row, ["opción de grado", "opcion de grado"])
                    if not opt:
                        continue

                    content = f"Programa: {program.program_name} (SNIES {program.snies}). Opción de grado: {opt}."
                    db.add(ProgramEmbedding(
                        program_id=program.id,
                        section="degree_option_row",
                        content=content,
                        chunk_index=0,
                        # FIX: embed_document usa prefijo "passage:" (indexado)
                        embedding=self.embedder.embed_document(content),
                    ))

            db.commit()
        print("📌 Embeddings tabulares sincronizados.")

    # ---------------------------------------------------------
    # 6) NARRATIVA -> EMBEDDINGS
    # ---------------------------------------------------------
    def _sync_narrative_embeddings(self, xls: pd.ExcelFile):
        sheet = _find_sheet(xls, "NARRATIVA")
        if not sheet:
            print("⚠️ No existe la hoja NARRATIVA para embeddings.")
            return

        df = _norm_cols(pd.read_excel(xls, sheet_name=sheet))

        with SessionLocal() as db:
            cleaned: Set[int] = set()

            for _, row in df.iterrows():
                snies = _get_snies_from_row(row)
                if not snies:
                    continue

                program = db.query(Program).filter(
                    Program.snies == snies).first()
                if not program:
                    continue

                if program.id not in cleaned:
                    db.query(ProgramEmbedding).filter(
                        ProgramEmbedding.program_id == program.id,
                        ProgramEmbedding.section.in_(list(NARRATIVE_SECTIONS)),
                    ).delete(synchronize_session=False)
                    cleaned.add(program.id)

                perfil_ingreso = _get_str_any(
                    row, ["perfil del ingreso", "perfil de ingreso", "perfil ingreso"])
                perfil_egresado = _get_str_any(
                    row, ["perfil de egresado", "perfil del egresado"])
                perfil_ocupacional = _get_str_any(
                    row, ["perfil ocupacional", "perfil ocupacional.1"])

                sections: Dict[str, str] = {
                    "perfil_ingreso": perfil_ingreso,
                    "perfil_egresado": perfil_egresado,
                    "perfil_ocupacional": perfil_ocupacional,
                    "diferencial": _get_str_any(row, [
                        "diferencial (¿por qué estudiar aquí?)",
                        "diferencial (por qué estudiar aquí)",
                        "diferencial"
                    ]),
                    "requisitos": _get_str_any(row, ["requisitos específicos", "requisitos especificos", "requisitos"]),
                    "descripcion": _get_str_any(
                        row, ["descipción", "descripcion", "descripción"])
                }

                for section, text in sections.items():
                    if not (text or "").strip():
                        continue

                    chunks = simple_chunk(text, max_chars=900)
                    for idx, chunk in enumerate(chunks):
                        # Ancla fuerte: nombre del programa + snies + sección
                        anchored = (
                            f"Programa: {program.program_name} (SNIES {program.snies}). "
                            f"Sección: {section}. "
                            f"{chunk}"
                        )

                        db.add(ProgramEmbedding(
                            program_id=program.id,
                            section=section,
                            content=anchored,
                            chunk_index=idx,
                            # FIX: embed_document usa prefijo "passage:" (indexado)
                            embedding=self.narrative_embedder.embed_document(
                                anchored),
                        ))

            db.commit()

        print("🧠 Embeddings de NARRATIVA sincronizados.")
