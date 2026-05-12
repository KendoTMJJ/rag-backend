# Referencia API — RAG Backend

Base URL: `http://localhost:8000` (dev)

Todos los endpoints de administración requieren el header:
```
x-internal-key: <RAG_INTERNAL_API_KEY>
```

---

## Salud

### `GET /health`
Verifica que el servidor esté corriendo.

**Respuesta:** `{ "status": "ok" }`

---

## Chat (Posgrados)

### `POST /chat`

Punto de entrada principal del RAG. Recibe una pregunta y devuelve una respuesta estructurada según la ruta del pipeline.

**Body:**
```json
{
  "question": "¿Cuál es la duración de la maestría en educación?",
  "chatSessionId": "web-abc123"
}
```

**Respuesta:**
```json
{
  "route": "DEFAULT_RAG",
  "data": {
    "answer": "La maestría en Educación tiene una duración de...",
    "resolved": true,
    "source": "rag",
    "snies": 12345,
    "buttons": []
  }
}
```

**Rutas posibles del pipeline:**

| Ruta | Descripción |
|---|---|
| `GREETING` | Saludo o presentación |
| `THANKS` | Agradecimiento |
| `LIST_PROGRAMS` | Lista todos los programas |
| `LIST_PROGRAMS_FILTERED` | Lista filtrada por criterio |
| `PROGRAM_SELECTED` | Usuario seleccionó un programa |
| `PROGRAM_FIELD` | Pregunta sobre campo específico (costo, duración, etc.) |
| `PROGRAM_OVERVIEW` | Resumen general de un programa |
| `CURRICULUM` | Malla curricular completa |
| `CURRICULUM_SEMESTER` | Materias de un semestre específico |
| `ELECTIVES` | Materias electivas |
| `DEGREE_OPTIONS` | Opciones de grado |
| `INSCRIPTION_LINK` | Enlace de inscripción |
| `INSCRIPTION_NO_LINK` | Inscripción sin enlace disponible |
| `PROGRAM_MINMAX` | Comparación de programas |
| `NARRATIVE_SQL` | Consulta respondida con datos estructurados |
| `NARRATIVE_NOT_FOUND` | Sin datos para la consulta SQL |
| `DEFAULT_RAG` | Búsqueda semántica general |
| `LOW_CONFIDENCE` | Respuesta con baja confianza |
| `NOT_FOUND` | Sin resultados |
| `OUT_OF_DOMAIN` | Pregunta fuera del dominio académico |
| `INJECTION_BLOCKED` | Intento de inyección de prompt detectado |
| `NEED_PROGRAM` | Se necesita especificar el programa |

---

## Helpdesk (público)

Endpoints consumidos directamente por n8n o el widget.

### `GET /helpdesk/categories`

Lista todas las categorías activas (excluye intents reservados: `saludo`, `despedida`, `desconocida`).

**Respuesta (array):**
```json
[{
  "intent": "pagos",
  "display_label": "Pagos",
  "description": "Información sobre pagos y matrículas",
  "has_document": true,
  "document_url": "https://api.midominio.com/helpdesk/document/pagos",
  "pdf_url": "https://api.midominio.com/helpdesk/document/pagos"
}]
```

> `pdf_url` es un alias de `document_url` mantenido para compatibilidad con n8n.

---

### `POST /helpdesk/classify`

Clasifica la pregunta del usuario en un intent de helpdesk.

**Body:**
```json
{
  "question": "¿Cómo puedo pagar mi matrícula?",
  "chatSessionId": "web-abc123"
}
```

**Respuesta:**
```json
{
  "intent": "pagos",
  "message": null
}
```

Para `intent = "saludo"` o `"desconocida"` el campo `message` contiene texto listo para mostrar al usuario.

---

### `GET /helpdesk/category/:intent`

Obtiene una categoría específica por su intent.

**Respuesta:** mismo formato que `/helpdesk/categories` (objeto único).

---

### `GET /helpdesk/document/:intent`

Descarga el documento adjunto a la categoría (PDF, Word, PPT...).

**Respuesta:** archivo binario con `Content-Disposition: inline`.  
Devuelve `404` si la categoría no existe o no tiene documento.

---

## Helpdesk (administración) *(requiere x-internal-key)*

### `GET /admin/helpdesk/categories`

Lista todas las categorías con información completa incluyendo si tienen documento.

**Query param opcional:** `?intent=pagos` para filtrar.

**Respuesta (array):**
```json
[{
  "id": 1,
  "intent": "pagos",
  "display_label": "Pagos",
  "description": "Información sobre pagos",
  "has_document": true,
  "document_url": "/helpdesk/document/pagos"
}]
```

---

### `GET /admin/helpdesk/categories/:id`

### `POST /admin/helpdesk/categories`
```json
{ "intent": "paz_y_salvos", "description": "Solicitud de paz y salvos" }
```

**Reglas del `intent`:** solo minúsculas, números y guión bajo. Se convierte automáticamente en `display_label` con formato legible.

### `PATCH /admin/helpdesk/categories/:id`
```json
{ "description": "Nueva descripción" }
```

### `DELETE /admin/helpdesk/categories/:id`

---

### `POST /admin/helpdesk/categories/:id/document`

Sube o reemplaza el documento de una categoría. Máximo **20 MB**.

**Tipos permitidos:** `pdf`, `doc`, `docx`, `ppt`, `pptx`

`multipart/form-data` con campo `file`.

---

### `DELETE /admin/helpdesk/categories/:id/document`

Elimina el documento de la categoría (la categoría permanece).

---

## Knowledge (administración) *(requiere x-internal-key)*

### `POST /admin/knowledge/upload`

Ingesta programas de posgrado desde un archivo Excel.

`multipart/form-data` con campo `file` (`.xlsx`).

El proceso:
1. Parsea el Excel con `program_excel_parser.py`
2. Genera embeddings con `intfloat/multilingual-e5-base`
3. Almacena en pgvector para búsqueda semántica

**Respuesta:**
```json
{
  "inserted": 12,
  "updated": 3,
  "errors": []
}
```

---

## Notas sobre embeddings

- **Indexado** (documentos): se usa `embed_document()` que antepone el prefijo `"passage:"` al texto.
- **Consultas**: se usa `embed_query()` que antepone el prefijo `"query:"` al texto.

Estos prefijos son requeridos por el modelo `intfloat/multilingual-e5-base` para obtener buena similitud semántica. **No invertirlos.**
