# Guía de Despliegue en Producción — Sistema RAG + Chat USTA Tunja

Guía paso a paso para desplegar el sistema completo en un servidor con dominio y DNS propios.  
El sistema está orquestado desde el repositorio `infra-deploy`, que contiene los tres servicios como submódulos de Git y un único `compose.yml`.

---

## Arquitectura del sistema

```
Internet
   │
   ▼
[Nginx] ← SSL termination (Certbot / Let's Encrypt)
   ├── /              → chat-frontend  (React SPA, puerto 3001 interno)
   ├── /api/chat/     → chat-backend   (NestJS,    puerto 3225 interno)
   ├── /api/rag/      → rag-backend    (FastAPI,   puerto 8000 interno)
   └── /n8n/          → n8n            (Orquestador, puerto 5678 interno)

[Ollama]  — corre en el HOST (no en Docker), accesible desde contenedores via host-gateway
[PostgreSQL × 2] — contenedores Docker (postgres-rag, postgres-chat)
```

---

## 1. Requisitos del servidor

| Requisito | Mínimo recomendado |
|---|---|
| OS | Ubuntu 22.04 LTS |
| CPU | 4 vCPU |
| RAM | 8 GB (16 GB si Ollama usa GPU) |
| Disco | 40 GB SSD (modelo LLM ocupa ~5 GB) |
| Docker + Docker Compose | Docker 24+, Compose v2 |
| Nginx | Cualquier versión reciente |
| Certbot | Para SSL automático |
| Ollama | Instalado en el host, no en Docker |
| Git | Para clonar con submódulos |

### Instalar Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Cerrar sesión y volver a entrar para aplicar el grupo
```

### Instalar Ollama en el host
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Descargar el modelo LLM
ollama pull llama3.1:8b-instruct-q4_K_M

# Verificar que el servidor queda escuchando
curl http://localhost:11434/api/tags
```

> **Importante:** Ollama debe arrancar automáticamente como servicio del sistema.  
> Si no, agrégalo con `sudo systemctl enable ollama && sudo systemctl start ollama`.

---

## 2. Clonar el repositorio de infraestructura

```bash
git clone --recurse-submodules https://github.com/KendoTMJJ/infra-deploy.git
cd infra-deploy
```

Si ya lo clonaste sin `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

Estructura resultante:
```
infra-deploy/
├── compose.yml
├── .env                  ← debes crearlo (ver sección 3)
├── rag-backend/          ← submódulo
├── chat-backend/         ← submódulo
└── chat-frontend/        ← submódulo
```

---

## 3. Configurar variables de entorno

Crea el archivo `.env` en la raíz de `infra-deploy/`:

```bash
cp .env.example .env   # si existe el ejemplo
# o créalo desde cero:
nano .env
```

### Contenido completo del `.env`

```dotenv
# ── PostgreSQL — RAG Backend ─────────────────────────────────────────────────
POSTGRES_RAG_USER=rag_user
POSTGRES_RAG_PASSWORD=CAMBIA_ESTO_rag
POSTGRES_RAG_DB=rag_db
POSTGRESQL_URL=postgresql://rag_user:CAMBIA_ESTO_rag@postgres-rag:5432/rag_db

# ── PostgreSQL — Chat Backend ────────────────────────────────────────────────
POSTGRES_CHAT_USER=chat_user
POSTGRES_CHAT_PASSWORD=CAMBIA_ESTO_chat
POSTGRES_CHAT_DB=chat_db
DATABASE_URL=postgresql://chat_user:CAMBIA_ESTO_chat@postgres-chat:5432/chat_db

# ── Ollama (LLM en el host) ──────────────────────────────────────────────────
# host-gateway resuelve a la IP del host desde dentro de los contenedores Docker
OLLAMA_BASE_URL=http://host-gateway:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
EMBEDDINGS_MODEL=intfloat/multilingual-e5-base

# ── RAG Backend ──────────────────────────────────────────────────────────────
RAG_INTERNAL_API_KEY=CAMBIA_ESTO_clave_interna_muy_larga
SESSION_TTL_SECONDS=7200
SESSION_MAX_SIZE=500

# URL pública del RAG backend (con dominio real en producción)
# Usado para construir los enlaces de documentos de helpdesk
PUBLIC_BASE_URL=https://tudominio.com/api/rag

# URL del catálogo de posgrados externo
CATALOG_POSGRADOS_URL=https://santotovirtual.edu.co/

# ── Chat Backend (NestJS) ────────────────────────────────────────────────────
RAG_URL=http://rag-backend:8000
JWT_SECRET=CAMBIA_ESTO_jwt_secret_muy_largo_y_aleatorio

# Cuenta administrador inicial (se crea al arrancar)
ADMIN_EMAIL=admin@tudominio.com
ADMIN_PASSWORD=CAMBIA_ESTO_admin_password_seguro

# URL pública del chat backend
PUBLIC_BASE_URL_CHAT=https://tudominio.com/api/chat

# ── n8n ──────────────────────────────────────────────────────────────────────
# Configurar DESPUÉS de que n8n esté corriendo y tengas el webhook URL real
N8N_WEBHOOK_URL=https://tudominio.com/n8n/webhook/TU_WEBHOOK_ID

# ── SMTP (notificaciones por correo) ────────────────────────────────────────
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=correo@tudominio.com
SMTP_PASSWORD=CAMBIA_ESTO_password_smtp

# ── Chat Frontend (React) ────────────────────────────────────────────────────
# Esta variable se BAKE en la imagen al hacer build — debe apuntar al chat backend
# Cambiarla después requiere reconstruir la imagen: docker compose build chat-frontend
VITE_SERVER_URL=https://tudominio.com/api/chat
```

> **Nota sobre `VITE_SERVER_URL`:** Esta variable se incrusta en el bundle de React
> durante el `docker compose build`. Si cambias el dominio después del build, debes
> reconstruir la imagen con `docker compose build chat-frontend`.

---

## 4. Levantar los servicios

```bash
# Primera vez — construye imágenes y arranca todo
docker compose up -d --build

# Ver logs en tiempo real
docker compose logs -f

# Ver estado de los contenedores
docker compose ps
```

### Verificar que cada servicio responde

```bash
# RAG backend
curl http://localhost:8000/health

# Chat backend
curl http://localhost:3225/health

# n8n
curl http://localhost:5678/healthz

# Chat frontend (sirve el HTML)
curl -s http://localhost:3001 | head -5
```

---

## 5. Configurar Nginx como reverse proxy

Instala Nginx en el host (no en Docker):

```bash
sudo apt install nginx -y
```

Crea el archivo de configuración del sitio. Primero configura sin SSL para verificar que todo funciona, luego agrega SSL con Certbot.

### 5.1 Configuración inicial (sin SSL)

```nginx
# /etc/nginx/sites-available/usta-tunja
server {
    listen 80;
    server_name tudominio.com www.tudominio.com;

    # Chat Frontend (React SPA)
    location / {
        proxy_pass         http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }

    # Chat Backend (NestJS — WebSockets incluidos)
    location /api/chat/ {
        proxy_pass         http://localhost:3225/;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    # RAG Backend (FastAPI)
    location /api/rag/ {
        proxy_pass         http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        # El RAG puede tardar hasta 60s respondiendo (LLM)
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        client_max_body_size 25M;  # Para subir archivos Excel / PDF
    }

    # n8n (Orquestador de flujos)
    location /n8n/ {
        proxy_pass         http://localhost:5678/;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/usta-tunja /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 5.2 Apuntar el DNS al servidor

En el panel de tu proveedor de dominio (Cloudflare, GoDaddy, etc.), crea estos registros DNS:

| Tipo | Nombre | Valor | TTL |
|---|---|---|---|
| A | `@` (raíz) | IP pública del servidor | 300 |
| A | `www` | IP pública del servidor | 300 |

Espera 5–30 minutos para que se propague. Verifica con:
```bash
dig tudominio.com +short
# debe devolver la IP del servidor
```

### 5.3 SSL con Certbot (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx -y

# Obtener e instalar certificado automáticamente
sudo certbot --nginx -d tudominio.com -d www.tudominio.com

# Certbot modifica nginx.conf automáticamente y agrega la redirección HTTP→HTTPS
sudo systemctl reload nginx
```

Certbot renueva el certificado automáticamente cada 90 días vía cron. Verifica:
```bash
sudo certbot renew --dry-run
```

---

## 6. Configuración de n8n (primera vez)

1. Abre `https://tudominio.com/n8n/` en el navegador.
2. Crea la cuenta de administrador.
3. Ve a **Settings → API Keys** → crea una API key.
4. Importa el workflow:
   - Menú → **Import from file** → selecciona `workflow.json`.
5. Actívalo (toggle en la esquina superior derecha).
6. Obtén el webhook URL del nodo trigger del workflow.
7. Actualiza el `.env`:
   ```dotenv
   N8N_WEBHOOK_URL=https://tudominio.com/n8n/webhook/TU_WEBHOOK_ID
   ```
8. Reinicia el chat-backend para que tome el nuevo webhook:
   ```bash
   docker compose restart chat-backend
   ```

### Actualizar el workflow vía API (sin UI)

Si necesitas actualizar el workflow programáticamente:

```bash
# Listar workflows para obtener el ID
curl -H "X-N8N-API-KEY: <tu-api-key>" https://tudominio.com/n8n/api/v1/workflows

# Actualizar (solo estos campos — el PUT rechaza propiedades adicionales)
node update_workflow.js   # script Node.js que hace GET, parchea nodes, hace PUT

# Activar
curl -X POST \
  -H "X-N8N-API-KEY: <tu-api-key>" \
  https://tudominio.com/n8n/api/v1/workflows/<id>/activate
```

> **Importante:** Las API keys de n8n expiran. Si obtienes `401 unauthorized`, crea
> una nueva en **Settings → API Keys**.

---

## 7. Cargar datos iniciales

### 7.1 Programas de posgrado (Excel)

```bash
curl -X POST https://tudominio.com/api/rag/admin/knowledge/upload \
  -H "x-internal-key: TU_RAG_INTERNAL_API_KEY" \
  -F "file=@programas.xlsx"
```

O desde el panel admin del chat-frontend: tab **Posgrados → Subir Excel**.

### 7.2 Categorías de mesa de ayuda (Helpdesk)

Desde el panel admin: tab **Mesa de Ayuda → Nueva categoría**.

Campos:
- `intent`: identificador único (solo letras, números y guión bajo — ej: `pagos_matricula`)
- `description`: descripción interna para el clasificador LLM (NO se muestra al usuario)
- `display_label`: nombre visible para el usuario (ej: "Pagos y Matrícula")
- Documento adjunto: PDF, Word o PPT (máx 20 MB) — opcional

### 7.3 Canales de soporte

Desde el panel admin: tab **Canales de soporte → Nuevo canal**.

- Contexto: `posgrados` o `mesa_ayuda`
- Número WhatsApp: formato `+57XXXXXXXXXX`
- Correo del área responsable

---

## 8. Verificación final del sistema

```bash
# RAG backend
curl https://tudominio.com/api/rag/health
# Esperado: {"status":"ok"}

# Chat backend
curl https://tudominio.com/api/chat/health
# Esperado: {"status":"ok"} o similar

# Flujo completo de chat
curl -X POST https://tudominio.com/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"hola","chatSessionId":"test-001"}'
```

---

## 9. Actualizar submódulos (nuevas versiones del código)

```bash
cd infra-deploy

# Actualizar un submódulo específico a su rama main
git -C rag-backend pull origin main
git -C chat-backend pull origin main
git -C chat-frontend pull origin main

# Reconstruir solo los servicios que cambiaron
docker compose build rag-backend
docker compose up -d rag-backend

# O reconstruir todo
docker compose up -d --build
```

---

## 10. Comandos útiles

```bash
# Ver logs de un servicio específico
docker compose logs -f rag-backend
docker compose logs -f chat-backend

# Reiniciar un servicio
docker compose restart rag-backend

# Detener todo
docker compose down

# Detener y eliminar volúmenes (¡DESTRUYE los datos de la BD!)
docker compose down -v

# Entrar a un contenedor
docker compose exec rag-backend bash
docker compose exec postgres-rag psql -U rag_user -d rag_db

# Ver uso de recursos
docker stats
```

---

## 11. Backups de PostgreSQL

```bash
# Backup manual
docker compose exec postgres-rag \
  pg_dump -U rag_user rag_db > backup_rag_$(date +%Y%m%d).sql

docker compose exec postgres-chat \
  pg_dump -U chat_user chat_db > backup_chat_$(date +%Y%m%d).sql

# Restaurar
cat backup_rag_20250101.sql | docker compose exec -T postgres-rag \
  psql -U rag_user -d rag_db
```

Para backups automáticos, agrega un cron job en el host:
```bash
crontab -e
# Agregar:
0 3 * * * cd /ruta/a/infra-deploy && docker compose exec -T postgres-rag pg_dump -U rag_user rag_db > /backups/rag_$(date +\%Y\%m\%d).sql
```

---

## 12. Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| `502 Bad Gateway` en Nginx | Contenedor no levantó o puerto incorrecto | `docker compose ps` y `docker compose logs <servicio>` |
| RAG tarda mucho o timeout | Ollama no corre en el host | `systemctl status ollama` y `curl http://localhost:11434/api/tags` |
| `VITE_SERVER_URL` apunta al lugar incorrecto | Variable bakeada con valor viejo | Actualizar `.env` y `docker compose build chat-frontend && docker compose up -d chat-frontend` |
| n8n `401 unauthorized` en API | API key expirada | Crear nueva en n8n → Settings → API Keys |
| n8n PUT `400 additional properties` | Se envió el JSON completo del workflow | Enviar solo `name`, `nodes`, `connections`, `settings: {executionOrder: 'v1'}` |
| Emojis corruptos (`??`) en nodos n8n | Workflow editado con PowerShell (encoding Windows-1252) | Usar Node.js para editar el JSON del workflow, nunca PowerShell heredoc |
| `EPERM` al escribir en WSL desde Windows | Permisos del sistema de archivos WSL | Operar desde dentro de WSL directamente (`wsl bash`) |
| Chat no recibe respuesta de n8n | `N8N_WEBHOOK_URL` incorrecto o n8n inactivo | Verificar URL del webhook y que el workflow esté activado |
| Primer arranque muy lento | RAG descarga ~550 MB del modelo de embeddings | Normal, ocurre solo la primera vez |
| Error `pgvector` al iniciar RAG | Extensión vector no instalada en la BD | `docker compose exec postgres-rag psql -U rag_user -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"` |
