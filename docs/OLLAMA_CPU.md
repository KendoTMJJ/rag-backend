# Ejecutar Ollama sin GPU (producción en CPU)

El modelo `llama3.1:8b-instruct-q4_K_M` fue seleccionado específicamente por ser la variante cuantizada Q4_K_M que ofrece el mejor balance entre calidad de respuesta y consumo de memoria en CPU.

---

## Requisitos mínimos de hardware

| Recurso | Mínimo | Recomendado |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU cores | 4 | 8+ |
| Disco | 6 GB libres | 10 GB libres |

---

## Instalación de Ollama

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# Descargar el modelo
ollama pull llama3.1:8b-instruct-q4_K_M
```

---

## Variables de entorno para CPU

Configurar antes de iniciar `ollama serve`:

```bash
# Número de hilos de CPU a usar (ajustar según el servidor)
export OLLAMA_NUM_PARALLEL=1          # una solicitud a la vez en CPU
export OLLAMA_MAX_LOADED_MODELS=1     # solo un modelo en memoria

# Opcional: limitar a N hilos específicos
export OLLAMA_NUM_THREAD=4
```

En un archivo de servicio systemd (`/etc/systemd/system/ollama.service`):

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_THREAD=4"
ExecStart=/usr/local/bin/ollama serve
```

---

## Impacto en tiempos de respuesta

En CPU, el modelo puede tardar entre **15 y 60 segundos** por respuesta dependiendo de la longitud del contexto y los cores disponibles. El RAG pipeline tiene un timeout configurable — asegurarse de que `httpx` en el cliente Ollama tenga timeout suficiente.

En `src/services/llm_service.py`, el cliente ya usa `timeout=None` para evitar cortes prematuros en CPU.

---

## Modelo de embeddings

`intfloat/multilingual-e5-base` corre en CPU usando PyTorch. Se carga al iniciar la aplicación y permanece en memoria. Requiere aproximadamente **1 GB de RAM** adicional.

No requiere configuración especial para CPU — funciona automáticamente si no hay GPU disponible (PyTorch detecta `cuda` como no disponible y usa `cpu`).

---

## Docker en producción sin GPU

```dockerfile
# No usar la imagen nvidia/cuda — usar Python base
FROM python:3.11-slim

# Ollama debe correr en el host o en un contenedor separado
# Apuntar OLLAMA_BASE_URL al host:
# OLLAMA_BASE_URL=http://host.docker.internal:11434  (Docker Desktop)
# OLLAMA_BASE_URL=http://172.17.0.1:11434            (Linux Docker bridge)
```

Ollama no corre bien dentro de un contenedor Docker en CPU — es preferible ejecutarlo directamente en el host y apuntar `OLLAMA_BASE_URL` al host desde el contenedor.
