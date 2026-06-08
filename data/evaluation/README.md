# Datasets para ejecutar la API

Estos datasets JSONL ejecutan secuencias ordenadas de preguntas contra la API usando contenido de `data/original_document.docx`.

El script imprime y guarda cada respuesta, y también calcula métricas RAG livianas para cada ejecución.

## Pares recomendados de modo/dataset

### mod1: Caché por pregunta exacta

Usar:

```env
CONVERSATION_CONTEXT_MODE=disabled
ANSWER_CACHE_MODE=question
```

Dataset:

```text
data/evaluation/mod1.jsonl
```

Este dataset repite preguntas exactas para poder observar el comportamiento de la caché, la latencia y el trade-off de reutilizar respuestas sin contexto del documento ni contexto conversacional.

### mod2: Contexto directo del documento

Usar:

```env
CONVERSATION_CONTEXT_MODE=disabled
ANSWER_CACHE_MODE=document_context
```

Dataset:

```text
data/evaluation/mod2.jsonl
```

Este dataset ejecuta preguntas directas e independientes, donde la recuperación de información debería ser suficiente. Es la configuración base de RAG, porque la reutilización de caché depende del contexto recuperado del documento.

### mod3: Caché sensible al contexto conversacional

Usar:

```env
CONVERSATION_CONTEXT_MODE=rewrite
ANSWER_CACHE_MODE=context_aware
CONVERSATION_HISTORY_LIMIT=10
```

Dataset:

```text
data/evaluation/mod3.jsonl
```

Este dataset envía conversaciones de múltiples turnos sin pasar el historial conversacional en la request. La API mantiene la conversación en memoria mientras el proceso está en ejecución. Sirve para probar la reescritura de consultas y el paso de LLM-as-a-judge usado por `ANSWER_CACHE_MODE=context_aware`.

Estos tres datasets están intencionalmente asociados a las tres configuraciones bajo prueba: caché por pregunta exacta (`mod1`), RAG directo con contexto del documento (`mod2`) y caché sensible al contexto conversacional con reescritura y judge (`mod3`).

## Ejecución

Iniciá la API con la configuración del `.env` que quieras probar y luego ejecutá:

```sh
python -m app.scripts.run_api_evaluation --dataset data/evaluation/mod3.jsonl
```

O ejecutá todos los datasets:

```sh
python -m app.scripts.run_api_evaluation
```

El script imprime los resultados en el orden de las requests y escribe la salida JSONL en:

```text
data/evaluation/results/
```

También escribe resúmenes de métricas y archivos de texto Prometheus en:

```text
data/evaluation/metrics/
```

Para agrupar las métricas de una ejecución en una carpeta con nombre:

```sh
python -m app.scripts.run_api_evaluation --dataset data/evaluation/mod1.jsonl --metrics-run-name cache-question-mode
```

Eso escribe las métricas en:

```text
data/evaluation/metrics/cache-question-mode/
```

Cada ejecución usa valores únicos de `user_name`, para que las conversaciones en memoria de una ejecución no se mezclen con las de otra.

## Métricas

Los datasets incluyen `expected_sections`, lo que permite calcular estas métricas de recuperación:

* `Recall@K`
* `MRR`
* `Context relevance`

El script también calcula aproximaciones léxicas automáticas para:

* `Groundedness`
* `Answer relevance`

Las métricas operacionales se miden directamente:

* `Latency total`
* `Latency by stage`, cuando la API devuelve diagnósticos
* `Estimated tokens per request`
* `Error rate`
* `Prompt injection attempts detected`
* `Cache hit rate`

`Citation accuracy` se emite solo cuando las respuestas incluyen citas entre corchetes que puedan compararse con `expected_sections`. El prompt actual de respuesta no solicita citas, por lo que esta métrica normalmente será `null`.

Usá `--k` para cambiar el valor de K usado por `Recall@K` y `MRR`:

```sh
python -m app.scripts.run_api_evaluation --dataset data/evaluation/mod2.jsonl --k 4
```

Los archivos `.prom` usan el formato de exposición de Prometheus. Para verlos en Grafana, el camino más simple es hacer que Prometheus scrapee o cargue esas métricas, y luego agregar Prometheus como fuente de datos en Grafana.
