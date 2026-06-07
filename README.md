# RAG Challenge API

API local construida con FastAPI para responder preguntas sobre un documento utilizando una arquitectura RAG con LLMs, embeddings y una base de datos vectorial.

## Stack inicial

- Python
- FastAPI
- Pydantic
- ChromaDB

## Configuracion de documento

`SOURCE_DOCUMENT_IS_DEFAULT=true` indica que se esta usando el documento base del
challenge. En ese modo la indexacion usa una estrategia especifica para ese
documento: divide el contenido por secciones tituladas, genera un chunk por
seccion y usa overlap `0`.

Si `SOURCE_DOCUMENT_IS_DEFAULT=false`, la indexacion usa el splitter generico y
respeta `TEXT_CHUNK_SIZE` y `TEXT_CHUNK_OVERLAP`.

# (En proceso de desarrollo)
