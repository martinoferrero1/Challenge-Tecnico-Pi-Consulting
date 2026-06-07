import asyncio

from app.core.config import settings
from app.infrastructure.pipelines.indexing_pipeline import (
    create_index_document_use_case,
)


async def run() -> None:
    print("Indexing document...")
    use_case = create_index_document_use_case(settings)
    print(f"Using document path: {settings.source_document_path}")
    result = await use_case.execute(settings.source_document_path)
    print("Document indexed successfully.")
    document = result.ingested_document.document
    print(f"Document ID: {document.id}")
    chunks_count = len(result.ingested_document.chunks)
    print(f"Document has {chunks_count} chunks.")
    print(
        f"Indexed document '{document.id}' with "
        f"{chunks_count} chunks and {result.embeddings_count} embeddings."
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
