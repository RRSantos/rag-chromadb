from langchain.schema.document import Document

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class Database:
    def __init__(self, persist_directory: str, embedding: Embeddings):
        self._db = Chroma(
            persist_directory=persist_directory, embedding_function=embedding
        )

    def _calculate_chunk_ids(self, chunks):
        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks

    def upsert_chunks(self, chunks: list[Document]) -> int:
        # Load the existing database.

        # Calculate Page IDs.
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = self._db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        # print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            # print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self._db.add_documents(new_chunks, ids=new_chunk_ids)
            # self._db.persist()
        # else:
        #     print("✅ No new documents to add")

        return len(new_chunks)
