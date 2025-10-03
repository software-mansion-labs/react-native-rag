import type { Embeddings } from '../interfaces/embeddings';
import type { VectorStore } from '../interfaces/vectorStore';
import type { GetResult, QueryResult } from '../types/common';
import { uuidv4 } from '../utils/uuidv4';
import { cosine } from '../utils/vectorMath';

/**
 * In-memory implementation of {@link VectorStore}.
 *
 * Useful for testing or ephemeral/local scenarios where persistence is not required.
 * Stores all vectors in a `Map` and performs cosine similarity search on demand.
 *
 * @example
 * const store = await new MemoryVectorStore({ embeddings }).load();
 * await store.add({ documents: ['hello world'] });
 * const [[top]] = await store.query({ queryTexts: ['hello'] });
 * console.log(top.id, top.similarity);
 */
export class MemoryVectorStore implements VectorStore {
  private embeddings: Embeddings;
  private rows = new Map<string, GetResult>();
  private embeddingDim?: number;

  /**
   * Creates a new in-memory vector store.
   */
  constructor({ embeddings }: { embeddings: Embeddings }) {
    this.embeddings = embeddings;
  }

  /**
   * Loads underlying dependencies (embeddings).
   * @returns Promise that resolves to the same instance.
   */
  public async load(): Promise<this> {
    await this.embeddings.load();
    this.embeddingDim = (await this.embeddings.embed('dummy')).length;
    return this;
  }

  /**
   * Unloads underlying dependencies (embeddings).
   * @returns Promise that resolves when unloading is complete.
   */
  public async unload(): Promise<void> {
    await this.embeddings.unload();
  }

  /**
   * Adds a document to the in-memory store.
   * @param params - Parameters for the operation.
   * @param params.id - ID for the document.
   * @param params.document - Raw text content for the document.
   * @param params.embedding - Embeddings for the document.
   * @param params.metadata - Metadata for the document.
   * @returns Promise that resolves to the ID of the newly added document.
   */
  public async add(params: {
    id?: string;
    document?: string;
    embedding?: number[];
    metadata?: Record<string, any>;
  }): Promise<string> {
    const { id = uuidv4(), document, embedding, metadata } = params;

    if (!document && !embedding) {
      throw new Error('document and embedding cannot be both undefined');
    }

    if (embedding && embedding.length !== this.embeddingDim) {
      throw new Error(
        `embedding dimension ${embedding.length} does not match collection embedding dimension ${this.embeddingDim}`
      );
    }

    if (this.rows.has(id)) {
      throw new Error(`id already exists: ${id}`);
    }

    this.rows.set(id, {
      id,
      document,
      embedding: embedding ?? (await this.embeddings.embed(document!)),
      metadata,
    });

    return id;
  }

  /**
   * Updates a document by ID.
   * Recomputes the embedding when `document` is provided and `embedding` is omitted.
   * @param params - Update parameters.
   * @param params.id - ID of the document to update.
   * @param params.document - New content.
   * @param params.embedding - New embedding.
   * @param params.metadata - New metadata.
   * @returns Promise that resolves when the update completes.
   */
  public async update(params: {
    id: string;
    document?: string;
    embedding?: number[];
    metadata?: Record<string, any>;
  }): Promise<void> {
    const { id, document, embedding, metadata } = params;

    if (embedding && embedding.length !== this.embeddingDim) {
      throw new Error(
        `embedding dimension ${embedding.length} does not match collection embedding dimension ${this.embeddingDim}`
      );
    }

    if (!this.rows.has(id)) {
      throw new Error(`id not found: ${id}`);
    }

    const oldRow = this.rows.get(id)!;

    this.rows.set(id, {
      id,
      document: document ?? oldRow.document,
      embedding:
        embedding ??
        (document ? await this.embeddings.embed(document!) : oldRow.embedding),
      metadata: metadata ?? oldRow.metadata,
    });
  }

  /**
   * Deletes documents by predicate.
   * @param params - Parameters for deletion.
   * @param params.predicate - Predicate to match documents for deletion.
   * @returns Promise that resolves once the documents are deleted.
   */
  public async delete(params: {
    predicate: (value: GetResult) => boolean;
  }): Promise<void> {
    const { predicate } = params;

    for (const [id, row] of this.rows) {
      if (predicate(row)) {
        this.rows.delete(id);
      }
    }
  }

  /**
   * Executes a cosine-similarity query over the in-memory vectors.
   * Provide exactly one of `queryText` or `queryEmbedding`.
   * @param params - Query parameters.
   * @param params.queryText - Raw query string to search for.
   * @param params.queryEmbedding - Precomputed query embedding.
   * @param params.nResults - Number of top results to return.
   * @param params.predicate - Function to filter results after retrieval.
   * @returns Promise that resolves to an array of {@link QueryResult}.
   */
  public async query(params: {
    queryText?: string;
    queryEmbedding?: number[];
    nResults?: number;
    predicate?: (value: QueryResult) => boolean;
  }): Promise<QueryResult[]> {
    const { queryText, queryEmbedding, nResults, predicate } = params;

    if (!queryText && !queryEmbedding) {
      throw new Error('queryText and queryEmbedding cannot be both undefined');
    }

    if (queryEmbedding && queryEmbedding.length !== this.embeddingDim) {
      throw new Error(
        `queryEmbedding dimension ${queryEmbedding.length} does not match collection embedding dimension ${this.embeddingDim}`
      );
    }

    const searchEmbedding =
      queryEmbedding ?? (await this.embeddings.embed(queryText!));

    return Array.from(this.rows.values())
      .map((r) => ({ ...r, similarity: cosine(searchEmbedding, r.embedding) }))
      .filter(predicate ?? (() => true))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, nResults);
  }
}
