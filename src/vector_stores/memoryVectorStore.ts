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
  private dim?: number;

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
   * Adds one or more documents to the in-memory store. Generates IDs when not provided.
   * @param params - Parameters for the operation.
   * @param params.ids - Optional IDs for each document (must match `documents.length`).
   * @param params.documents - Raw text content for each document.
   * @param params.embeddings - Optional embeddings for each document.
   * @param params.metadatas - Optional metadata for each document (aligned by index).
   * @returns Promise that resolves to the IDs of the newly added documents.
   */
  public async add(params: {
    ids?: string[];
    documents: string[];
    embeddings?: number[][];
    metadatas?: Record<string, any>[];
  }): Promise<string[]> {
    const { embeddings, documents, metadatas } = params;
    const ids = params.ids ?? documents.map(() => uuidv4());

    const idsLength = ids.length;
    this.assertLengthMatchIds(embeddings, idsLength);
    this.assertLengthMatchIds(documents, idsLength);
    this.assertLengthMatchIds(metadatas, idsLength);

    for (const id of ids) {
      if (this.rows.has(id)) {
        throw new Error(`id already exists: ${id}`);
      }
    }

    if (embeddings) {
      for (const emb of embeddings) {
        this.assertAndSetDim(emb);
      }
    }

    for (let i = 0; i < idsLength; i++) {
      this.rows.set(ids[i]!, {
        id: ids[i]!,
        document: documents[i]!,
        embedding: embeddings
          ? embeddings[i]!
          : await this.embeddings.embed(documents[i]!),
        metadata: metadatas ? metadatas[i]! : undefined,
      });
    }

    return ids;
  }

  /**
   * Updates one or more documents by ID. If `documents` are provided and
   * `embeddings` are not, fresh embeddings are generated automatically.
   * @param params - Parameters for the update.
   * @param params.ids - IDs of the documents to update.
   * @param params.embeddings - New embeddings (optional; aligned by index if provided).
   * @param params.documents - New content (optional; aligned by index if provided).
   * @param params.metadatas - New metadata (optional; aligned by index if provided).
   * @returns Promise that resolves when the update completes.
   */
  public async update(params: {
    ids: string[];
    embeddings?: number[][];
    documents?: string[];
    metadatas?: Record<string, any>[];
  }): Promise<void> {
    const { ids, embeddings, documents, metadatas } = params;

    const n = ids.length;
    this.assertLengthMatchIds(embeddings, n);
    this.assertLengthMatchIds(documents, n);
    this.assertLengthMatchIds(metadatas, n);

    for (const id of ids) {
      if (!this.rows.has(id)) {
        throw new Error(`id not found: ${id}`);
      }
    }

    if (embeddings) {
      for (const emb of embeddings) {
        this.assertAndSetDim(emb);
      }
    }

    for (let i = 0; i < n; i++) {
      const id = ids[i]!;
      const row = this.rows.get(id)!;

      this.rows.set(id, {
        id,
        document: documents ? documents[i]! : row.document,
        embedding: embeddings
          ? embeddings[i]!
          : documents
            ? await this.embeddings.embed(documents[i]!)
            : row.embedding,
        metadata: metadatas ? metadatas[i]! : row.metadata,
      });
    }
  }

  /**
   * Deletes documents by IDs and/or predicate.
   * @param params - Parameters for deletion.
   * @param params.ids - List of document IDs to delete.
   * @param params.predicate - Predicate to match documents for deletion.
   * @returns Promise that resolves when deletion completes.
   */
  public async delete(params: {
    ids?: string[];
    predicate?: (value: GetResult) => boolean;
  }): Promise<void> {
    const { ids, predicate } = params;

    if (ids && predicate) {
      for (const id of ids) {
        if (!this.rows.has(id)) {
          throw new Error(`id not found: ${id}`);
        }
      }

      for (const id of ids) {
        const row = this.rows.get(id)!;
        if (predicate(row)) {
          this.rows.delete(id);
        }
      }
    } else if (ids) {
      for (const id of ids) {
        if (!this.rows.has(id)) {
          throw new Error(`id not found: ${id}`);
        }
      }

      for (const id of ids) {
        this.rows.delete(id);
      }
    } else if (predicate) {
      for (const [id, row] of this.rows) {
        if (predicate(row)) {
          this.rows.delete(id);
        }
      }
    }
  }

  /**
   * Executes a cosine-similarity query over the in-memory vectors.
   * Provide exactly one of `queryTexts` or `queryEmbeddings`.
   * @param params - Query parameters.
   * @param params.queryTexts - Raw query strings to search for.
   * @param params.queryEmbeddings - Precomputed query embeddings.
   * @param params.nResults - Number of top results to return.
   * @param params.ids - Restrict the search to these document IDs.
   * @param params.predicate - Function to filter results after retrieval.
   * @returns Promise resolving to arrays of scored results for each query.
   */
  public async query(params: {
    queryTexts?: string[];
    queryEmbeddings?: number[][];
    nResults?: number;
    ids?: string[];
    predicate?: (value: QueryResult) => boolean;
  }): Promise<QueryResult[][]> {
    const {
      queryTexts,
      queryEmbeddings,
      nResults,
      ids,
      predicate = () => true,
    } = params;
    if (!queryTexts === !queryEmbeddings) {
      throw new Error(
        'Exactly one of queryTexts or queryEmbeddings must be provided'
      );
    }

    if (ids) {
      for (const id of ids) {
        if (!this.rows.has(id)) {
          throw new Error(`id not found: ${id}`);
        }
      }
    }

    const queries: number[][] = [];

    if (queryEmbeddings) {
      for (const emb of queryEmbeddings) {
        this.assertAndSetDim(emb);
        queries.push(emb);
      }
    } else if (queryTexts) {
      for (const text of queryTexts) {
        const emb = await this.embeddings.embed(text);
        queries.push(emb);
      }
    }

    const pool: GetResult[] = ids?.length
      ? ids.map((id) => this.rows.get(id)!)
      : Array.from(this.rows.values());

    const result: QueryResult[][] = [];

    for (const q of queries) {
      const scored = pool
        .map(
          (r) =>
            ({
              ...r,
              similarity: cosine(q, r.embedding),
            }) as QueryResult
        )
        .filter(predicate)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, nResults);

      result.push(scored);
    }

    return result;
  }

  /**
   * Ensures all embeddings share the same dimensionality, setting it on first use.
   */
  private assertAndSetDim(vec: number[]) {
    if (!Array.isArray(vec) || vec.length === 0) {
      throw new Error('embedding must be a non-empty vector');
    }
    if (this.dim === undefined) {
      this.dim = vec.length;
    } else if (vec.length !== this.dim) {
      throw new Error(
        `embedding dimension ${vec.length} does not match collection dimension ${this.dim}`
      );
    }
  }

  /**
   * Verifies optional arrays match expected length.
   */
  private assertLengthMatchIds<T>(arr: T[] | undefined, idsLength: number) {
    if (arr && arr.length !== idsLength) {
      throw new Error('array length must match ids length');
    }
  }
}
