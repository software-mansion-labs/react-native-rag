import {
  type VectorStore,
  type Embeddings,
  type QueryResult,
  type GetResult,
  uuidv4,
} from 'react-native-rag';
import { open, type DB } from '@op-engineering/op-sqlite';

/**
 * SQLite-backed implementation of {@link VectorStore} using `@op-engineering/op-sqlite`.
 *
 * Stores embeddings as an F32 blob and uses vector indexing helpers for efficient
 * cosine-similarity search. Suitable for on-device retrieval in React Native apps.
 *
 * @example
 * const store = await new OPSQLiteVectorStore({ name: 'vector-db', embeddings }).load();
 * await store.add({ documents: ['hello world'] });
 * const [[top]] = await store.query({ queryTexts: ['hello'] });
 * console.log(top.id, top.similarity);
 */
export class OPSQLiteVectorStore implements VectorStore {
  private name: string;
  private embeddings: Embeddings;
  private embeddingDim?: number;

  db: DB;

  /**
   * Creates a new SQLite-backed vector store.
   */
  constructor({ name, embeddings }: { name: string; embeddings: Embeddings }) {
    this.name = name;
    this.embeddings = embeddings;
    this.db = open({ name: this.name });
  }

  /**
   * Opens the database, loads embeddings, and ensures schema/indexes exist.
   * @returns Promise that resolves to the same instance.
   */
  public async load(): Promise<this> {
    await this.embeddings.load();
    this.embeddingDim = (await this.embeddings.embed('dummy')).length;
    await this.db.execute(`
      CREATE TABLE IF NOT EXISTS vectors (
          id TEXT PRIMARY KEY,
          document TEXT NOT NULL,
          embedding F32_BLOB(${this.embeddingDim}) NOT NULL,
          metadata JSON DEFAULT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_vectors_embedding ON vectors(
          libsql_vector_idx(embedding, 'compress_neighbors=float8', 'max_neighbors=100')
      );
    `);
    return this;
  }

  /**
   * Closes embeddings and the underlying database connection.
   * @returns Promise that resolves when unloading is complete.
   */
  public async unload(): Promise<void> {
    await this.embeddings.unload();
    this.db.close();
  }

  /**
   * Inserts documents with embeddings. Generates IDs when not provided.
   * @param params - Parameters for the operation.
   * @param params.ids - Optional IDs for each document (must match `documents.length`). If not provided, IDs will be generated.
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
      const existing = await this.db.execute(
        'SELECT 1 FROM vectors WHERE id = ? LIMIT 1',
        [id]
      );
      if (existing.rows.length > 0) {
        throw new Error(`id already exists: ${id}`);
      }
    }

    if (embeddings) {
      for (const emb of embeddings) {
        this.assertEmbeddingDim(emb);
      }
    }

    for (let i = 0; i < idsLength; i++) {
      const meta = metadatas?.[i] ? JSON.stringify(metadatas[i]) : null;
      await this.db.execute(
        'INSERT INTO vectors(id, document, embedding, metadata) VALUES (?, ?, vector(?), ?)',
        [
          ids[i]!,
          documents[i]!,
          this.arrayToScalar(
            embeddings
              ? embeddings[i]!
              : await this.embeddings.embed(documents[i]!)
          ),
          meta,
        ]
      );
    }

    return ids;
  }

  /**
   * Updates documents by ID. If `documents` are provided and `embeddings` are not,
   * new embeddings are computed.
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

    const idsLength = ids.length;
    this.assertLengthMatchIds(embeddings, idsLength);
    this.assertLengthMatchIds(documents, idsLength);
    this.assertLengthMatchIds(metadatas, idsLength);

    for (const id of ids) {
      const existing = await this.db.execute(
        'SELECT 1 FROM vectors WHERE id = ? LIMIT 1',
        [id]
      );
      if (existing.rows.length === 0) {
        throw new Error(`id not found: ${id}`);
      }
    }

    if (embeddings) {
      for (const emb of embeddings) {
        this.assertEmbeddingDim(emb);
      }
    }

    for (let i = 0; i < idsLength; i++) {
      const id = ids[i]!;
      const row = await this.db.execute(
        'SELECT document, embedding, metadata FROM vectors WHERE id = ?',
        [id]
      );
      await this.db.execute(
        `
        UPDATE vectors
        SET document = ?,
            embedding = vector(?),
            metadata = ?
        WHERE id = ?
        `,
        [
          documents ? documents[i]! : row.rows[0]!.document!,
          embeddings
            ? this.arrayToScalar(embeddings[i]!)
            : documents
              ? this.arrayToScalar(await this.embeddings.embed(documents[i]!))
              : row.rows[0]!.embedding!,
          metadatas ? JSON.stringify(metadatas[i]) : row.rows[0]!.metadata!,
          id,
        ]
      );
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
        const existing = await this.db.execute(
          'SELECT 1 FROM vectors WHERE id = ? LIMIT 1',
          [id]
        );
        if (existing.rows.length === 0) {
          throw new Error(`id not found: ${id}`);
        }
      }

      const existingRows = await this.getRowsByIds(ids);
      const toDelete = existingRows.filter(predicate).map((r) => r.id);
      if (toDelete.length > 0) {
        await this.db.execute(
          `DELETE FROM vectors WHERE id IN (${toDelete.map(() => '?').join(',')})`,
          toDelete
        );
      }
    } else if (ids) {
      for (const id of ids) {
        const existing = await this.db.execute(
          'SELECT 1 FROM vectors WHERE id = ? LIMIT 1',
          [id]
        );
        if (existing.rows.length === 0) {
          throw new Error(`id not found: ${id}`);
        }
      }

      await this.db.execute(
        `DELETE FROM vectors WHERE id IN (${ids.map(() => '?').join(',')})`,
        ids
      );
    } else if (predicate) {
      const allRows = await this.db.execute(
        'SELECT id, document, embedding, metadata FROM vectors'
      );
      const toDelete: string[] = [];
      for (const row of allRows.rows) {
        const getRes = this.rowToGetResult(row);
        if (predicate(getRes)) {
          toDelete.push(getRes.id);
        }
      }

      if (toDelete.length > 0) {
        await this.db.execute(
          `DELETE FROM vectors WHERE id IN (${toDelete.map(() => '?').join(',')})`,
          toDelete
        );
      }
    }
  }

  /**
   * Executes a cosine-similarity query using SQLite vector functions.
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
        const existing = await this.db.execute(
          'SELECT 1 FROM vectors WHERE id = ? LIMIT 1',
          [id]
        );
        if (existing.rows.length === 0) {
          throw new Error(`id not found: ${id}`);
        }
      }
    }

    const queries: number[][] = [];

    if (queryEmbeddings) {
      for (const emb of queryEmbeddings) {
        this.assertEmbeddingDim(emb);
        queries.push(emb);
      }
    } else if (queryTexts) {
      for (const text of queryTexts) {
        const emb = await this.embeddings.embed(text);
        queries.push(emb);
      }
    }

    const pool: GetResult[] = ids?.length
      ? await this.getRowsByIds(ids)
      : (
          await this.db.execute(
            'SELECT id, document, embedding, metadata FROM vectors'
          )
        ).rows.map((r: any) => this.rowToGetResult(r));

    const results: QueryResult[][] = [];

    for (const q of queries) {
      const qScalar = this.arrayToScalar(q);
      let res;
      if (ids && ids.length) {
        res = await this.db.execute(
          `
          SELECT
            id,
            document,
            embedding,
            metadata,
            (1.0 - vector_distance_cos(embedding, vector(?))) AS similarity
          FROM vectors
          WHERE vectors.id IN (${ids.map(() => '?').join(',')})
          ORDER BY similarity DESC
        `,
          [qScalar, ...pool.map((r) => r.id)]
        );
      } else {
        res = await this.db.execute(
          `
          SELECT
            id,
            document,
            embedding,
            metadata,
            (1.0 - vector_distance_cos(embedding, vector(?))) AS similarity
          FROM vectors
          ORDER BY similarity DESC
        `,
          [qScalar]
        );
      }

      const scored = res.rows
        .map((r) => this.rowToGetResult(r) as QueryResult)
        .filter(predicate)
        .slice(0, nResults);

      results.push(scored);
    }

    return results;
  }

  /**
   * Drops the vectors table entirely.
   */
  public async deleteVectorStore(): Promise<void> {
    await this.db.execute('DROP TABLE IF EXISTS vectors;');
  }

  /**
   * Maps a DB row to a {@link GetResult} object.
   */
  private rowToGetResult(row: any): GetResult {
    const embedding = Array.isArray(row.embedding)
      ? (row.embedding as number[])
      : Array.from(row.embedding as Float32Array);
    return {
      id: row.id as string,
      document: (row.document as string) ?? '',
      embedding,
      metadata: row.metadata
        ? (JSON.parse(row.metadata as string) as Record<string, any>)
        : undefined,
    };
  }

  /**
   * Fetches rows by IDs and returns them as {@link GetResult} objects.
   */
  private async getRowsByIds(ids: string[]): Promise<GetResult[]> {
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => '?').join(',');
    const res = await this.db.execute(
      `SELECT id, document, embedding, metadata FROM vectors WHERE id IN (${placeholders})`,
      ids
    );
    return res.rows.map((r: any) => this.rowToGetResult(r));
  }

  /**
   * Converts an array of numbers to a string representation suitable for `vector(?)` binding.
   */
  private arrayToScalar(arr: number[]): string {
    return `[${arr.join(',')}]`;
  }

  /**
   * Verifies optional arrays match expected length.
   */
  private assertLengthMatchIds<T>(arr: T[] | undefined, idsLength: number) {
    if (arr && arr.length !== idsLength) {
      throw new Error('array length must match ids length');
    }
  }

  /**
   * Ensures all embeddings share the same dimensionality, setting it on first use.
   */
  private assertEmbeddingDim(vec: number[]) {
    if (!Array.isArray(vec) || vec.length === 0) {
      throw new Error('embedding must be a non-empty vector');
    }
    if (this.embeddingDim === undefined) {
      this.embeddingDim = vec.length;
    } else if (vec.length !== this.embeddingDim) {
      throw new Error(
        `embedding dimension ${vec.length} does not match collection dimension ${this.embeddingDim}`
      );
    }
  }
}
