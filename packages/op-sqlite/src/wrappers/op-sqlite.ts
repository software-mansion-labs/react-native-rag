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
 * await store.add({ document: 'hello world' });
 * const [top] = await store.query({ queryText: 'hello', nResults: 1 });
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
   * Inserts a document with an embedding. Generates an ID when not provided.
   * @param params - Parameters for the operation.
   * @param params.id - ID for the document. If not provided, it will be auto-generated.
   * @param params.document - Raw text content for the document.
   * @param params.embedding - Embedding for the document. If not provided, it will be generated based on the `document`.
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

    if (
      (
        await this.db.execute('SELECT 1 FROM vectors WHERE id = ? LIMIT 1', [
          id,
        ])
      ).rows.length > 0
    ) {
      throw new Error(`id already exists: ${id}`);
    }

    await this.db.execute(
      'INSERT INTO vectors(id, document, embedding, metadata) VALUES (?, ?, vector(?), ?)',
      [
        id,
        document ?? '',
        `[${(embedding ?? (await this.embeddings.embed(document!))).join(',')}]`,
        metadata ? JSON.stringify(metadata) : null,
      ]
    );

    return id;
  }

  /**
   * Updates a document by ID.
   * @param params - Parameters for the update.
   * @param params.id - ID of the document to update.
   * @param params.document - New content for the document.
   * @param params.embedding - New embeddings for the document. If not provided, it will be generated based on the `document`.
   * @param params.metadata - New metadata for the document.
   * @returns Promise that resolves when the update completes.
   */
  public async update(params: {
    id: string;
    embedding?: number[];
    document?: string;
    metadata?: Record<string, any>;
  }): Promise<void> {
    const { id, document, embedding, metadata } = params;

    if (embedding && embedding.length !== this.embeddingDim) {
      throw new Error(
        `embedding dimension ${embedding.length} does not match collection embedding dimension ${this.embeddingDim}`
      );
    }

    if (
      (
        await this.db.execute('SELECT 1 FROM vectors WHERE id = ? LIMIT 1', [
          id,
        ])
      ).rows.length === 0
    ) {
      throw new Error(`id not found: ${id}`);
    }

    await this.db.execute(
      'UPDATE vectors SET document = ?, embedding = vector(?), metadata = ? WHERE id = ?',
      [
        document ?? '',
        `[${(embedding ?? (await this.embeddings.embed(document!))).join(',')}]`,
        metadata ? JSON.stringify(metadata) : null,
        id,
      ]
    );
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

    for (const row of (
      await this.db.execute(
        'SELECT id, document, embedding, metadata FROM vectors'
      )
    ).rows) {
      const getResult = this.rowToGetResult(row);
      if (predicate(getResult)) {
        await this.db.execute('DELETE FROM vectors WHERE id = ?', [
          getResult.id,
        ]);
      }
    }
  }

  /**
   * Executes a cosine-similarity query over stored vectors.
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

    const res = await this.db.execute(
      'SELECT id, document, embedding, metadata, (1.0 - vector_distance_cos(embedding, vector(?))) AS similarity FROM vectors ORDER BY similarity DESC',
      [`[${searchEmbedding.join(',')}]`]
    );

    return res.rows
      .map((r: any) => ({
        ...this.rowToGetResult(r),
        similarity: r.similarity as number,
      }))
      .filter(predicate ?? (() => true))
      .slice(0, nResults);
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
    return {
      id: row.id,
      document: row.document,
      embedding: Array.from(new Float32Array(row.embedding)),
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
    };
  }
}
