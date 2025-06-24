import { type VectorStore, type Embeddings, uuidv4 } from 'react-native-rag';
import { open, type DB } from '@op-engineering/op-sqlite';

export interface OPSQLiteVectorStoreParams {
  name: string;
  embeddings: Embeddings;
}

export class OPSQLiteVectorStore implements VectorStore {
  private name: string;
  private embeddings: Embeddings;

  db: DB;

  constructor({ name, embeddings }: OPSQLiteVectorStoreParams) {
    this.name = name;
    this.embeddings = embeddings;
    this.db = open({
      name: this.name,
    });
  }

  async init() {
    await this.embeddings.load();
    const embedding_dim = (await this.embeddings.embed('dummy')).length;
    await this.db.execute(`
      CREATE TABLE IF NOT EXISTS vectors (
          id TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          embedding F32_BLOB(${embedding_dim}) NOT NULL,
          metadata JSON DEFAULT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_vectors_embedding ON vectors(
          libsql_vector_idx(embedding, 'compress_neighbors=float8', 'max_neighbors=100')
      );
    `);
    return this;
  }

  async add(document: string, metadata?: Record<string, any>) {
    const embedding = await this.embeddings.embed(document);
    const uuid = uuidv4();
    await this.db.execute(
      'INSERT INTO vectors(id, content, embedding, metadata) VALUES (?, ?, vector(?), ?)',
      [uuid, document, `[${embedding.join(',')}]`, JSON.stringify(metadata)]
    );
    return uuid;
  }

  async update(id: string, document?: string, metadata?: Record<string, any>) {
    if (!document && !metadata) {
      throw new Error(
        'At least one of document or metadata must be provided for update.'
      );
    }

    const response = await this.db.execute(
      'SELECT content, embedding, metadata FROM vectors WHERE id = ?',
      [id]
    );

    if (response.rows.length === 0) {
      throw new Error(`Document with id ${id} does not exist.`);
    }

    const existingRow = response.rows[0]!;

    const exisitingDocument = existingRow.content as string;
    const exisitingEmbedding = existingRow.embedding as Float32Array;
    const existingMetadata = existingRow.metadata;

    const updatedDocument = document ?? exisitingDocument;
    const updatedEmbedding =
      document !== exisitingDocument
        ? await this.embeddings.embed(updatedDocument)
        : exisitingEmbedding;
    const updatedMetadata =
      metadata ??
      (existingMetadata ? JSON.parse(existingMetadata as string) : {});

    await this.db.execute(
      'UPDATE vectors SET content = ?, embedding = vector(?), metadata = ? WHERE id = ?',
      [
        updatedDocument,
        `[${updatedEmbedding.join(',')}]`,
        JSON.stringify(updatedMetadata),
        id,
      ]
    );
  }

  async delete(id: string) {
    const response = await this.db.execute(
      'SELECT id FROM vectors WHERE id = ?',
      [id]
    );
    if (response.rows.length > 0) {
      await this.db.execute('DELETE FROM vectors WHERE id = ?', [id]);
    } else {
      throw new Error(`Document with id ${id} does not exist.`);
    }
  }

  async similaritySearch(query: string, k: number = 3) {
    const queryEmbedding = await this.embeddings.embed(query);
    const vectorStr = `[${queryEmbedding.join(',')}]`;
    const results = await this.db.execute(
      `
        SELECT id, content, vector_distance_cos(embedding, vector(?)) as cosine_distance, metadata
        FROM vectors
        ORDER BY cosine_distance ASC
        LIMIT ?;
      `,
      [vectorStr, k]
    );
    return results.rows.map((row) => {
      const id = row.id as string;
      const content = row.content as string;
      const metadata = row.metadata
        ? (JSON.parse(row.metadata as string) as Record<string, any>)
        : undefined;
      const similarity = 1 - (row.cosine_distance as number); // Convert cosine distance to similarity
      return { id, content, metadata, similarity };
    });
  }

  async deleteVectorStore() {
    await this.db.execute('DROP TABLE IF EXISTS vectors;');
  }
}
