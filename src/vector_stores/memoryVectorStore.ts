import type { Embeddings } from '../interfaces/embeddings';
import type { VectorStore } from '../interfaces/vectorStore';
import type { SearchResult } from '../types/common';
import { uuidv4 } from '../utils/uuidv4';
import { cosine } from '../utils/vectorMath';

interface MemoryVector {
  id: string;
  content: string;
  embedding: number[];
  metadata?: Record<string, any>;
}

interface MemoryVectorStoreParams {
  embeddings: Embeddings;
}

export class MemoryVectorStore implements VectorStore {
  private embeddings: Embeddings;

  private memoryVectors: Map<string, MemoryVector> = new Map();

  constructor({ embeddings }: MemoryVectorStoreParams) {
    this.embeddings = embeddings;
  }

  async load(): Promise<this> {
    await this.embeddings.load();
    return this;
  }

  async unload(): Promise<void> {
    await this.embeddings.unload();
  }

  async add(document: string, metadata?: Record<string, any>): Promise<string> {
    const embedding = await this.embeddings.embed(document);
    const uuid = uuidv4();
    this.memoryVectors.set(uuid, {
      id: uuid,
      content: document,
      embedding,
      metadata,
    });
    return uuid;
  }

  async update(
    id: string,
    document?: string,
    metadata?: Record<string, any>
  ): Promise<void> {
    if (!document && !metadata) {
      throw new Error(
        'At least one of document or metadata must be provided for update.'
      );
    }

    if (this.memoryVectors.has(id)) {
      const existingVector = this.memoryVectors.get(id)!;
      const updatedContent = document ?? existingVector.content;
      const updatedMetadata = metadata ?? existingVector.metadata;

      let embedding = existingVector.embedding;
      if (document) {
        embedding = await this.embeddings.embed(updatedContent);
      }

      this.memoryVectors.set(id, {
        id,
        content: updatedContent,
        embedding,
        metadata: updatedMetadata,
      });
    } else {
      throw new Error(`Document with id ${id} does not exist.`);
    }
  }

  async delete(id: string): Promise<void> {
    if (this.memoryVectors.has(id)) {
      this.memoryVectors.delete(id);
    } else {
      throw new Error(`Document with id ${id} does not exist.`);
    }
  }

  async similaritySearch(
    query: string,
    k: number = 3,
    predicate: (value: SearchResult) => boolean = () => true
  ): Promise<SearchResult[]> {
    const queryEmbedding = await this.embeddings.embed(query);
    const results = Array.from(this.memoryVectors.values())
      .map((memoryVector) => ({
        id: memoryVector.id,
        content: memoryVector.content,
        metadata: memoryVector.metadata,
        similarity: cosine(queryEmbedding, memoryVector.embedding),
      }))
      .filter(predicate)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
    return results;
  }
}
