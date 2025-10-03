import type { GetResult, QueryResult } from '../types/common';

/**
 * Defines the essential operations for a vector store.
 * A vector store efficiently stores and retrieves high-dimensional vectors,
 * facilitating similarity searches for AI applications like semantic search.
 * It provides core functionalities for managing documents (add, update, delete)
 * and performing similarity-based queries.
 */
export interface VectorStore {
  /**
   * Initializes the vector store, loading necessary resources.
   * @returns Promise that resolves to the initialized vector store instance.
   */
  load: () => Promise<this>;

  /**
   * Unloads the vector store, releasing any resources used.
   * @returns Promise that resolves when the vector store is unloaded.
   */
  unload: () => Promise<void>;

  /**
   * Adds a document to the vector store.
   * @param params - Object containing:
   * @param params.id - The ID of the document. If not provided, it will be auto-generated.
   * @param params.document - Raw text content of the document.
   * @param params.embedding - Embedding for the document. If not provided, it will be generated based on the `document`.
   * @param params.metadata - Metadata associated with the document.
   * @returns Promise that resolves to the ID of the newly added document.
   */
  add(params: {
    id?: string;
    document?: string;
    embedding?: number[];
    metadata?: Record<string, any>;
  }): Promise<string>;

  /**
   * Updates a document in the vector store by its ID.
   * @param params - Object containing:
   * @param params.id - The ID of the document to update.
   * @param params.document - New content for the document.
   * @param params.embedding - New embedding for the document. If not provided, it will be generated based on the `document`.
   * @param params.metadata - New metadata for the document.
   * @returns Promise that resolves once the document is updated.
   */
  update(params: {
    id: string;
    document?: string;
    embedding?: number[];
    metadata?: Record<string, any>;
  }): Promise<void>;

  /**
   * Deletes documents from the vector store by the provided predicate.
   * @param params - Object containing:
   * @param params.predicate - Predicate to match documents for deletion.
   * @returns Promise that resolves once the documents are deleted.
   */
  delete(params: { predicate: (value: GetResult) => boolean }): Promise<void>;

  /**
   * Performs a similarity search against the stored vectors.
   * @param params - Object containing:
   * @param params.queryText - The raw query string to search for.
   * @param params.queryEmbedding - Pre-computed embedding for the query.
   * @param params.nResults - The number of top similar results to return.
   * @param params.predicate - Function to filter results after retrieval.
   * @returns Promise that resolves to an array of {@link QueryResult}.
   */
  query(params: {
    queryText?: string;
    queryEmbedding?: number[];
    nResults?: number;
    predicate?: (value: QueryResult) => boolean;
  }): Promise<QueryResult[]>;
}
