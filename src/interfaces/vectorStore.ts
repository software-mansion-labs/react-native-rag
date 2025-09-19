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
   * @returns A promise that resolves to the initialized vector store instance.
   */
  load: () => Promise<this>;

  /**
   * Unloads the vector store, releasing any resources used.
   * @returns A promise that resolves when the vector store is unloaded.
   */
  unload: () => Promise<void>;

  /**
   * Adds documents to the vector store.
   * @param params - Object containing:
   *   - `ids`: (optional) The IDs of the documents. If not provided, they will be auto-generated.
   *   - `documents`: Raw text content of the documents.
   *   - `embeddings` (optional): Embeddings for the documents.
   *   - `metadatas` (optional): Metadata associated with each document.
   * @returns A promise that resolves to the IDs of the newly added documents.
   */
  add(params: {
    ids?: string[];
    documents: string[];
    embeddings?: number[][];
    metadatas?: Record<string, any>[];
  }): Promise<string[]>;

  /**
   * Updates documents in the vector store by their IDs.
   * If `documents` are provided, and `embeddings` are not, new embeddings will be generated.
   * @param params - Object containing:
   *   - `ids`: The IDs of the documents to update.
   *   - `embeddings` (optional): New embeddings for the documents.
   *   - `documents` (optional): New content for the documents.
   *   - `metadatas` (optional): New metadata for the documents.
   * @returns A promise that resolves when the documents are updated.
   */
  update(params: {
    ids: string[];
    embeddings?: number[][];
    documents?: string[];
    metadatas?: Record<string, any>[];
  }): Promise<void>;

  /**
   * Deletes documents from the vector store.
   * @param params - Object containing:
   *   - `ids` (optional): List of document IDs to delete.
   *   - `predicate` (optional): Predicate to match documents for deletion.
   * @returns A promise that resolves when the documents are deleted.
   */
  delete(params: {
    ids?: string[];
    predicate?: (value: GetResult) => boolean;
  }): Promise<void>;

  /**
   * Performs a similarity search against the stored vectors.
   * @param params - Object containing:
   *   - `queryTexts` (optional): The raw query strings to search for.
   *   - `queryEmbeddings` (optional): Pre-computed embeddings for the queries.
   *   - `nResults` (optional): The number of top similar results to return per query.
   *   - `ids` (optional): Restrict the search to these document IDs.
   *   - `predicate` (optional): Function to filter results after retrieval.
   * @returns A promise that resolves to an array of result arrays (one per query).
   */
  query(params: {
    queryTexts?: string[];
    queryEmbeddings?: number[][];
    nResults?: number;
    ids?: string[];
    predicate?: (value: QueryResult) => boolean;
  }): Promise<QueryResult[][]>;
}
