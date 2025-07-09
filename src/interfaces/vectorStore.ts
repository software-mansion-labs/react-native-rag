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
   * Adds a single document to the vector store.
   * @param document The content of the document.
   * @param metadata Optional metadata associated with the document.
   * @returns The ID of the newly added document.
   */
  add(document: string, metadata?: Record<string, any>): Promise<string>;

  /**
   * Updates a single document in the vector store by its ID.
   * You must provide at least one of `document` or `metadata` to update.
   * If `document` is provided, a new embedding will be generated.
   * @param id The ID of the document to update.
   * @param document Optional new content for the document.
   * @param metadata Optional new metadata for the document.
   */
  update(
    id: string,
    document?: string,
    metadata?: Record<string, any>
  ): Promise<void>;

  /**
   * Deletes a single document from the vector store by its ID.
   * @param id The ID of the document to delete.
   */
  delete(id: string): Promise<void>;

  /**
   * Performs a similarity search against the stored vectors.
   * @param query The query string to search for.
   * @param k The number of top similar results to return. Defaults to 3.
   * @returns An array of objects containing the ID, content, metadata, and similarity score for each result.
   */
  similaritySearch(
    query: string,
    k?: number,
    queryEmbedding?: number[]
  ): Promise<
    {
      id: string;
      content: string;
      metadata?: Record<string, any>;
      similarity: number;
    }[]
  >;
}
