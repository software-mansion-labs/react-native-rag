/**
 * Defines the essential operations for an embedding model.
 * An embedding model converts text into high-dimensional numerical vectors
 * (embeddings), capturing semantic meaning. These embeddings are crucial
 * for tasks like similarity search, clustering, and other NLP applications.
 * This interface covers model lifecycle (loading, unloading) and the core
 * embedding generation capability.
 */
export interface Embeddings {
  /**
   * Loads the embedding model and its necessary resources (e.g., weights, tokenizer) into memory.
   * This method should be called before attempting to generate any embeddings.
   * @returns A promise that resolves to the instance of the Embeddings class once loaded.
   */
  load: () => Promise<this>;

  /**
   * Unloads the embedding model and its associated resources from memory.
   * This is typically used to free up system resources when the model is no longer needed.
   * @returns A promise that resolves once the model unloading is complete.
   */
  unload: () => Promise<void>;

  /**
   * Generates a numerical embedding (vector) for a given text string.
   * @param text The input text for which to generate an embedding.
   * @returns A promise that resolves to an array of numbers representing the embedding vector.
   */
  embed: (text: string) => Promise<number[]>;
}
