/**
 * Defines the essential operations for an embedding model.
 * An embedding model converts text into high-dimensional numerical vectors
 * (embeddings), capturing semantic meaning. These embeddings are crucial
 * for tasks like similarity search, clustering, and other NLP applications.
 * This interface covers model lifecycle (loading, unloading) and the core
 * embedding generation capability.
 */
export interface EmbeddingsInterface {
  /**
   * Loads the embedding model and its necessary resources (e.g., weights, tokenizer) into memory.
   * This method should be called before attempting to generate any embeddings.
   * @returns A promise that resolves to the instance of the Embeddings class once loaded.
   */
  load: () => Promise<this>;

  /**
   * Deletes or unloads the embedding model and its associated resources from memory.
   * This is typically used to free up system resources when the model is no longer needed.
   */
  delete: () => void;

  /**
   * Generates a numerical embedding (vector) for a given text string.
   * @param text The input text for which to generate an embedding.
   * @returns A promise that resolves to an array of numbers representing the embedding vector.
   */
  embed: (text: string) => Promise<number[]>;
}

/**
 * Abstract base class for all embedding model implementations.
 * This class implements the EmbeddingsInterface and provides a common
 * structure for managing embedding models, including their initialization
 * and resource management. Concrete implementations must provide
 * the actual embedding logic.
 * @template EmbeddingsParams The type of parameters required by the specific embedding model.
 */
export declare abstract class Embeddings<EmbeddingsParams>
  implements EmbeddingsInterface
{
  constructor(params: EmbeddingsParams);

  /**
   * Loads the embedding model resources (e.g., weights, tokenizer) into memory.
   * This should be called before attempting to generate any embeddings.
   * @returns A promise that resolves to the instance of the Embeddings class once loaded.
   */
  abstract load: () => Promise<this>;

  /**
   * Deletes or unloads the embedding model and its associated resources from memory.
   * This is typically used to free up system resources when the model is no longer needed.
   */
  abstract delete: () => void;

  /**
   * Generates a numerical embedding (vector) for a given text string.
   * @param text The input text for which to generate an embedding.
   * @returns A promise that resolves to an array of numbers representing the embedding vector.
   */
  abstract embed: (text: string) => Promise<number[]>;
}
