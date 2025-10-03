/**
 * Chat message in a conversation.
 */
export interface Message {
  /** Sender role. */
  role: 'user' | 'assistant' | 'system';
  /** Message text content. */
  content: string;
}

/**
 * Defines a type for a resource's origin or location.
 * - From React Native assets folder (For Files < 512MB)
 * - From remote URL
 * - From local file system
 */
export type ResourceSource = string | number | object;

/**
 * Single retrieval result.
 */
export interface GetResult {
  /** Document identifier. */
  id: string;
  /** Retrieved document text. */
  document?: string;
  /** Embedding vector for the document. */
  embedding: number[];
  /** Document metadata. */
  metadata?: Record<string, any>;
}

/**
 * Retrieval result with cosine similarity score.
 */
export interface QueryResult extends GetResult {
  /** Similarity score. */
  similarity: number;
}
