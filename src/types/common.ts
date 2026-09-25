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
