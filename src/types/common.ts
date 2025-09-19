/**
 * Represents a chat message exchanged in a conversation.
 * - `role`: Identifies the sender (`user`, `assistant`, or `system`).
 * - `content`: The text content of the message.
 */
export interface Message {
  role: 'user' | 'assistant' | 'system';
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
 * Represents a single retrieval result.
 * Each field is aligned by index.
 * - `document`: Retrieved document text.
 * - `embedding`: Embedding vector for the document.
 * - `id`: Document identifier.
 * - `metadata`: Optional metadata object (`Record<string, any>`).
 */
export interface GetResult {
  document: string;
  embedding: number[];
  id: string;
  metadata?: Record<string, any>;
}

/**
 * Represents a single scored result from a similarity query.
 * Extends {@link GetResult} with a `similarity` score in the range [-1, 1].
 */
export interface QueryResult extends GetResult {
  similarity: number;
}
