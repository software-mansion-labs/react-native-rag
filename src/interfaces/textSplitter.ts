/**
 * Defines the core functionality for splitting text into smaller chunks.
 * This is essential for NLP and LLM applications, helping to manage
 * text size for model token limits, improve RAG, and enable focused
 * text analysis. Implementations will define specific splitting strategies
 * (e.g., by character, word, or semantic boundaries) while aiming to
 * preserve context.
 */
export interface TextSplitter {
  /**
   * Splits a given text into an array of strings (chunks).
   * @param text The input text to be split.
   * @returns A promise that resolves to an array of string chunks.
   */
  splitText: (text: string) => Promise<string[]>;
}
