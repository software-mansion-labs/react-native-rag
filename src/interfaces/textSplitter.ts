/**
 * Defines the core functionality for splitting text into smaller chunks.
 * This is essential for NLP and LLM applications, helping to manage
 * text size for model token limits, improve RAG, and enable focused
 * text analysis. Implementations will define specific splitting strategies
 * (e.g., by character, word, or semantic boundaries) while aiming to
 * preserve context.
 */
export interface TextSplitterInterface {
  /**
   * Splits a given text into an array of strings (chunks).
   * @param text The input text to be split.
   * @returns A promise that resolves to an array of string chunks.
   */
  splitText: (text: string) => Promise<string[]>;
}

/**
 * Abstract base class for all text splitter implementations.
 * This class implements the TextSplitterInterface and provides a common
 * structure for text splitting. Concrete implementations must define
 * the specific logic for how text is divided into chunks based on
 * their `TextSplitterParams`.
 * @template TextSplitterParams The type of parameters required by the specific text splitter.
 */
export declare abstract class TextSplitter<TextSplitterParams>
  implements TextSplitterInterface
{
  constructor(params: TextSplitterParams);

  /**
   * Splits a given text into an array of strings (chunks).
   * @param text The input text to be split.
   * @returns A promise that resolves to an array of string chunks.
   */
  abstract splitText: (text: string) => Promise<string[]>;
}
