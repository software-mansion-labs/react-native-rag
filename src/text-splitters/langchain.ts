import {
  CharacterTextSplitter as LangchainCharacterTextSplitter,
  RecursiveCharacterTextSplitter as LangchainRecursiveCharacterTextSplitter,
  TokenTextSplitter as LangchainTokenTextSplitter,
  MarkdownTextSplitter as LangchainMarkdownTextSplitter,
  LatexTextSplitter as LangchainLatexTextSplitter,
} from '@langchain/textsplitters';
import type { TextSplitter } from '../interfaces/textSplitter';

/**
 * Common parameters for configuring text splitters.
 */
interface TextSplitterParams {
  /**
   * The maximum size of each text chunk.
   */
  chunkSize: number;
  /**
   * The amount of overlap between consecutive text chunks, to maintain context.
   */
  chunkOverlap: number;
}

/**
 * A text splitter that splits text based on a fixed character count.
 * This is a wrapper around Langchain's `CharacterTextSplitter`.
 */
export class CharacterTextSplitter implements TextSplitter {
  private textSplitter: LangchainCharacterTextSplitter;

  /**
   * Creates an instance of CharacterTextSplitter.
   * @param {TextSplitterParams} params - The parameters for the text splitter.
   * @param {number} params.chunkSize - The maximum size of each text chunk.
   * @param {number} params.chunkOverlap - The amount of overlap between consecutive text chunks.
   */
  constructor({ chunkSize, chunkOverlap }: TextSplitterParams) {
    this.textSplitter = new LangchainCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
    });
  }

  /**
   * Splits the given text into chunks based on character count.
   * @param {string} text - The input text to be split.
   * @returns {Promise<string[]>} A promise that resolves to an array of string chunks.
   */
  async splitText(text: string): Promise<string[]> {
    return this.textSplitter.splitText(text);
  }
}

/**
 * A text splitter that attempts to preserve semantic boundaries by
 * recursively trying different separators.
 * This is a wrapper around Langchain's `RecursiveCharacterTextSplitter`.
 */
export class RecursiveCharacterTextSplitter implements TextSplitter {
  private textSplitter: LangchainRecursiveCharacterTextSplitter;

  /**
   * Creates an instance of RecursiveCharacterTextSplitter.
   * @param {TextSplitterParams} params - The parameters for the text splitter.
   * @param {number} params.chunkSize - The maximum size of each text chunk.
   * @param {number} params.chunkOverlap - The amount of overlap between consecutive text chunks.
   */
  constructor({ chunkSize, chunkOverlap }: TextSplitterParams) {
    this.textSplitter = new LangchainRecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
    });
  }

  /**
   * Splits the given text into chunks using a recursive character splitting strategy.
   * @param {string} text - The input text to be split.
   * @returns {Promise<string[]>} A promise that resolves to an array of string chunks.
   */
  async splitText(text: string): Promise<string[]> {
    return this.textSplitter.splitText(text);
  }
}

/**
 * A text splitter that splits text based on a token count rather than character count.
 * This is a wrapper around Langchain's `TokenTextSplitter`.
 */
export class TokenTextSplitter implements TextSplitter {
  private textSplitter: LangchainTokenTextSplitter;

  /**
   * Creates an instance of TokenTextSplitter.
   * @param {TextSplitterParams} params - The parameters for the text splitter.
   * @param {number} params.chunkSize - The maximum size of each text chunk (in tokens).
   * @param {number} params.chunkOverlap - The amount of overlap between consecutive text chunks (in tokens).
   */
  constructor({ chunkSize, chunkOverlap }: TextSplitterParams) {
    this.textSplitter = new LangchainTokenTextSplitter({
      chunkSize,
      chunkOverlap,
    });
  }

  /**
   * Splits the given text into chunks based on token count.
   * @param {string} text - The input text to be split.
   * @returns {Promise<string[]>} A promise that resolves to an array of string chunks.
   */
  async splitText(text: string): Promise<string[]> {
    return this.textSplitter.splitText(text);
  }
}

/**
 * A text splitter specifically designed for Markdown documents,
 * attempting to preserve Markdown structure within chunks.
 * This is a wrapper around Langchain's `MarkdownTextSplitter`.
 */
export class MarkdownTextSplitter implements TextSplitter {
  private textSplitter: LangchainMarkdownTextSplitter;

  /**
   * Creates an instance of MarkdownTextSplitter.
   * @param {TextSplitterParams} params - The parameters for the text splitter.
   * @param {number} params.chunkSize - The maximum size of each text chunk.
   * @param {number} params.chunkOverlap - The amount of overlap between consecutive text chunks.
   */
  constructor({ chunkSize, chunkOverlap }: TextSplitterParams) {
    this.textSplitter = new LangchainMarkdownTextSplitter({
      chunkSize,
      chunkOverlap,
    });
  }

  /**
   * Splits the given Markdown text into chunks.
   * @param {string} text - The input Markdown text to be split.
   * @returns {Promise<string[]>} A promise that resolves to an array of string chunks.
   */
  async splitText(text: string): Promise<string[]> {
    return this.textSplitter.splitText(text);
  }
}

/**
 * A text splitter specifically designed for LaTeX documents,
 * attempting to preserve LaTeX structure within chunks.
 * This is a wrapper around Langchain's `LatexTextSplitter`.
 */
export class LatexTextSplitter implements TextSplitter {
  private textSplitter: LangchainLatexTextSplitter;

  /**
   * Creates an instance of LatexTextSplitter.
   * @param {TextSplitterParams} params - The parameters for the text splitter.
   * @param {number} params.chunkSize - The maximum size of each text chunk.
   * @param {number} params.chunkOverlap - The amount of overlap between consecutive text chunks.
   */
  constructor({ chunkSize, chunkOverlap }: TextSplitterParams) {
    this.textSplitter = new LangchainLatexTextSplitter({
      chunkSize,
      chunkOverlap,
    });
  }

  /**
   * Splits the given LaTeX text into chunks.
   * @param {string} text - The input LaTeX text to be split.
   * @returns {Promise<string[]>} A promise that resolves to an array of string chunks.
   */
  async splitText(text: string): Promise<string[]> {
    return this.textSplitter.splitText(text);
  }
}
