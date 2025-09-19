import { RecursiveCharacterTextSplitter } from '../text-splitters/langchain';
import type { GetResult, Message, QueryResult } from '../types/common';
import type { LLM } from '../interfaces/llm';
import type { TextSplitter } from '../interfaces/textSplitter';
import type { VectorStore } from '../interfaces/vectorStore';
import { uuidv4 } from '../utils/uuidv4';

/**
 * Core Retrieval Augmented Generation orchestrator.
 *
 * The `RAG` class coordinates a `VectorStore` and an `LLM` to:
 * - Split and ingest documents for retrieval
 * - Retrieve relevant context for a query
 * - Generate responses with or without augmented context
 *
 * @example
 * const rag = await new RAG({ vectorStore, llm }).load();
 * const answer = await rag.generate({ input: 'What is RAG?' });
 * console.log(answer);
 */
export class RAG {
  private vectorStore: VectorStore;
  private llm: LLM;

  /**
   * Creates a new RAG instance.
   * @param params - Object containing the implementations.
   * @param params.vectorStore - Vector store used for retrieval.
   * @param params.llm - Large Language Model used for generation.
   */
  constructor({ vectorStore, llm }: { vectorStore: VectorStore; llm: LLM }) {
    this.vectorStore = vectorStore;
    this.llm = llm;
  }

  /**
   * Initializes the RAG system by loading the vector store and LLM.
   * @returns A promise that resolves to the same `RAG` instance.
   */
  async load(): Promise<this> {
    await this.vectorStore.load();
    await this.llm.load();
    return this;
  }

  /**
   * Unloads the RAG system, releasing resources used by the vector store and LLM.
   * @returns A promise that resolves when unloading is complete.
   */
  async unload(): Promise<void> {
    await this.vectorStore.unload();
    await this.llm.unload();
  }

  /**
   * Splits a document into chunks and adds them to the vector store.
   * If no `textSplitter` is provided, a default
   * `RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 100 })` is used.
   *
   * @param document - The content of the document to split and add.
   * @param metadataGenerator - Optional function to generate metadata for each chunk.
   * Must return an array which length is equal to the number of chunks.
   * @param textSplitter - Optional text splitter implementation.
   * @returns Promise that resolves to the IDs of the newly added chunks.
   */
  async splitAddDocument(
    document: string,
    metadataGenerator?: (chunks: string[]) => Record<string, any>[],
    textSplitter?: TextSplitter
  ): Promise<string[]> {
    if (!textSplitter) {
      textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100,
      });
    }
    const chunks = await textSplitter.splitText(document);
    const ids: string[] = chunks.map(() => uuidv4());
    const metadatas = metadataGenerator ? metadataGenerator(chunks) : undefined;

    if (metadatas && metadatas.length !== chunks.length) {
      throw new Error(
        `metadataGenerator must return metadata for all chunks: expected ${chunks.length}, got ${metadatas.length}`
      );
    }

    await this.vectorStore.add({
      ids,
      documents: chunks,
      metadatas,
    });

    return ids;
  }

  /**
   * Adds documents to the vector store.
   * @param params - Parameters for the operation.
   * @param params.ids - Optional IDs for each document (must match `documents.length`).
   * @param params.documents - Raw text content for each document.
   * @param params.embeddings - Optional embeddings for each document.
   * @param params.metadatas - Optional metadata for each document (aligned by index).
   * @returns Promise that resolves to the IDs of the newly added documents.
   */
  async addDocument(params: {
    ids?: string[];
    documents: string[];
    embeddings?: number[][];
    metadatas?: Record<string, any>[];
  }): Promise<string[]> {
    return this.vectorStore.add(params);
  }

  /**
   * Updates documents in the vector store by their IDs.
   * @param params - Parameters for the update.
   * @param params.ids - IDs of the documents to update.
   * @param params.embeddings - New embeddings (optional; aligned by index if provided).
   * @param params.documents - New content (optional; aligned by index if provided).
   * @param params.metadatas - New metadata (optional; aligned by index if provided).
   * @returns Promise that resolves when the update completes.
   */
  async updateDocument(params: {
    ids: string[];
    embeddings?: number[][];
    documents?: string[];
    metadatas?: Record<string, any>[];
  }): Promise<void> {
    return this.vectorStore.update(params);
  }

  /**
   * Deletes documents from the vector store.
   * @param params - Parameters for deletion.
   * @param params.ids - List of document IDs to delete.
   * @param params.predicate - Predicate to match documents for deletion.
   * @returns Promise that resolves when deletion completes.
   */
  async deleteDocument(params: {
    ids?: string[];
    predicate?: (value: GetResult) => boolean;
  }): Promise<void> {
    return this.vectorStore.delete(params);
  }

  private questionGenerator(messages: Message[]): string {
    const lastMessage = messages[messages.length - 1];
    return lastMessage?.content || '';
  }

  private promptGenerator(
    messages: Message[],
    retrievedDocs: QueryResult[]
  ): string {
    const lastMessage = messages[messages.length - 1];
    return `Message: ${lastMessage?.content || ''}
Context: ${retrievedDocs.map((result) => result.document).join('\n')}`;
  }

  /**
   * Generates a response based on the input messages and retrieved documents.
   * If `augmentedGeneration` is true, it retrieves relevant documents from the vector store
   * and includes them in the prompt for the LLM.
   *
   * @param params - Generation parameters.
   * @param params.input - Input messages or a single string.
   * @param params.augmentedGeneration - Whether to augment with retrieved context (default: true).
   * @param params.nResults - Number of docs to retrieve (default: 3).
   * @param params.predicate - Optional filter applied to retrieved docs.
   * @param params.questionGenerator - Maps the message list to a search query (default: last message content).
   * @param params.promptGenerator - Builds the context-augmented prompt from messages and retrieved docs.
   * @param params.callback - Optional token callback for streaming.
   * @returns Promise that resolves to the generated text.
   */
  public async generate(params: {
    input: Message[] | string;
    augmentedGeneration?: boolean;
    nResults?: number;
    predicate?: (value: QueryResult) => boolean;
    questionGenerator?: (messages: Message[]) => string;
    promptGenerator?: (
      messages: Message[],
      retrievedDocs: QueryResult[]
    ) => string;
    callback?: (token: string) => void;
  }): Promise<string> {
    let { input } = params;
    const {
      augmentedGeneration = true,
      nResults = 3,
      predicate = () => true,
      questionGenerator = this.questionGenerator,
      promptGenerator = this.promptGenerator,
      callback = () => {},
    } = params;

    if (typeof input === 'string') {
      input = [{ role: 'user', content: input }];
    }
    if (!input.length) {
      throw new Error('No messages provided');
    }

    if (!augmentedGeneration) {
      return this.llm.generate(input, callback);
    }

    const lastMessage = input[input.length - 1];
    if (!lastMessage?.content) {
      throw new Error('Last message has no content');
    }
    const retrievedDocs = await this.vectorStore.query({
      queryTexts: [questionGenerator(input)],
      nResults,
      queryEmbeddings: undefined,
      predicate,
    });
    const prompt = promptGenerator(input, retrievedDocs[0] ?? []);
    const augmentedInput: Message[] = [
      ...input,
      { role: 'user', content: prompt },
    ];
    return this.llm.generate(augmentedInput, callback);
  }

  /**
   * Interrupts the ongoing text generation process.
   * @returns Promise that resolves when the interruption is complete.
   */
  async interrupt(): Promise<void> {
    return this.llm.interrupt();
  }
}
