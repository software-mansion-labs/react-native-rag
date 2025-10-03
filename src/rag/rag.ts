import { RecursiveCharacterTextSplitter } from '../text-splitters/langchain';
import type { GetResult, Message, QueryResult } from '../types/common';
import type { LLM } from '../interfaces/llm';
import type { TextSplitter } from '../interfaces/textSplitter';
import type { VectorStore } from '../interfaces/vectorStore';
import { uuidv4 } from '../utils/uuidv4';

/**
 * Orchestrates Retrieval Augmented Generation.
 * Coordinates a `VectorStore` and an `LLM` to ingest, retrieve, and generate.
 *
 * @example
 * const rag = await new RAG({ vectorStore, llm }).load();
 * const answer = await rag.generate({ input: 'What is RAG?' });
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
   * @param params - Parameters for the operation.
   * @param params.document - The content of the document to split and add.
   * @param params.metadataGenerator - Function to generate metadata for each chunk. Must return an array which length is equal to the number of chunks.
   * @param params.textSplitter - Text splitter implementation.
   * @returns Promise that resolves to the IDs of the newly added chunks.
   */
  async splitAddDocument(params: {
    document: string;
    metadataGenerator?: (chunks: string[]) => Record<string, any>[];
    textSplitter?: TextSplitter;
  }): Promise<string[]> {
    let { document, metadataGenerator, textSplitter } = params;

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

    for (let i = 0; i < ids.length; i++) {
      await this.vectorStore.add({
        id: ids[i],
        document: chunks[i],
        metadata: metadatas ? metadatas[i] : undefined,
      });
    }

    return ids;
  }

  /**
   * Adds a document to the vector store.
   * @param params - Parameters for the operation.
   * @param params.id - ID for the document.
   * @param params.document - Raw text content for the document.
   * @param params.embedding - Embedding for the document.
   * @param params.metadata - Metadata for the document.
   * @returns Promise that resolves to the ID of the newly added document.
   */
  async addDocument(params: {
    id?: string;
    document?: string;
    embedding?: number[];
    metadata?: Record<string, any>;
  }): Promise<string> {
    return this.vectorStore.add(params);
  }

  /**
   * Updates a document in the vector store by its ID.
   * @param params - Parameters for the update.
   * @param params.id - The ID of the document to update.
   * @param params.document - New content for the document.
   * @param params.embedding - New embedding for the document. If not provided, it will be generated based on the `document`.
   * @param params.metadata - New metadata for the document.
   * @returns Promise that resolves once the document is updated.
   */
  async updateDocument(params: {
    id: string;
    embedding?: number[];
    document?: string;
    metadata?: Record<string, any>;
  }): Promise<void> {
    return this.vectorStore.update(params);
  }

  /**
   * Deletes documents from the vector store by the provided predicate.
   * @param params - Parameters for deletion.
   * @param params.predicate - Predicate to match documents for deletion.
   * @returns Promise that resolves once the documents are deleted.
   */
  async deleteDocument(params: {
    predicate: (value: GetResult) => boolean;
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
   * @param params - Generation parameters.
   * @param params.input - Input messages or a single string.
   * @param params.augmentedGeneration - Whether to augment with retrieved context (default: true).
   * @param params.nResults - Number of docs to retrieve (default: 3).
   * @param params.predicate - Filter applied to retrieved docs.
   * @param params.questionGenerator - Maps the message list to a search query (default: last message content).
   * @param params.promptGenerator - Builds the context-augmented prompt from messages and retrieved docs.
   * @param params.callback - Token callback for streaming.
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
      queryText: questionGenerator(input),
      nResults,
      predicate,
    });
    const prompt = promptGenerator(input, retrievedDocs);
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
