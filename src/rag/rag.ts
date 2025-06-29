import { RecursiveCharacterTextSplitter } from '../text-splitters/langchain';
import type { Message } from '../types/common';
import type { LLM } from '../interfaces/llm';
import type { TextSplitter } from '../interfaces/textSplitter';
import type { VectorStore } from '../interfaces/vectorStore';

export interface RAGParams {
  vectorStore: VectorStore;
  llm: LLM;
}

export class RAG {
  private vectorStore: VectorStore;
  private llm: LLM;

  constructor({ vectorStore, llm }: RAGParams) {
    this.vectorStore = vectorStore;
    this.llm = llm;
  }

  /**
   * Initializes the RAG system by loading the vector store and LLM.
   * @returns A promise that resolves to the RAG instance.
   */
  async load(): Promise<this> {
    await this.vectorStore.load();
    await this.llm.load();
    return this;
  }

  /**
   * Unloads the RAG system, releasing resources used by the vector store and LLM.
   * @returns A promise that resolves when the RAG system is unloaded.
   */
  async unload(): Promise<void> {
    await this.vectorStore.unload();
    await this.llm.unload();
  }

  /**
   * Splits a document into chunks (smaller documents) and adds them to the vector store.
   * @param document The content of the document to split and add.
   * @param metadataGenerator An optional function to generate metadata for each chunk.
   * @param textSplitter An optional text splitter to use. If not provided, a default `RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 100 })` will be used.
   * @returns A promise that resolves to an array of IDs of the newly added document chunks.
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
    const ids: string[] = [];
    const metadatas = metadataGenerator ? metadataGenerator(chunks) : undefined;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]!;
      const metadata = metadatas?.[i];
      const id = await this.vectorStore.add(chunk, metadata);
      ids.push(id);
    }
    return ids;
  }

  /**
   * Adds a document to the vector store.
   * @param document The content of the document to add.
   * @param metadata Optional metadata associated with the document.
   * @returns A promise that resolves to the ID of the newly added document.
   */
  async addDocument(
    document: string,
    metadata?: Record<string, any>
  ): Promise<string> {
    return this.vectorStore.add(document, metadata);
  }

  /**
   * Updates a document in the vector store by its ID.
   * @param id The ID of the document to update.
   * @param document Optional new content for the document.
   * @param metadata Optional new metadata for the document.
   * @returns A promise that resolves when the document is updated.
   */
  async updateDocument(
    id: string,
    document?: string,
    metadata?: Record<string, any>
  ): Promise<void> {
    return this.vectorStore.update(id, document, metadata);
  }

  /**
   * Deletes a document from the vector store by its ID.
   * @param id The ID of the document to delete.
   * @returns A promise that resolves when the document is deleted.
   */
  async deleteDocument(id: string): Promise<void> {
    return this.vectorStore.delete(id);
  }

  private questionGenerator(messages: Message[]): string {
    const lastMessage = messages[messages.length - 1];
    return lastMessage?.content || '';
  }

  private promptGenerator(
    messages: Message[],
    retrievedDocs: { content: string }[]
  ): string {
    const lastMessage = messages[messages.length - 1];
    return `Message: ${lastMessage?.content || ''}
Context: ${retrievedDocs.map((doc) => doc.content).join('\n')}`;
  }

  /**
   * Generates a response based on the input messages and retrieved documents.
   * If `augmentedGeneration` is true, it retrieves relevant documents from the vector store
   * and includes them in the prompt for the LLM.
   * @param input The input messages or a single string message.
   * @param options Optional parameters for generation:
   * - `augmentedGeneration`: Whether to use augmented generation with retrieved documents (default: true).
   * - `k`: Number of documents to retrieve (default: 3).
   * - `questionGenerator`: Function to generate the question from messages for the document retrieval step. (default: uses last user message).
   * - `promptGenerator`: Function to generate the prompt from messages and retrieved documents (default: uses last user message and retrieved docs).
   * - `callback`: Optional callback function to handle token generation.
   * @returns A promise that resolves to the generated response string.
   */
  public async generate(
    input: Message[] | string,
    options: {
      augmentedGeneration?: boolean;
      k?: number;
      questionGenerator?: (messages: Message[]) => string;
      promptGenerator?: (
        messages: Message[],
        retrievedDocs: { content: string }[]
      ) => string;
      callback?: (token: string) => void;
    } = {}
  ): Promise<string> {
    if (typeof input === 'string') {
      input = [{ role: 'user', content: input }];
    }
    if (!input.length) {
      throw new Error('No messages provided');
    }

    const {
      augmentedGeneration = true,
      k = 3,
      questionGenerator = this.questionGenerator,
      promptGenerator = this.promptGenerator,
      callback = () => {},
    } = options;

    if (!augmentedGeneration) {
      return this.llm.generate(input, callback);
    }

    const lastMessage = input[input.length - 1];
    if (!lastMessage?.content) {
      throw new Error('Last message has no content');
    }
    const retrievedDocs = await this.vectorStore.similaritySearch(
      questionGenerator(input),
      k
    );
    const prompt = promptGenerator(input, retrievedDocs);
    const augmentedInput: Message[] = [
      ...input,
      { role: 'user', content: prompt },
    ];
    return this.llm.generate(augmentedInput, callback);
  }

  /**
   * Interrupts the ongoing text generation process.
   * @returns A promise that resolves when the interruption is complete.
   */
  async interrupt(): Promise<void> {
    return this.llm.interrupt();
  }
}
