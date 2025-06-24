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

  async load(): Promise<this> {
    await this.vectorStore.init();
    await this.llm.load();
    return this;
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

  public async generate(
    input: Message[] | string,
    augmentedGeneration: boolean = true,
    options: {
      k?: number;
      questionGenerator?: (messages: Message[]) => string;
      promptGenerator?: (
        messages: Message[],
        retrievedDocs: { content: string }[]
      ) => string;
    } = {},
    callback: (token: string) => void = () => {}
  ): Promise<string> {
    if (typeof input === 'string') {
      input = [{ role: 'user', content: input }];
    }
    if (!input.length) {
      throw new Error('No messages provided');
    }
    if (!augmentedGeneration) {
      return this.llm.generate(input, callback);
    }
    const {
      k = 3,
      questionGenerator = this.questionGenerator,
      promptGenerator = this.promptGenerator,
    } = options;
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

  interrupt() {
    this.llm.interrupt();
  }
}
