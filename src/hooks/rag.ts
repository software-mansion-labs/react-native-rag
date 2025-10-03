import { useEffect, useState, useCallback, useMemo } from 'react';
import { RAG } from '../rag/rag';
import type { Message, QueryResult, GetResult } from '../types/common';
import type { TextSplitter } from '../interfaces/textSplitter';
import type { VectorStore } from '../interfaces/vectorStore';
import type { LLM } from '../interfaces/llm';

/**
 * React hook for Retrieval Augmented Generation.
 * Manages load/unload, generation, and document storage.
 *
 * @returns State and operations: `response`, `isReady`, `isGenerating`, `isStoring`, `error`, `generate`, `interrupt`, `splitAddDocument`, `addDocument`, `updateDocument`, `deleteDocument`.
 */
export function useRAG({
  vectorStore,
  llm,
  preventLoad = false,
}: {
  vectorStore: VectorStore;
  llm: LLM;
  preventLoad?: boolean;
}) {
  const [response, setResponse] = useState<string>('');
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStoring, setIsStoring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const rag = useMemo(() => new RAG({ vectorStore, llm }), [vectorStore, llm]);

  useEffect(() => {
    setError(null);

    if (preventLoad) return;

    (async () => {
      try {
        await rag.load();
        setIsReady(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'RAG load error.');
      }
    })();

    const unload = async () => {
      try {
        await rag.unload();
        setIsReady(false);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'RAG unload error.');
      }
    };

    return () => {
      if (isReady) {
        unload();
      }
    };
  }, [rag, preventLoad, isReady]);

  /**
   * Generates a text response.
   * @param params - Parameters for the generation.
   * @param params.input - User input as a string or array of messages.
   * @param params.augmentedGeneration - Whether to use RAG augmentation (default: true).
   * @param params.nResults - Number of documents to retrieve for augmentation (default: 3).
   * @param params.predicate - Predicate to filter retrieved documents.
   * @param params.questionGenerator - Function to generate a question from messages.
   * @param params.promptGenerator - Function to generate a prompt from messages and retrieved documents.
   * @param params.callback - Callback function for streaming tokens.
   * @returns A promise that resolves to the generated text.
   * @throws Error if RAG is not ready or is currently generating.
   */
  const generate = useCallback(
    async (params: {
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
    }): Promise<string> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isGenerating) throw new Error('RAG busy generating.');
      setResponse('');
      setError(null);
      try {
        setIsGenerating(true);
        return await rag.generate({
          ...params,
          callback: (token: string) => {
            setResponse((prev) => prev + token);
            params.callback?.(token);
          },
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Generation error.');
        throw e;
      } finally {
        setIsGenerating(false);
      }
    },
    [rag, isReady, isGenerating]
  );

  /**
   * Interrupts ongoing text generation.
   * @throws {Error} If not ready or not generating.
   * @returns A promise that resolves when the interruption is complete.
   */
  const interrupt = useCallback(async (): Promise<void> => {
    if (!isReady) throw new Error('RAG not ready.');
    if (!isGenerating) throw new Error('RAG not generating.');
    return rag.interrupt();
  }, [rag, isGenerating, isReady]);

  /**
   * Splits and adds a document to the vector store.
   * @param params - Parameters for the operation.
   * @param params.document - Document content.
   * @param params.metadataGenerator - Metadata generator.
   * @param params.textSplitter - Text splitter.
   * @returns IDs of added chunks.
   */
  const splitAddDocument = useCallback(
    async (params: {
      document: string;
      metadataGenerator?: (chunks: string[]) => Record<string, any>[];
      textSplitter?: TextSplitter;
    }): Promise<string[]> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.splitAddDocument(params);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Split/add doc error.');
        throw e;
      } finally {
        setIsStoring(false);
      }
    },
    [rag, isReady, isStoring]
  );

  /**
   * Adds a document to the vector store.
   * @param params - Parameters for the operation.
   * @param params.id - The ID of the document. If not provided, it will be auto-generated.
   * @param params.document - Raw text content of the document.
   * @param params.embedding - Embedding for the document. If not provided, it will be generated based on the `document`.
   * @param params.metadata - Metadata associated with the document.
   * @returns A promise that resolves to the ID of the newly added document.
   * @throws Error if RAG is not ready or is currently storing.
   */
  const addDocument = useCallback(
    async (params: {
      id?: string;
      document?: string;
      embedding?: number[];
      metadata?: Record<string, any>;
    }): Promise<string> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.addDocument(params);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Add doc error.');
        throw e;
      } finally {
        setIsStoring(false);
      }
    },
    [rag, isReady, isStoring]
  );

  /**
   * Updates a document in the vector store by its ID.
   * @param params - Parameters for the update.
   * @param params.id - The ID of the document to update.
   * @param params.document - New content for the document.
   * @param params.embedding - New embedding for the document. If not provided, it will be generated based on the `document`.
   * @param params.metadata - New metadata for the document.
   * @returns A promise that resolves when the document is updated.
   * @throws Error if RAG is not ready or is currently storing.
   */
  const updateDocument = useCallback(
    async (params: {
      id: string;
      embedding?: number[];
      document?: string;
      metadata?: Record<string, any>;
    }): Promise<void> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.updateDocument(params);
      } finally {
        setIsStoring(false);
      }
    },
    [rag, isReady, isStoring]
  );

  /**
   * Deletes documents from the vector store by the provided predicate.
   * @param params - Parameters for deletion.
   * @param params.predicate - Predicate to match documents for deletion.
   * @returns A promise that resolves once the documents are deleted.
   * @throws Error if RAG is not ready or is currently storing.
   */
  const deleteDocument = useCallback(
    async (params: {
      predicate: (value: GetResult) => boolean;
    }): Promise<void> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.deleteDocument(params);
      } finally {
        setIsStoring(false);
      }
    },
    [rag, isReady, isStoring]
  );

  return {
    response,
    isReady,
    isGenerating,
    isStoring,
    error,
    generate,
    interrupt,
    splitAddDocument,
    addDocument,
    updateDocument,
    deleteDocument,
  };
}
