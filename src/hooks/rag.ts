import { useEffect, useState, useCallback, useMemo } from 'react';
import { RAG } from '../rag/rag';
import type { Message, QueryResult, GetResult } from '../types/common';
import type { TextSplitter } from '../interfaces/textSplitter';
import type { VectorStore } from '../interfaces/vectorStore';
import type { LLM } from '../interfaces/llm';

/**
 * A React hook for Retrieval Augmented Generation (RAG).
 * Manages RAG system lifecycle, loading, unloading, generation, and document storage.
 *
 * @param params - RAG configuration.
 * @returns An object with state and RAG operations: `response`, `isReady`, `isGenerating`, `isStoring`, `error`, and functions `generate`, `interrupt`, `splitAddDocument`, `addDocument`, `updateDocument`, `deleteDocument`.
 *
 * @example
 * // Basic usage in a component
 * const { isReady, response, generate } = useRAG({ vectorStore, llm });
 * useEffect(() => {
 *   if (!isReady) return;
 *   generate({ input: 'What is RAG?' });
 * }, [isReady]);
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
   * @param params - Object containing:
   *   - `input`: User input as a string or array of messages.
   *   - `augmentedGeneration` (optional): Whether to use RAG augmentation (default: true).
   *   - `nResults` (optional): Number of documents to retrieve for augmentation (default: 3).
   *   - `predicate` (optional): Predicate to filter retrieved documents.
   *   - `questionGenerator` (optional): Function to generate a question from messages.
   *   - `promptGenerator` (optional): Function to generate a prompt from messages and retrieved documents.
   *   - `callback` (optional): Callback function for streaming tokens.
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
   * @param document - Document content.
   * @param metadataGenerator - Optional metadata generator.
   * @param textSplitter - Optional text splitter.
   * @returns IDs of added chunks.
   */
  const splitAddDocument = useCallback(
    async (
      document: string,
      metadataGenerator?: (chunks: string[]) => Record<string, any>[],
      textSplitter?: TextSplitter
    ): Promise<string[]> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.splitAddDocument(
          document,
          metadataGenerator,
          textSplitter
        );
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
   * Adds documents to the vector store.
   * @param params - Object containing:
   *   - `ids`: (optional) The IDs of the documents. If not provided, they will be auto-generated.
   *   - `documents`: Raw text content of the documents.
   *   - `embeddings` (optional): Embeddings for the documents.
   *   - `metadatas` (optional): Metadata associated with each document.
   * @returns A promise that resolves to the IDs of the newly added documents.
   * @throws Error if RAG is not ready or is currently storing.
   */
  const addDocument = useCallback(
    async (params: {
      ids?: string[];
      documents: string[];
      embeddings?: number[][];
      metadatas?: Record<string, any>[];
    }): Promise<string[]> => {
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
   * Updates documents in the vector store.
   * @param params - Object containing:
   *   - `ids`: The IDs of the documents to update.
   *   - `embeddings` (optional): New embeddings for the documents.
   *   - `documents` (optional): New content for the documents.
   *   - `metadatas` (optional): New metadata for the documents.
   * @returns A promise that resolves when the documents are updated.
   * @throws Error if RAG is not ready or is currently storing.
   */
  const updateDocument = useCallback(
    async (params: {
      ids: string[];
      embeddings?: number[][];
      documents?: string[];
      metadatas?: Record<string, any>[];
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
   * Deletes documents from the vector store.
   * @param params - Object containing:
   *   - `ids` (optional): List of document IDs to delete.
   *   - `predicate` (optional): Predicate to match documents for deletion.
   * @returns A promise that resolves when the documents are deleted.
   * @throws Error if RAG is not ready or is currently storing.
   */
  const deleteDocument = useCallback(
    async (params: {
      ids?: string[];
      predicate?: (value: GetResult) => boolean;
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
