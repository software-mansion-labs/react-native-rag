import { useEffect, useState, useCallback, useMemo } from 'react';
import { RAG, type RAGParams } from '../rag/rag';
import type { Message } from '../types/common';
import type { TextSplitter } from '../interfaces/textSplitter';

/**
 * Parameters for the useRAG hook.
 */
interface UseRAGParams extends RAGParams {
  /**
   * Controls automatic loading of the RAG system (vector store and LLM)
   * on mount or when changed from `true` to `false`. Set `true` to defer loading.
   */
  preventLoad?: boolean;
}

/**
 * A React hook for Retrieval Augmented Generation (RAG).
 * Manages RAG system lifecycle, loading, unloading, generation, and document storage.
 *
 * @param {UseRAGParams} params - RAG configuration (vectorStore, llm, preventLoad).
 * @returns {object} RAG state and functions.
 * @returns {string} return.response - Current generated text.
 * @returns {boolean} return.isReady - True if RAG system is loaded.
 * @returns {boolean} return.isGenerating - True if LLM is generating.
 * @returns {boolean} return.isStoring - True if document operation is in progress.
 * @returns {string | null} return.error - Last error message.
 * @returns {function} return.generate - Function to generate text with or without RAG augmentation.
 * @returns {function} return.interrupt - Function to stop current generation.
 * @returns {function} return.splitAddDocument - Splits and adds a document to the vector store.
 * @returns {function} return.addDocument - Adds a single document to the vector store.
 * @returns {function} return.updateDocument - Updates a document in the vector store.
 * @returns {function} return.deleteDocument - Deletes a document from the vector store.
 */
export function useRAG({ vectorStore, llm, preventLoad }: UseRAGParams) {
  const [response, setResponse] = useState<string>('');
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStoring, setIsStoring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const rag = useMemo(() => new RAG({ vectorStore, llm }), [vectorStore, llm]);

  useEffect(() => {
    setError(null);

    if (preventLoad) return;

    const load = async () => {
      try {
        await rag.load();
        setIsReady(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'RAG load error.');
      }
    };
    load();

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
   * @param {Message[] | string} input - Input messages or query.
   * @param {object} [options={}] - Generation options (augmentedGeneration, k, questionGenerator, promptGenerator).
   * @returns {Promise<string>} Complete generated string.
   * @throws {Error} If not ready or busy.
   */
  const generate = useCallback(
    async (
      input: Message[] | string,
      options: {
        augmentedGeneration?: boolean;
        k?: number;
        questionGenerator?: (messages: Message[]) => string;
        promptGenerator?: (
          messages: Message[],
          retrievedDocs: { content: string }[]
        ) => string;
      } = {}
    ): Promise<string> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isGenerating) throw new Error('RAG busy generating.');
      setResponse('');
      setError(null);
      try {
        setIsGenerating(true);
        return await rag.generate(input, {
          ...options,
          callback: (token: string) => {
            setResponse((prev) => prev + token);
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
   * @param {string} document - Document content.
   * @param {(chunks: string[]) => Record<string, any>[]} [metadataGenerator] - Optional metadata generator.
   * @param {TextSplitter} [textSplitter] - Optional text splitter.
   * @returns {Promise<string[]>} IDs of added chunks.
   * @throws {Error} If not ready or busy storing.
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
   * Adds a single document to the vector store.
   * @param {string} document - Document content.
   * @param {Record<string, any>} [metadata] - Optional metadata.
   * @returns {Promise<string>} ID of added document.
   * @throws {Error} If not ready or busy storing.
   */
  const addDocument = useCallback(
    async (
      document: string,
      metadata?: Record<string, any>
    ): Promise<string> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.addDocument(document, metadata);
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
   * Updates a document in the vector store.
   * @param {string} id - Document ID.
   * @param {string} [document] - New content.
   * @param {Record<string, any>} [metadata] - New metadata.
   * @returns {Promise<void>}
   * @throws {Error} If not ready or busy storing.
   */
  const updateDocument = useCallback(
    async (
      id: string,
      document?: string,
      metadata?: Record<string, any>
    ): Promise<void> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.updateDocument(id, document, metadata);
      } finally {
        setIsStoring(false);
      }
    },
    [rag, isReady, isStoring]
  );

  /**
   * Deletes a document from the vector store.
   * @param {string} id - Document ID.
   * @returns {Promise<void>}
   * @throws {Error} If not ready or busy storing.
   */
  const deleteDocument = useCallback(
    async (id: string): Promise<void> => {
      if (!isReady) throw new Error('RAG not ready.');
      if (isStoring) throw new Error('RAG busy storing.');
      setError(null);
      try {
        setIsStoring(true);
        return await rag.deleteDocument(id);
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
