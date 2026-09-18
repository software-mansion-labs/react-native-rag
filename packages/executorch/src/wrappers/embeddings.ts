import type { Embeddings } from 'react-native-rag';
import {
  createTextEmbedder,
  download,
  type TextEmbedder,
  type TextEmbedderModel,
} from 'react-native-executorch';

/**
 * Parameters for {@link ExecuTorchEmbeddings}.
 */
interface ExecuTorchEmbeddingsParams {
  /** Path or URL of the ExecuTorch embedding model (`.pte`). */
  modelPath: string;
  /** Path or URL of the tokenizer (`tokenizer.json`). */
  tokenizerPath: string;
  /** Optional prompt prepended to every input before embedding. */
  defaultPrompt?: string;
  /** Download progress callback (0-1). */
  onDownloadProgress?: (progress: number) => void;
}

/**
 * ExecuTorch-based implementation of {@link Embeddings} for React Native.
 */
export class ExecuTorchEmbeddings implements Embeddings {
  private embedder: TextEmbedder | null = null;
  private model: TextEmbedderModel;
  private onDownloadProgress: (progress: number) => void;

  /**
   * Creates a new ExecuTorch embeddings instance.
   * @param params - Parameters for the instance.
   * @param params.modelPath - Path or URL of the embedding model.
   * @param params.tokenizerPath - Path or URL of the tokenizer.
   * @param params.defaultPrompt - Optional prompt prepended to every input.
   * @param params.onDownloadProgress - Download progress callback (0-1).
   */
  constructor({
    modelPath,
    tokenizerPath,
    defaultPrompt,
    onDownloadProgress = () => {},
  }: ExecuTorchEmbeddingsParams) {
    this.model = { modelPath, tokenizerPath, defaultPrompt };
    this.onDownloadProgress = onDownloadProgress;
  }

  /**
   * Downloads (if needed) and loads the model and tokenizer via `react-native-executorch`.
   * @returns Promise that resolves to the same instance.
   */
  async load() {
    if (!this.embedder) {
      const resolved = await download(this.model, {
        onProgress: this.onDownloadProgress,
      });
      this.embedder = await createTextEmbedder(resolved);
    }
    return this;
  }

  /**
   * Unloads the underlying model and releases its native resources.
   */
  async unload() {
    this.embedder?.dispose();
    this.embedder = null;
  }

  /**
   * Generates an embedding vector for the given text.
   * @param text - Input string to embed.
   * @returns Promise that resolves to the embedding vector.
   */
  async embed(text: string): Promise<number[]> {
    if (!this.embedder) {
      throw new Error('Text embedder not loaded. Call load() first.');
    }
    return Array.from(await this.embedder.embed(text));
  }
}
