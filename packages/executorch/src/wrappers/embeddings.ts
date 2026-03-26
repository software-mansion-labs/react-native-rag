import type { Embeddings, ResourceSource } from 'react-native-rag';
import { TextEmbeddingsModule } from 'react-native-executorch';

/**
 * Parameters for {@link ExecuTorchEmbeddings}.
 */
interface ExecuTorchEmbeddingsParams {
  /** Source of the ExecuTorch embedding model. */
  modelSource: ResourceSource;
  /** Source of the tokenizer model. */
  tokenizerSource: ResourceSource;
  /** Download progress callback (0-1). */
  onDownloadProgress?: (progress: number) => void;
}

/**
 * ExecuTorch-based implementation of {@link Embeddings} for React Native.
 */
export class ExecuTorchEmbeddings implements Embeddings {
  private module: TextEmbeddingsModule | null = null;
  private modelSource: ResourceSource;
  private tokenizerSource: ResourceSource;
  private onDownloadProgress: (progress: number) => void;

  private isLoaded = false;

  /**
   * Creates a new ExecuTorch embeddings instance.
   * @param params - Parameters for the instance.
   * @param params.modelSource - Source of the embedding model.
   * @param params.tokenizerSource - Source of the tokenizer.
   * @param params.onDownloadProgress - Download progress callback (0-1).
   */
  constructor({
    modelSource,
    tokenizerSource,
    onDownloadProgress = () => {},
  }: ExecuTorchEmbeddingsParams) {
    this.modelSource = modelSource;
    this.tokenizerSource = tokenizerSource;
    this.onDownloadProgress = onDownloadProgress;
  }

  /**
   * Loads model and tokenizer via `react-native-executorch`.
   * @returns Promise that resolves to the same instance.
   */
  async load() {
    if (!this.isLoaded) {
      this.module = await TextEmbeddingsModule.fromCustomModel(
        this.modelSource,
        this.tokenizerSource,
        this.onDownloadProgress
      );
      this.isLoaded = true;
    }
    return this;
  }

  /**
   * Unloads the underlying module.
   * Note: current ExecuTorch unload is synchronous.
   * Awaiting this method will not guarantee completion.
   */
  async unload() {
    console.warn(
      'This function will call a synchronous unload on the instance of TextEmbeddingsModule from React Native ExecuTorch. Awaiting this method will not guarantee completion. This may change in future versions to support async unload.'
    );
    this.module?.delete();
    this.module = null;
    this.isLoaded = false;
  }

  /**
   * Generates an embedding vector for the given text.
   * @param text - Input string to embed.
   * @returns Promise that resolves to the embedding vector.
   */
  async embed(text: string): Promise<number[]> {
    if (!this.module) {
      throw new Error('TextEmbeddingsModule not loaded. Call load() first.');
    }
    return Array.from(await this.module.forward(text));
  }
}
