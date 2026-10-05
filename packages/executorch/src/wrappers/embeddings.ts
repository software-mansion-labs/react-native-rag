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
  /** The in-flight {@link load} call, if any, shared by overlapping callers. */
  private loading: Promise<void> | null = null;
  /** Whether the most recent call was {@link load} rather than {@link unload}. */
  private wantLoaded = false;
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
   * Overlapping calls share one load. An {@link unload} issued while the load
   * is in flight wins unless {@link load} is called again before it settles.
   * @returns Promise that resolves to the same instance.
   */
  async load() {
    this.wantLoaded = true;
    if (!this.embedder) {
      this.loading ??= this.doLoad().finally(() => {
        this.loading = null;
      });
      await this.loading;
    }
    return this;
  }

  private async doLoad(): Promise<void> {
    const resolved = await download(this.model, {
      onProgress: this.onDownloadProgress,
    });
    const embedder = await createTextEmbedder(resolved);
    // unload() was called while loading and nothing asked for the model since.
    if (!this.wantLoaded) {
      embedder.dispose();
      return;
    }
    this.embedder = embedder;
  }

  /**
   * Unloads the underlying model and releases its native resources. A load
   * still in flight is awaited first so that nothing is left behind.
   */
  async unload() {
    this.wantLoaded = false;
    if (this.loading) {
      // Its outcome belongs to the load() caller; a failed load leaves nothing to release.
      await this.loading.catch(() => {});
    }
    // load() was called again while we waited; the resources now belong to it.
    if (this.wantLoaded) return;
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
