import type { Embeddings, ResourceSource } from 'react-native-rag';
import { TextEmbeddingsModule } from 'react-native-executorch';

interface ExecuTorchEmbeddingsParams {
  modelSource: ResourceSource;
  tokenizerSource: ResourceSource;

  onDownloadProgress?: (progress: number) => void;
}

export class ExecuTorchEmbeddings implements Embeddings {
  private module: TextEmbeddingsModule;
  private modelSource: ResourceSource;
  private tokenizerSource: ResourceSource;
  private onDownloadProgress: (progress: number) => void;

  private isLoaded = false;

  constructor({
    modelSource,
    tokenizerSource,
    onDownloadProgress = () => {},
  }: ExecuTorchEmbeddingsParams) {
    this.module = new TextEmbeddingsModule();
    this.modelSource = modelSource;
    this.tokenizerSource = tokenizerSource;
    this.onDownloadProgress = onDownloadProgress;
  }

  async load() {
    if (!this.isLoaded) {
      await this.module.load(
        {
          modelSource: this.modelSource,
          tokenizerSource: this.tokenizerSource,
        },
        this.onDownloadProgress
      );
      this.isLoaded = true;
    }
    return this;
  }

  async unload() {
    console.warn(
      'This function will call a synchronous unload on the instance of TextEmbeddingsModule from React Native ExecuTorch. Awaiting this method will not guarantee completion. This may change in future versions to support async unload.'
    );
    this.module.delete();
  }

  async embed(text: string): Promise<number[]> {
    return Array.from(await this.module.forward(text));
  }
}
