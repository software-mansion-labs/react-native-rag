import type { Embeddings, ResourceSource } from 'react-native-rag';
import { TextEmbeddingsModule } from 'react-native-executorch';

interface ExecuTorchEmbeddingsParams {
  modelSource: ResourceSource;
  tokenizerSource: ResourceSource;

  onDownloadProgress?: (progress: number) => void;
}

export class ExecuTorchEmbeddings implements Embeddings {
  private modelSource: ResourceSource;
  private tokenizerSource: ResourceSource;
  private onDownloadProgress: (progress: number) => void;

  private isLoaded = false;

  constructor({
    modelSource,
    tokenizerSource,
    onDownloadProgress = () => {},
  }: ExecuTorchEmbeddingsParams) {
    this.modelSource = modelSource;
    this.tokenizerSource = tokenizerSource;
    this.onDownloadProgress = onDownloadProgress;
  }

  async load() {
    if (!this.isLoaded) {
      TextEmbeddingsModule.onDownloadProgress(this.onDownloadProgress);
      await TextEmbeddingsModule.load(this.modelSource, this.tokenizerSource);
      this.isLoaded = true;
    }
    return this;
  }

  delete() {
    console.log('TextEmbeddingsModule does not support delete');
  }

  async embed(text: string): Promise<number[]> {
    return TextEmbeddingsModule.forward(text);
  }
}
