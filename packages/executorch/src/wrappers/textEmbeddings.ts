import type { Embeddings, ResourceSource } from 'react-native-rag';
import { TextEmbeddingsModule } from 'react-native-executorch';

type ExecuTorchTextEmbeddingsParams = {
  modelSource: ResourceSource;
  tokenizerSource: ResourceSource;
  meanPooling?: boolean;
  onDownloadProgress?: (progress: number) => void;
};

export class ExecuTorchTextEmbeddings implements Embeddings {
  private modelSource: ResourceSource;
  private tokenizerSource: ResourceSource;
  private meanPooling: boolean | undefined;
  private onDownloadProgress: (progress: number) => void;
  private module: TextEmbeddingsModule;

  private isLoaded = false;

  constructor({
    modelSource,
    tokenizerSource,
    meanPooling,
    onDownloadProgress = () => {},
  }: ExecuTorchTextEmbeddingsParams) {
    this.modelSource = modelSource;
    this.tokenizerSource = tokenizerSource;
    this.meanPooling = meanPooling;
    this.onDownloadProgress = onDownloadProgress;

    this.module = new TextEmbeddingsModule();
  }

  async load() {
    if (!this.isLoaded) {
      await this.module.load(
        this.modelSource,
        this.tokenizerSource,
        this.meanPooling,
        this.onDownloadProgress
      );
      this.isLoaded = true;
    }
    return this;
  }

  async unload() {
    console.log(
      'React Native ExecuTorch TextEmbeddingsModule does not support unload'
    );
  }

  async embed(text: string): Promise<number[]> {
    return Array.from(await this.module.forward(text));
  }
}
