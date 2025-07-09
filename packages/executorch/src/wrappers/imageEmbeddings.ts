import type { Embeddings, ResourceSource } from 'react-native-rag';
import { ImageEmbeddingsModule } from 'react-native-executorch';

type ExecuTorchImageEmbeddingsParams = {
  modelSource: ResourceSource;
  onDownloadProgress?: (progress: number) => void;
};

export class ExecuTorchImageEmbeddings implements Embeddings {
  private modelSource: ResourceSource;
  private onDownloadProgress: (progress: number) => void;
  private module: ImageEmbeddingsModule;

  private isLoaded = false;

  constructor({
    modelSource,
    onDownloadProgress = () => {},
  }: ExecuTorchImageEmbeddingsParams) {
    this.modelSource = modelSource;
    this.onDownloadProgress = onDownloadProgress;

    this.module = new ImageEmbeddingsModule();
  }

  async load() {
    if (!this.isLoaded) {
      await this.module.load(this.modelSource, this.onDownloadProgress);
      this.isLoaded = true;
    }
    return this;
  }

  async unload() {
    console.log(
      'React Native ExecuTorch TextEmbeddingsModule does not support unload'
    );
  }

  async embed(imageSource: string): Promise<number[]> {
    return Array.from(await this.module.forward(imageSource));
  }
}
