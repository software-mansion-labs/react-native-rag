import { LLMModule, type ChatConfig } from 'react-native-executorch';
import type { LLM, ResourceSource, Message } from 'react-native-rag';

interface ExecuTorchLLMParams {
  modelSource: ResourceSource;
  tokenizerSource: ResourceSource;
  tokenizerConfigSource: ResourceSource;

  onDownloadProgress?: (progress: number) => void;
  responseCallback?: (response: string) => void;
  messageHistoryCallback?: (messageHistory: Message[]) => void;

  chatConfig?: Partial<ChatConfig>;
}

export class ExecuTorchLLM implements LLM {
  private modelSource: ResourceSource;
  private tokenizerSource: ResourceSource;
  private tokenizerConfigSource: ResourceSource;
  private onDownloadProgress: (progress: number) => void;
  private responseCallback: (response: string) => void;
  private chatConfig: Partial<ChatConfig> | undefined;

  private isLoaded = false;

  constructor({
    modelSource,
    tokenizerSource,
    tokenizerConfigSource,
    onDownloadProgress = () => {},
    responseCallback = () => {},
    chatConfig,
  }: ExecuTorchLLMParams) {
    this.modelSource = modelSource;
    this.tokenizerSource = tokenizerSource;
    this.tokenizerConfigSource = tokenizerConfigSource;
    this.onDownloadProgress = onDownloadProgress;
    this.responseCallback = responseCallback;
    this.chatConfig = chatConfig;
  }

  async load() {
    if (!this.isLoaded) {
      await LLMModule.load({
        modelSource: this.modelSource,
        tokenizerSource: this.tokenizerSource,
        tokenizerConfigSource: this.tokenizerConfigSource,
        onDownloadProgressCallback: this.onDownloadProgress,
        responseCallback: this.responseCallback,
      });
      LLMModule.configure({
        chatConfig: this.chatConfig,
      });
      this.isLoaded = true;
    }
    return this;
  }

  async interrupt() {
    console.warn(
      'This function will call a synchronous interrupt on the LLMModule from React Native ExecuTorch. Awaiting this method will not guarantee completion. This may change in future versions to support async interrupt.'
    );
    LLMModule.interrupt();
  }

  async unload() {
    console.warn(
      'This function will call a synchronous unload on the LLMModule from React Native ExecuTorch. Awaiting this method will not guarantee completion. This may change in future versions to support async unload.'
    );
    LLMModule.delete();
  }

  async generate(messages: Message[], callback: (token: string) => void) {
    LLMModule.setTokenCallback({ tokenCallback: callback });
    return LLMModule.generate(messages);
  }
}
