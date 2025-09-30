import { LLMModule, type ChatConfig } from 'react-native-executorch';
import type { LLM, ResourceSource, Message } from 'react-native-rag';

/**
 * Parameters for {@link ExecuTorchLLM}.
 */
interface ExecuTorchLLMParams {
  /** Source of the LLM model. */
  modelSource: ResourceSource;
  /** Source of the tokenizer model. */
  tokenizerSource: ResourceSource;
  /** Source of the tokenizer config. */
  tokenizerConfigSource: ResourceSource;

  /** Download progress callback (0-1). */
  onDownloadProgress?: (progress: number) => void;
  /** Callback invoked with final full response string. */
  responseCallback?: (response: string) => void;
  /** Reserved: callback for message history changes (not wired currently). */
  messageHistoryCallback?: (messageHistory: Message[]) => void;

  /** Chat configuration forwarded to ExecuTorch. */
  chatConfig?: Partial<ChatConfig>;
}

/**
 * ExecuTorch-based implementation of {@link LLM} for React Native.
 */
export class ExecuTorchLLM implements LLM {
  private module: LLMModule;

  private modelSource: ResourceSource;
  private tokenizerSource: ResourceSource;
  private tokenizerConfigSource: ResourceSource;
  private onDownloadProgress: (progress: number) => void;
  private chatConfig: Partial<ChatConfig> | undefined;

  private isLoaded = false;

  /**
   * Creates a new ExecuTorch LLM instance.
   * @param params - Parameters for the instance.
   * @param params.modelSource - Source of the LLM model.
   * @param params.tokenizerSource - Source of the tokenizer.
   * @param params.tokenizerConfigSource - Source of the tokenizer config.
   * @param params.onDownloadProgress - Download progress callback (0-1).
   * @param params.responseCallback - Callback invoked with final full response string.
   * @param params.chatConfig - Chat configuration forwarded to ExecuTorch.
   */
  constructor({
    modelSource,
    tokenizerSource,
    tokenizerConfigSource,
    onDownloadProgress = () => {},
    responseCallback = () => {},
    chatConfig,
  }: ExecuTorchLLMParams) {
    this.module = new LLMModule({
      responseCallback: responseCallback,
    });
    this.modelSource = modelSource;
    this.tokenizerSource = tokenizerSource;
    this.tokenizerConfigSource = tokenizerConfigSource;
    this.onDownloadProgress = onDownloadProgress;
    this.chatConfig = chatConfig;
  }

  /**
   * Loads the model and config via `react-native-executorch`, and applies configuration.
   * @returns Promise that resolves to the same instance.
   */
  async load() {
    if (!this.isLoaded) {
      await this.module.load(
        {
          modelSource: this.modelSource,
          tokenizerSource: this.tokenizerSource,
          tokenizerConfigSource: this.tokenizerConfigSource,
        },
        this.onDownloadProgress
      );
      this.module.configure({
        chatConfig: this.chatConfig,
      });
      this.isLoaded = true;
    }
    return this;
  }

  /**
   * Interrupts current generation.
   * Note: current ExecuTorch interrupt is synchronous.
   * Awaiting this method will not guarantee completion.
   */
  async interrupt() {
    console.warn(
      'This function will call a synchronous interrupt on the instance of LLMModule from React Native ExecuTorch. Awaiting this method will not guarantee completion. This may change in future versions to support async interrupt.'
    );
    this.module.interrupt();
  }

  /**
   * Unloads the underlying module.
   * Note: current ExecuTorch unload is synchronous.
   * Awaiting this method will not guarantee completion.
   */
  async unload() {
    console.warn(
      'This function will call a synchronous unload on the instance of LLMModule from React Native ExecuTorch. Awaiting this method will not guarantee completion. This may change in future versions to support async unload.'
    );
    this.module.delete();
    this.isLoaded = false;
  }

  /**
   * Generates a completion from a list of messages, streaming tokens to `callback`.
   * @param messages - Conversation history for the model.
   * @param callback - Token-level streaming callback.
   * @returns Promise that resolves to the full generated string.
   */
  async generate(messages: Message[], callback: (token: string) => void) {
    this.module.setTokenCallback({ tokenCallback: callback });
    return this.module.generate(messages);
  }
}
