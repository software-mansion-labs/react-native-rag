import type { LLM, Message } from 'react-native-rag';
import {
  createResourceScope,
  download,
  llm,
  nlp,
  wrapAsync,
  type LLMModel,
  type ResourceScope,
} from 'react-native-executorch';
import RNBlobUtil from 'react-native-blob-util';
import { scheduleOnRN } from 'react-native-worklets';

/**
 * System prompt used when the message history has none.
 * Mirrors the default that `react-native-executorch` 0.9 injected.
 */
export const DEFAULT_SYSTEM_PROMPT =
  "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user. Don't return too much text.";

/** Tokens kept free for the response when `generationConfig.maxNewTokens` is not set. */
const DEFAULT_RESPONSE_RESERVE_TOKENS = 512;

/**
 * Parameters for {@link ExecuTorchLLM}.
 */
interface ExecuTorchLLMParams extends LLMModel {
  /** Download progress callback (0-1). */
  onDownloadProgress?: (progress: number) => void;
  /** Generation configuration forwarded to ExecuTorch (temperature, max tokens, ...). */
  generationConfig?: llm.LLMGenerationConfig;
  /**
   * System prompt prepended when the message history contains no `system` message.
   * Defaults to {@link DEFAULT_SYSTEM_PROMPT}. Pass an empty string to disable.
   */
  systemPrompt?: string;
}

/**
 * Runs one full generation on the worklet runtime.
 * Resets the KV cache, prefills the rendered prompt and decodes until the model
 * emits one of its stop tokens or `interrupt()` is called. Tokens are forwarded
 * to the React Native thread via `scheduleOnRN`.
 */
function generateWorklet(
  runner: llm.LLMRunner,
  prompt: llm.Prompt,
  options: {
    readonly config: llm.LLMGenerationConfig;
    readonly stopTokens: readonly string[];
    readonly onToken: (token: string) => void;
  }
): string {
  'worklet';
  const { config, stopTokens, onToken } = options;

  let response = '';
  let stopped = false;
  runner.reset();
  runner.generate(prompt, config, (token: string) => {
    if (stopped) return;

    // The tokenizer config can name several terminal tokens (eos, eot, pad) and the
    // runner does not halt on all of them by itself, so stop explicitly on any of them.
    if (stopTokens.includes(token)) {
      stopped = true;
      runner.stop();
      return;
    }

    response += token;
    scheduleOnRN(onToken, token);
  });
  return response;
}

function countTokensWorklet(tokenizer: nlp.Tokenizer, text: string): number {
  'worklet';
  return tokenizer.encode(text).length;
}

/**
 * ExecuTorch-based implementation of {@link LLM} for React Native.
 *
 * Each {@link generate} call is stateless: the full message history is rendered
 * through the model's chat template and fed to the runner from a fresh KV cache.
 * When the history does not fit the model's context window, the oldest turns are dropped.
 * The model itself is loaded once in {@link load} and kept in memory.
 */
export class ExecuTorchLLM implements LLM {
  private scope: ResourceScope | null = null;
  private runner: llm.LLMRunner | null = null;
  private preprocessor: llm.ChatPreprocessor | null = null;
  private tokenizer: nlp.Tokenizer | null = null;
  private stopTokens: readonly string[] = [];
  /** The in-flight {@link load} call, if any, shared by overlapping callers. */
  private loading: Promise<void> | null = null;
  /** Whether the most recent call was {@link load} rather than {@link unload}. */
  private wantLoaded = false;
  /** The in-flight {@link generate} call, if any. */
  private pending: Promise<string> | null = null;
  /** Set by {@link interrupt} so a request that lands before the native generate starts is honoured. */
  private interruptRequested = false;

  private model: LLMModel;
  private onDownloadProgress: (progress: number) => void;
  private generationConfig: llm.LLMGenerationConfig;
  private systemPrompt: string;

  /**
   * Creates a new ExecuTorch LLM instance.
   * @param params - Parameters for the instance.
   * @param params.modelPath - Path or URL of the LLM model (`.pte`).
   * @param params.tokenizerPath - Path or URL of the tokenizer (`tokenizer.json`).
   * @param params.tokenizerConfigPath - Path or URL of the tokenizer config (`tokenizer_config.json`).
   * @param params.onDownloadProgress - Download progress callback (0-1).
   * @param params.generationConfig - Generation configuration forwarded to ExecuTorch.
   * @param params.systemPrompt - System prompt used when the history has none. Empty string disables it.
   */
  constructor({
    onDownloadProgress = () => {},
    generationConfig = {},
    systemPrompt = DEFAULT_SYSTEM_PROMPT,
    ...model
  }: ExecuTorchLLMParams) {
    this.model = model;
    this.onDownloadProgress = onDownloadProgress;
    this.generationConfig = generationConfig;
    this.systemPrompt = systemPrompt;
  }

  /**
   * Downloads (if needed) and loads the model, tokenizer and chat template via `react-native-executorch`.
   * Overlapping calls share one load. An {@link unload} issued while the load
   * is in flight wins unless {@link load} is called again before it settles.
   * @returns Promise that resolves to the same instance.
   */
  async load() {
    this.wantLoaded = true;
    if (!this.runner) {
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

    const tokenizerConfigStr = await RNBlobUtil.fs.readFile(
      resolved.tokenizerConfigPath,
      'utf8'
    );
    const { chatTemplate, stopTokens } = llm.parseTokenizerConfig(
      JSON.parse(tokenizerConfigStr)
    );

    // Everything native is owned by the scope, so a failure halfway releases what was already created.
    const scope = createResourceScope();
    try {
      const preprocessor = scope.track(
        llm.createChatPreprocessor({
          chatTemplate,
          modalities: resolved.modalities,
          preprocessorConfig: resolved.preprocessorConfig,
        })
      );
      const tokenizer = scope.track(
        await wrapAsync(nlp.loadTokenizer)(resolved.tokenizerPath)
      );
      const runner = scope.track(
        await wrapAsync(llm.createLLMRunner)(
          resolved.modelPath,
          resolved.tokenizerPath,
          resolved.modalities
        )
      );

      // unload() was called while loading and nothing asked for the model since.
      if (!this.wantLoaded) {
        scope.dispose();
        return;
      }

      this.scope = scope;
      this.stopTokens = stopTokens;
      this.preprocessor = preprocessor;
      this.tokenizer = tokenizer;
      this.runner = runner;
    } catch (error) {
      scope.dispose();
      throw error;
    }
  }

  /**
   * Interrupts the current generation and resolves once it has settled. The
   * pending {@link generate} promise resolves with the tokens produced so far.
   * Does nothing when no generation is running.
   */
  async interrupt() {
    const { pending } = this;
    if (!pending) return;
    this.interruptRequested = true;
    this.runner?.stop();
    // The caller of generate() receives its outcome; here only its completion matters.
    await pending.catch(() => {});
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
    this.scope?.dispose();
    this.scope = null;
    this.runner = null;
    this.preprocessor = null;
    this.tokenizer = null;
  }

  /**
   * Drops the oldest non-system messages until the rendered prompt leaves room
   * for the response in the model's context window. System messages and the
   * last message are always kept.
   */
  private async fitToContext(
    messages: Message[],
    runner: llm.LLMRunner,
    preprocessor: llm.ChatPreprocessor,
    tokenizer: nlp.Tokenizer
  ): Promise<Message[]> {
    const { maxSeqLen } = runner.getKVCacheState();
    const reserve = Math.min(
      this.generationConfig.maxNewTokens ?? DEFAULT_RESPONSE_RESERVE_TOKENS,
      Math.floor(maxSeqLen / 2)
    );
    const budget = maxSeqLen - reserve;
    const countTokens = wrapAsync(countTokensWorklet);

    let fitted = messages;
    for (;;) {
      const { text } = preprocessor.render(fitted, { addGenPrompt: true });
      if ((await countTokens(tokenizer, text)) <= budget) return fitted;

      const last = fitted.length - 1;
      const oldest = fitted.findIndex(
        (message, index) => message.role !== 'system' && index < last
      );
      if (oldest === -1) return fitted;

      // Drop the oldest message, plus the assistant replies that would be left dangling at the start.
      let end = oldest + 1;
      while (end < last && fitted[end]!.role === 'assistant') end++;
      fitted = [...fitted.slice(0, oldest), ...fitted.slice(end)];
    }
  }

  /**
   * Generates a completion from a list of messages, streaming tokens to `callback`.
   * @param messages - Conversation history for the model.
   * @param callback - Token-level streaming callback.
   * @returns Promise that resolves to the full generated string.
   */
  async generate(messages: Message[], callback: (token: string) => void) {
    const { runner, preprocessor, tokenizer } = this;
    if (!runner || !preprocessor || !tokenizer) {
      throw new Error('LLM not loaded. Call load() first.');
    }
    // The preprocessor and the KV cache are shared, so turns cannot overlap.
    if (this.pending) {
      throw new Error('LLM is already generating. Call interrupt() first.');
    }
    this.interruptRequested = false;

    const pending = this.run(
      messages,
      callback,
      runner,
      preprocessor,
      tokenizer
    );
    this.pending = pending;
    try {
      return await pending;
    } finally {
      this.pending = null;
    }
  }

  private async run(
    messages: Message[],
    callback: (token: string) => void,
    runner: llm.LLMRunner,
    preprocessor: llm.ChatPreprocessor,
    tokenizer: nlp.Tokenizer
  ): Promise<string> {
    const hasSystemMessage = messages.some(({ role }) => role === 'system');
    const history: Message[] =
      this.systemPrompt && !hasSystemMessage
        ? [{ role: 'system', content: this.systemPrompt }, ...messages]
        : messages;

    try {
      const fitted = await this.fitToContext(
        history,
        runner,
        preprocessor,
        tokenizer
      );
      const prompt = preprocessor.process(fitted, fitted.length, {
        addGenPrompt: true,
      });
      // ExecuTorch clears its stop flag when the native generate starts, so an
      // interrupt() that landed during the steps above would otherwise be lost.
      if (this.interruptRequested) return '';
      return await wrapAsync(generateWorklet)(runner, prompt, {
        config: this.generationConfig,
        stopTokens: this.stopTokens,
        onToken: callback,
      });
    } catch (error) {
      // Leave a clean KV cache behind. The reset must never replace the error it is unwinding.
      try {
        runner.reset();
      } catch {}
      throw error;
    } finally {
      preprocessor.clear();
    }
  }
}
