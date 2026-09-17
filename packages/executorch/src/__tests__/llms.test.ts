import type { Message } from 'react-native-rag';
import { ExecuTorchLLM } from '../wrappers/llms';

jest.mock('react-native-executorch', () => {
  const runner = {
    reset: jest.fn(),
    stop: jest.fn(),
    dispose: jest.fn(),
    generate: jest.fn(
      (_prompt: string, _config: unknown, onToken: (token: string) => void) => {
        ['Hel', 'lo', '<eos>'].forEach(onToken);
        return {};
      }
    ),
  };
  const preprocessor = {
    process: jest.fn((history: unknown) => JSON.stringify(history)),
    clear: jest.fn(),
    dispose: jest.fn(),
  };
  return {
    __runner: runner,
    __preprocessor: preprocessor,
    download: jest.fn(async (source: unknown, options?: any) => {
      options?.onProgress?.(1);
      return source;
    }),
    wrapAsync:
      (fn: (...args: any[]) => any) =>
      async (...args: any[]) =>
        fn(...args),
    llm: {
      parseTokenizerConfig: jest.fn(() => ({
        chatTemplate: 'TEMPLATE',
        eosToken: '<eos>',
      })),
      createChatPreprocessor: jest.fn(() => preprocessor),
      createLLMRunner: jest.fn(() => runner),
    },
  };
});
jest.mock('react-native-blob-util', () => ({
  fs: { readFile: jest.fn(async () => '{"chat_template":"TEMPLATE"}') },
}));
jest.mock('react-native-worklets', () => ({
  scheduleOnRN: (fn: (...args: any[]) => void, ...args: any[]) => fn(...args),
}));

const mockExecutorch = jest.requireMock('react-native-executorch');
const mockRunner = mockExecutorch.__runner;
const mockPreprocessor = mockExecutorch.__preprocessor;

const model = {
  modelPath: 'https://example.com/model.pte',
  tokenizerPath: 'https://example.com/tokenizer.json',
  tokenizerConfigPath: 'https://example.com/tokenizer_config.json',
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe('ExecuTorchLLM', () => {
  it('downloads, parses the tokenizer config and creates the runner once', async () => {
    const onDownloadProgress = jest.fn();
    const llm = new ExecuTorchLLM({ ...model, onDownloadProgress });

    await llm.load();
    await llm.load();

    expect(mockExecutorch.download).toHaveBeenCalledTimes(1);
    expect(mockExecutorch.download).toHaveBeenCalledWith(
      model,
      expect.objectContaining({ onProgress: onDownloadProgress })
    );
    expect(onDownloadProgress).toHaveBeenCalledWith(1);
    expect(mockExecutorch.llm.createChatPreprocessor).toHaveBeenCalledWith(
      expect.objectContaining({ chatTemplate: 'TEMPLATE' })
    );
    expect(mockExecutorch.llm.createLLMRunner).toHaveBeenCalledTimes(1);
    expect(mockExecutorch.llm.createLLMRunner).toHaveBeenCalledWith(
      model.modelPath,
      model.tokenizerPath,
      undefined
    );
  });

  it('throws when generating before load', async () => {
    const llm = new ExecuTorchLLM(model);
    await expect(llm.generate([], () => {})).rejects.toThrow(/load\(\)/);
  });

  it('renders the full history, streams tokens without EOS and returns the text', async () => {
    const llm = new ExecuTorchLLM({
      ...model,
      generationConfig: { temperature: 0.2 },
    });
    await llm.load();

    const messages: Message[] = [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi' },
    ];
    const tokens: string[] = [];
    const result = await llm.generate(messages, (t) => tokens.push(t));

    expect(mockPreprocessor.process).toHaveBeenCalledWith(messages, 2, {
      addGenPrompt: true,
    });
    expect(mockRunner.reset).toHaveBeenCalledTimes(1);
    expect(mockRunner.generate).toHaveBeenCalledWith(
      JSON.stringify(messages),
      { echo: false, temperature: 0.2 },
      expect.any(Function)
    );
    expect(tokens).toEqual(['Hel', 'lo']);
    expect(result).toBe('Hello');
    expect(mockPreprocessor.clear).toHaveBeenCalledTimes(1);
  });

  it('is stateless across turns and keeps augmented RAG turns intact', async () => {
    const llm = new ExecuTorchLLM(model);
    await llm.load();

    const turn1: Message[] = [{ role: 'user', content: 'Q1' }];
    await llm.generate(turn1, () => {});

    const turn2: Message[] = [
      { role: 'user', content: 'Q1' },
      { role: 'assistant', content: 'A1' },
      { role: 'user', content: 'Q2' },
      { role: 'user', content: 'Message: Q2\nContext: doc' },
    ];
    await llm.generate(turn2, () => {});

    expect(mockPreprocessor.process).toHaveBeenNthCalledWith(1, turn1, 1, {
      addGenPrompt: true,
    });
    expect(mockPreprocessor.process).toHaveBeenNthCalledWith(2, turn2, 4, {
      addGenPrompt: true,
    });
    expect(mockRunner.reset).toHaveBeenCalledTimes(2);
    expect(mockExecutorch.llm.createLLMRunner).toHaveBeenCalledTimes(1);
  });

  it('clears the preprocessor even when generation fails', async () => {
    const llm = new ExecuTorchLLM(model);
    await llm.load();
    mockRunner.generate.mockImplementationOnce(() => {
      throw new Error('boom');
    });

    await expect(
      llm.generate([{ role: 'user', content: 'x' }], () => {})
    ).rejects.toThrow('boom');
    expect(mockPreprocessor.clear).toHaveBeenCalledTimes(1);
  });

  it('maps interrupt to runner.stop and unload to dispose', async () => {
    const llm = new ExecuTorchLLM(model);
    await llm.interrupt();
    expect(mockRunner.stop).not.toHaveBeenCalled();

    await llm.load();
    await llm.interrupt();
    expect(mockRunner.stop).toHaveBeenCalledTimes(1);

    await llm.unload();
    expect(mockPreprocessor.dispose).toHaveBeenCalledTimes(1);
    expect(mockRunner.dispose).toHaveBeenCalledTimes(1);
    await expect(
      llm.generate([{ role: 'user', content: 'x' }], () => {})
    ).rejects.toThrow(/load\(\)/);
  });
});
