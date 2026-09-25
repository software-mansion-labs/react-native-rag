import { ExecuTorchEmbeddings } from '../wrappers/embeddings';

jest.mock('react-native-executorch', () => {
  const embedder = {
    embed: jest.fn(async () => new Float32Array([0.5, -1, 2])),
    dispose: jest.fn(),
  };
  return {
    __embedder: embedder,
    download: jest.fn(async (source: unknown, options?: any) => {
      options?.onProgress?.(0.5);
      options?.onProgress?.(1);
      return source;
    }),
    createTextEmbedder: jest.fn(async () => embedder),
  };
});

const mockExecutorch = jest.requireMock('react-native-executorch');
const mockEmbedder = mockExecutorch.__embedder;

const model = {
  modelPath: 'https://example.com/model.pte',
  tokenizerPath: 'https://example.com/tokenizer.json',
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe('ExecuTorchEmbeddings', () => {
  it('downloads with progress and creates the embedder once', async () => {
    const onDownloadProgress = jest.fn();
    const embeddings = new ExecuTorchEmbeddings({
      ...model,
      defaultPrompt: 'query: ',
      onDownloadProgress,
    });

    await embeddings.load();
    await embeddings.load();

    expect(mockExecutorch.download).toHaveBeenCalledTimes(1);
    expect(mockExecutorch.download).toHaveBeenCalledWith(
      { ...model, defaultPrompt: 'query: ' },
      expect.objectContaining({ onProgress: onDownloadProgress })
    );
    expect(onDownloadProgress).toHaveBeenLastCalledWith(1);
    expect(mockExecutorch.createTextEmbedder).toHaveBeenCalledTimes(1);
  });

  it('throws when embedding before load', async () => {
    const embeddings = new ExecuTorchEmbeddings(model);
    await expect(embeddings.embed('x')).rejects.toThrow(/load\(\)/);
  });

  it('returns a plain number array from the Float32Array', async () => {
    const embeddings = new ExecuTorchEmbeddings(model);
    await embeddings.load();

    const vector = await embeddings.embed('hello');

    expect(mockEmbedder.embed).toHaveBeenCalledWith('hello');
    expect(Array.isArray(vector)).toBe(true);
    expect(vector).toEqual([0.5, -1, 2]);
  });

  it('disposes on unload and requires load again afterwards', async () => {
    const embeddings = new ExecuTorchEmbeddings(model);
    await embeddings.load();
    await embeddings.unload();

    expect(mockEmbedder.dispose).toHaveBeenCalledTimes(1);
    await expect(embeddings.embed('x')).rejects.toThrow(/load\(\)/);
  });
});
