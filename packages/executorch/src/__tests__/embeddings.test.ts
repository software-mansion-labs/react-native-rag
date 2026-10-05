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

  it('shares one native load between overlapping load() calls', async () => {
    const embeddings = new ExecuTorchEmbeddings(model);

    await Promise.all([embeddings.load(), embeddings.load()]);

    expect(mockExecutorch.download).toHaveBeenCalledTimes(1);
    expect(mockExecutorch.createTextEmbedder).toHaveBeenCalledTimes(1);
  });

  it('disposes a load that unload() overtook, and keeps one re-requested by load()', async () => {
    const deferDownload = () => {
      let release!: () => void;
      mockExecutorch.download.mockImplementationOnce(
        (source: unknown) =>
          new Promise((resolve) => {
            release = () => resolve(source);
          })
      );
      return () => release();
    };

    const embeddings = new ExecuTorchEmbeddings(model);
    let release = deferDownload();
    const loading = embeddings.load();
    const unloading = embeddings.unload();
    release();
    await Promise.all([loading, unloading]);
    expect(mockExecutorch.createTextEmbedder).toHaveBeenCalledTimes(1);
    expect(mockEmbedder.dispose).toHaveBeenCalledTimes(1);
    await expect(embeddings.embed('x')).rejects.toThrow(/load\(\)/);

    release = deferDownload();
    const first = embeddings.load();
    const reunloading = embeddings.unload();
    const second = embeddings.load();
    release();
    await Promise.all([first, reunloading, second]);
    expect(mockExecutorch.createTextEmbedder).toHaveBeenCalledTimes(2);
    expect(mockEmbedder.dispose).toHaveBeenCalledTimes(1);
    await expect(embeddings.embed('x')).resolves.toEqual([0.5, -1, 2]);
  });

  it('forwards the prompt matching the kind and none otherwise', async () => {
    const embeddings = new ExecuTorchEmbeddings({
      ...model,
      defaultPrompt: 'default: ',
      documentPrompt: 'passage: ',
      queryPrompt: 'query: ',
    });
    await embeddings.load();

    await embeddings.embed('a', { kind: 'document' });
    await embeddings.embed('b', { kind: 'query' });
    await embeddings.embed('c');

    // Without a per-call prompt the embedder falls back to the model's defaultPrompt.
    expect(mockExecutorch.createTextEmbedder).toHaveBeenCalledWith(
      expect.objectContaining({ defaultPrompt: 'default: ' })
    );
    expect(mockEmbedder.embed.mock.calls).toEqual([
      ['a', 'passage: '],
      ['b', 'query: '],
      ['c', undefined],
    ]);
  });

  it('lets an empty prompt disable the prefix for one side only', async () => {
    const embeddings = new ExecuTorchEmbeddings({
      ...model,
      defaultPrompt: 'query: ',
      documentPrompt: '',
    });
    await embeddings.load();

    await embeddings.embed('a', { kind: 'document' });
    await embeddings.embed('b', { kind: 'query' });

    expect(mockEmbedder.embed.mock.calls).toEqual([
      ['a', ''],
      ['b', undefined],
    ]);
  });

  it('throws when embedding before load', async () => {
    const embeddings = new ExecuTorchEmbeddings(model);
    await expect(embeddings.embed('x')).rejects.toThrow(/load\(\)/);
  });

  it('returns a plain number array from the Float32Array', async () => {
    const embeddings = new ExecuTorchEmbeddings(model);
    await embeddings.load();

    const vector = await embeddings.embed('hello');

    expect(mockEmbedder.embed).toHaveBeenCalledWith('hello', undefined);
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
