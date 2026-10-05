import {
  CharacterTextSplitter,
  LatexTextSplitter,
  MarkdownTextSplitter,
  RecursiveCharacterTextSplitter,
  TokenTextSplitter,
} from '../text-splitters/langchain';
import type { TextSplitter } from '../interfaces/textSplitter';

// TokenTextSplitter fetches tiktoken encodings over the network. Replace the
// encoder with a deterministic whitespace tokenizer: one token per word, with
// the leading whitespace attached, so chunk sizes are counted in words.
jest.mock('@langchain/core/utils/tiktoken', () => {
  const table: string[] = [];
  const idOf = (t: string) => {
    let i = table.indexOf(t);
    if (i === -1) i = table.push(t) - 1;
    return i;
  };
  return {
    getEncoding: jest.fn(async () => ({
      encode: (text: string) => (text.match(/\s*\S+/g) ?? []).map(idOf),
      decode: (ids: number[]) => ids.map((i) => table[i]).join(''),
    })),
  };
});

const SPLITTERS: [
  string,
  (p: { chunkSize: number; chunkOverlap: number }) => TextSplitter,
][] = [
  ['CharacterTextSplitter', (p) => new CharacterTextSplitter(p)],
  [
    'RecursiveCharacterTextSplitter',
    (p) => new RecursiveCharacterTextSplitter(p),
  ],
  ['TokenTextSplitter', (p) => new TokenTextSplitter(p)],
  ['MarkdownTextSplitter', (p) => new MarkdownTextSplitter(p)],
  ['LatexTextSplitter', (p) => new LatexTextSplitter(p)],
];

describe.each(SPLITTERS)('%s (shared TextSplitter contract)', (_name, make) => {
  test('implements splitText returning a promise of strings', async () => {
    const splitter = make({ chunkSize: 100, chunkOverlap: 0 });
    const chunks = await splitter.splitText('hello world');
    expect(Array.isArray(chunks)).toBe(true);
    chunks.forEach((c) => expect(typeof c).toBe('string'));
  });

  test('keeps a short text as a single chunk', async () => {
    const splitter = make({ chunkSize: 100, chunkOverlap: 0 });
    await expect(splitter.splitText('short text')).resolves.toEqual([
      'short text',
    ]);
  });

  test('returns no chunks for empty input', async () => {
    const splitter = make({ chunkSize: 100, chunkOverlap: 0 });
    await expect(splitter.splitText('')).resolves.toEqual([]);
  });

  test('splits a long text into several chunks that cover the content', async () => {
    const splitter = make({ chunkSize: 20, chunkOverlap: 0 });
    const words = Array.from({ length: 40 }, (_, i) => `word${i}`);
    // Paragraph-separated so every splitter, including the fixed-separator
    // CharacterTextSplitter, has boundaries to cut on.
    const text = words.join('\n\n');
    const chunks = await splitter.splitText(text);
    expect(chunks.length).toBeGreaterThan(1);
    // Every word survives in at least one chunk.
    const joined = chunks.join(' ');
    words.forEach((w) => expect(joined).toContain(w));
  });

  test('rejects an overlap larger than the chunk size', () => {
    expect(() => make({ chunkSize: 10, chunkOverlap: 20 })).toThrow();
  });
});
