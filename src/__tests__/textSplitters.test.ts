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

describe('CharacterTextSplitter', () => {
  test('splits on the default paragraph separator and respects chunkSize', async () => {
    const splitter = new CharacterTextSplitter({
      chunkSize: 12,
      chunkOverlap: 0,
    });
    const chunks = await splitter.splitText('aaaa\n\nbbbb\n\ncccc\n\ndddd');
    expect(chunks).toEqual(['aaaa\n\nbbbb', 'cccc\n\ndddd']);
  });

  test('applies chunkOverlap between consecutive chunks', async () => {
    const splitter = new CharacterTextSplitter({
      chunkSize: 10,
      chunkOverlap: 4,
    });
    const chunks = await splitter.splitText('aaaa\n\nbbbb\n\ncccc');
    expect(chunks).toEqual(['aaaa\n\nbbbb', 'bbbb\n\ncccc']);
  });
});

describe('RecursiveCharacterTextSplitter', () => {
  test('prefers paragraph, then line, then word boundaries', async () => {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 12,
      chunkOverlap: 0,
    });
    const chunks = await splitter.splitText(
      'para one\n\npara two line\nnext line here'
    );
    expect(chunks).toEqual([
      'para one',
      'para two',
      'line',
      'next line',
      'here',
    ]);
  });

  test('never produces a chunk longer than chunkSize when boundaries exist', async () => {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 25,
      chunkOverlap: 5,
    });
    const text = Array.from({ length: 30 }, (_, i) => `w${i}`).join(' ');
    const chunks = await splitter.splitText(text);
    chunks.forEach((c) => expect(c.length).toBeLessThanOrEqual(25));
  });
});

describe('TokenTextSplitter', () => {
  test('measures chunkSize in tokens rather than characters', async () => {
    const splitter = new TokenTextSplitter({ chunkSize: 3, chunkOverlap: 0 });
    // Each word is one token for the default cl100k_base encoding.
    const chunks = await splitter.splitText('one two three four five six');
    expect(chunks).toEqual(['one two three', ' four five six']);
  });

  test('applies token overlap', async () => {
    const splitter = new TokenTextSplitter({ chunkSize: 3, chunkOverlap: 1 });
    const chunks = await splitter.splitText('one two three four five');
    expect(chunks).toEqual(['one two three', ' three four five']);
  });
});

describe('MarkdownTextSplitter', () => {
  test('splits at Markdown headings before falling back to paragraphs', async () => {
    const splitter = new MarkdownTextSplitter({
      chunkSize: 30,
      chunkOverlap: 0,
    });
    const md =
      '# Title\n\nIntro text.\n\n## Section A\n\nBody of A.\n\n## Section B\n\nBody of B.';
    const chunks = await splitter.splitText(md);
    expect(chunks).toEqual([
      '# Title\n\nIntro text.',
      '## Section A\n\nBody of A.',
      '## Section B\n\nBody of B.',
    ]);
  });

  test('treats a closing code fence as a split boundary', async () => {
    const splitter = new MarkdownTextSplitter({
      chunkSize: 40,
      chunkOverlap: 0,
    });
    const md =
      '# Code\n\n```ts\nconst a = 1;\nconst b = 2;\n```\n\n# After\n\nDone.';
    const chunks = await splitter.splitText(md);
    expect(chunks).toEqual([
      '# Code\n\n```ts\nconst a = 1;\nconst b = 2;',
      '```\n\n# After\n\nDone.',
    ]);
  });

  test('keeps a whole document as one chunk when it fits', async () => {
    const splitter = new MarkdownTextSplitter({
      chunkSize: 60,
      chunkOverlap: 0,
    });
    const md =
      '# Code\n\n```ts\nconst a = 1;\nconst b = 2;\n```\n\n# After\n\nDone.';
    await expect(splitter.splitText(md)).resolves.toEqual([md]);
  });
});

describe('LatexTextSplitter', () => {
  test('splits at LaTeX sectioning commands', async () => {
    const splitter = new LatexTextSplitter({ chunkSize: 40, chunkOverlap: 0 });
    const tex =
      '\\section{Intro}\nSome intro text.\n\\section{Method}\nMethod text here.\n\\subsection{Detail}\nDetail text.';
    const chunks = await splitter.splitText(tex);
    expect(chunks).toEqual([
      '\\section{Intro}\nSome intro text.',
      '\\section{Method}\nMethod text here.',
      '\\subsection{Detail}\nDetail text.',
    ]);
  });

  test('falls back to paragraph and line boundaries inside a section', async () => {
    const splitter = new LatexTextSplitter({ chunkSize: 20, chunkOverlap: 0 });
    const tex = '\\section{S}\nfirst line of text\n\nsecond paragraph';
    const chunks = await splitter.splitText(tex);
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c) => expect(c.length).toBeLessThanOrEqual(20));
    expect(chunks.join('\n')).toContain('second paragraph');
  });
});
