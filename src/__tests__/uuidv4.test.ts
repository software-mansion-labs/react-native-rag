import { uuidv4 } from '../utils/uuidv4';

const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe('uuidv4', () => {
  test('produces a canonical lowercase v4 UUID', () => {
    for (let i = 0; i < 200; i++) {
      expect(uuidv4()).toMatch(UUID_V4);
    }
  });

  test('sets the version nibble to 4 and the variant bits to 10xx', () => {
    const id = uuidv4();
    expect(id.charAt(14)).toBe('4');
    expect(['8', '9', 'a', 'b']).toContain(id.charAt(19));
  });

  test('is unique across many calls', () => {
    const ids = new Set<string>();
    for (let i = 0; i < 5000; i++) {
      ids.add(uuidv4());
    }
    expect(ids.size).toBe(5000);
  });

  test('derives every byte from Math.random', () => {
    const spy = jest.spyOn(Math, 'random').mockReturnValue(0);
    try {
      // All-zero bytes, with version and variant forced in.
      expect(uuidv4()).toBe('00000000-0000-4000-8000-000000000000');
      expect(spy).toHaveBeenCalledTimes(16);
    } finally {
      spy.mockRestore();
    }
  });

  test('masks random values into a single byte', () => {
    const spy = jest.spyOn(Math, 'random').mockReturnValue(0.999999);
    try {
      // 0.999999 * 256 = 255.99 -> byte 0xff everywhere before masking.
      expect(uuidv4()).toBe('ffffffff-ffff-4fff-bfff-ffffffffffff');
    } finally {
      spy.mockRestore();
    }
  });
});
