import { uuidv4 } from '../utils/uuidv4';

const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe('uuidv4', () => {
  test('produces a canonical lowercase v4 UUID', () => {
    for (let i = 0; i < 200; i++) {
      expect(uuidv4()).toMatch(UUID_V4);
    }
  });

  test('forces the version and variant bits into all-zero bytes', () => {
    const spy = jest.spyOn(Math, 'random').mockReturnValue(0);
    try {
      expect(uuidv4()).toBe('00000000-0000-4000-8000-000000000000');
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
