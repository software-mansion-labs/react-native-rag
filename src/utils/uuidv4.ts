const min = 0;
const max = 256;
const RANDOM_LENGTH = 16;

/**
 * Generates an array of cryptographically strong pseudo-random numbers.
 * Each number is an 8-bit unsigned integer (0-255).
 * @returns An array of 16 random bytes.
 */
const rng = () => {
  const result = new Array<number>(RANDOM_LENGTH);
  for (let j = 0; j < RANDOM_LENGTH; j++) {
    // eslint-disable-next-line no-bitwise
    result[j] = 0xff & (Math.random() * (max - min) + min);
  }
  return result;
};

/**
 * A lookup table to convert a single byte (0-255) to its two-character hexadecimal string representation.
 */
const byteToHex: string[] = [];
for (let i = 0; i < 256; i++) {
  byteToHex[i] = (i + 0x100).toString(16).slice(1);
}

/**
 * Generates a compliant Version 4 UUID (Universally Unique Identifier).
 * @returns A string representing a UUID v4.
 */
export function uuidv4() {
  const buf = rng();
  // eslint-disable-next-line no-bitwise
  buf[6] = (buf[6]! & 0x0f) | 0x40;
  // eslint-disable-next-line no-bitwise
  buf[8] = (buf[8]! & 0x3f) | 0x80;

  return (
    byteToHex[buf[0]!]! +
    byteToHex[buf[1]!]! +
    byteToHex[buf[2]!]! +
    byteToHex[buf[3]!]! +
    '-' +
    byteToHex[buf[4]!]! +
    byteToHex[buf[5]!]! +
    '-' +
    byteToHex[buf[6]!]! +
    byteToHex[buf[7]!]! +
    '-' +
    byteToHex[buf[8]!]! +
    byteToHex[buf[9]!]! +
    '-' +
    byteToHex[buf[10]!]! +
    byteToHex[buf[11]!]! +
    byteToHex[buf[12]!]! +
    byteToHex[buf[13]!]! +
    byteToHex[buf[14]!]! +
    byteToHex[buf[15]!]!
  );
}
