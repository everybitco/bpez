# bpez

BPEZ is a compression experiment where i wasnt trying to invent any thing new but wanted to beat industry standard Gzip by making the data a bit easier to compress before Gzip even touches it.

### Compression

Standard Deflate is a generalist. It looks for repeated strings and packs bits, but it struggles with noisy data. My hypothesis was that if i use BPE as a pre-processor it could collapse the most frequent patterns into single bytes. I know people have done this before, but this is for my own understanding, and this would effectively lower the entropy of the stream, allowing the Deflate stage to pack the data tighter than it normally could. The results below show the hypothesis was correct, and i successfully shrank the `1GB enwik9` dataset to `294 MB`, beating standard Gzip/Pigz by a small margin of `14 MB`.

- Scan the block to count every adjacent byte pair (AA, AB... ZZ).
- Identify the single most frequent pair, find a byte value that is currently unused in the block and replace every instance of the pair with that single byte.
- Repeat this cycle 64 (can be changed) times per block.

Once BPE is done data is passed to Zlib.

### Decompression

Zlib decompresses the block, revealing the BPE transformed data.

The first 3 bytes of the buffer contain the instruction for the current cycle:
- Byte 0: The High byte of the original pair.
- Byte 1: The Low byte of the original pair.
- Byte 2: The Replacement byte used in this cycle.

Instead of writing a Python for loop to iterate through the 1MB buffer (which would take seconds per block), we can use `bytearray.replace()`. It performs memory moves and byte swapping at hardware speeds.

| Algorithm       | Compressed Size | Ratio  | Compression Time | Decompression Time |
|-----------------|-----------------|--------|------------------|--------------------|
| Bzip2           | 243 MB          | 24.3%  | 66.3s            | 32.7s              |
| Zstd            | 266 MB          | 26.6%  | 43.0s            | 2.3s               |
| **BPEZ**        | 294 MB          | 29.4%  | 1670.5s          | 12.5s              |
| Gzip            | 308 MB          | 30.8%  | 46.7s            | 6.0s               |
| Pigz            | 308 MB          | 30.8%  | 6.1s             | 3.6s               |

### Usage

**Data**: `wget -c http://mattmahoney.net/dc/enwik9.zip`

**Compression**: `python3 bpez.py c <input_file> <output_file>`

**Decompression**: `python3 bpez.py d <input_file> <output_file>`

I've used `hyperfine` to measure how long commands take to run
```
hyperfine --warmup 1 --runs 2 --export-markdown results_comp_two.md \
  'python3 bpez.py c enwik9.txt enwik9.bpe' \
  'gzip -k -9 -c enwik9.txt > enwik9.gz' \
  'bzip2 -k -9 -c enwik9.txt > enwik9.bz2' \
  'zstd -k -9 -c enwik9.txt > enwik9.zst' \
  'pigz -k -9 -c enwik9.txt > enwik9.pigz'
```
