import sys
import struct
import multiprocessing as mp
from typing import List, Tuple, Optional
from dataclasses import dataclass
import array
import time

BLOCK_SIZE = 0x8000 
BUF_BYTES = 3 
MIN_FREQ = 3  
BYTE = 256
MAX_CYCLES = 240

BPE_DEBUG = 0


@dataclass
class Digraph:
    __slots__ = ['di', 'count', 'gen']
    di: int  # 16-bit digraph value
    count: int
    gen: int
    
    def __lt__(self, other):
        return self.count > other.count


def print_char(c: int) -> str:
    if 32 <= c < 127:
        return chr(c)
    return f"{c:02x}"


def HI(di: int) -> int:
    return (di >> 8) & 0xFF


def LO(di: int) -> int:
    return di & 0xFF


def DI(hi: int, lo: int) -> int:
    return ((hi << 8) | lo) & 0xFFFF


def inc_digraph(di: int, digraphs: List[int], dilist: List[Digraph], gen: int) -> None:
    if digraphs[di] < MIN_FREQ:
        digraphs[di] += 1
        if digraphs[di] == MIN_FREQ:
            new_dig = Digraph(di=di, count=MIN_FREQ - 1, gen=gen)
            dilist.append(new_dig)
            digraphs[di] = MIN_FREQ + len(dilist) - 1
        else:
            return
    
    idx = digraphs[di] - MIN_FREQ
    assert dilist[idx].di == di
    dilist[idx].count += 1
    dilist[idx].gen = gen


def dec_digraph(di: int, digraphs: List[int], dilist: List[Digraph], gen: int) -> None:
    if BPE_DEBUG >= 4:
        print(f"decrementing {print_char(HI(di))}:{print_char(LO(di))}")
    
    if digraphs[di] >= MIN_FREQ:
        idx = digraphs[di] - MIN_FREQ
        assert dilist[idx].di == di
        dilist[idx].count -= 1
        dilist[idx].gen = gen
    else:
        assert digraphs[di] > 0
        digraphs[di] -= 1


def wsort(dilist: List[Digraph], gen: int) -> None:
    if not dilist:
        return
    
    ordered = []
    noisy = []
    
    # Separate into modified (noisy) and unmodified (ordered) lists
    for dig in dilist:
        if gen == dig.gen:
            noisy.append(dig)
        else:
            ordered.append(dig)
    
    if not noisy:
        return
    
    if gen == 255 or len(noisy) > 1000:
        noisy.sort()
    elif len(noisy) < 50:
        # Insertion sort for very small lists
        for i in range(1, len(noisy)):
            value = noisy[i]
            j = i - 1
            while j >= 0 and value < noisy[j]:
                noisy[j + 1] = noisy[j]
                j -= 1
            noisy[j + 1] = value
    else:
        # Shell sort for medium lists (good cache performance)
        gap = len(noisy) // 2
        while gap > 0:
            for i in range(gap, len(noisy)):
                temp = noisy[i]
                j = i
                while j >= gap and noisy[j - gap] > temp:
                    noisy[j] = noisy[j - gap]
                    j -= gap
                noisy[j] = temp
            gap //= 2
    
    dilist.clear()
    i, j = 0, 0
    while i < len(ordered) or j < len(noisy):
        if i >= len(ordered):
            dilist.append(noisy[j])
            j += 1
        elif j >= len(noisy):
            dilist.append(ordered[i])
            i += 1
        elif noisy[j] < ordered[i]:
            dilist.append(noisy[j])
            j += 1
        else:
            dilist.append(ordered[i])
            i += 1


def compress_block(block_data: bytes, block_num: int) -> Tuple[bytes, int, int]:
    src = bytearray(block_data)
    dest = bytearray()
    block_len = len(src)
    
    if block_len == 0:
        return bytes(), 0, 0
    
    # Use arrays for better memory efficiency
    digraphs = array.array('H', [0] * (BYTE * BYTE))
    used = array.array('H', [0] * BYTE)
    dilist: List[Digraph] = []
    
    # Count initial digraphs
    used[src[0]] += 1
    for i in range(1, block_len):
        used[src[i]] += 1
        inc_digraph(DI(src[i-1], src[i]), digraphs, dilist, -1)
    
    # Perform BPE cycles
    cycle = 0
    while cycle < MAX_CYCLES:
        wsort(dilist, cycle - 1)
        
        # Remove low frequency digraphs
        prev_len = len(dilist)
        new_len = len(dilist)
        for i in range(len(dilist)):
            if dilist[i].count < MIN_FREQ:
                new_len = i
                break
            digraphs[dilist[i].di] = MIN_FREQ + i
        
        if new_len == 0:
            dilist.clear()
            break
        
        # Update removed digraphs
        for i in range(new_len, prev_len):
            digraphs[dilist[i].di] = dilist[i].count
        
        dilist = dilist[:new_len]
        
        # Find unused byte for replacement
        replace_idx = 0
        while replace_idx < BYTE and used[replace_idx]:
            replace_idx += 1
        
        if replace_idx == BYTE:
            break
        
        hi = HI(dilist[0].di)
        lo = LO(dilist[0].di)
        
        # Write header
        dest.clear()
        dest.extend([hi, lo, replace_idx])
        
        used[hi] += 1
        used[lo] += 1
        used[replace_idx] += 1
        inc_digraph(dilist[0].di, digraphs, dilist, cycle)
        inc_digraph(DI(lo, replace_idx), digraphs, dilist, cycle)
        inc_digraph(DI(replace_idx, src[0]), digraphs, dilist, cycle)
        
        # Replace digraphs in body
        i = 0
        while i < block_len:
            if i < block_len - 1 and src[i] == hi and src[i+1] == lo:
                used[hi] -= 1
                used[lo] -= 1
                used[replace_idx] += 1
                dec_digraph(dilist[0].di, digraphs, dilist, cycle)
                if len(dest) > 0:
                    dec_digraph(DI(dest[-1], hi), digraphs, dilist, cycle)
                    inc_digraph(DI(dest[-1], replace_idx), digraphs, dilist, cycle)
                if i < block_len - 2:
                    dec_digraph(DI(lo, src[i+2]), digraphs, dilist, cycle)
                    inc_digraph(DI(replace_idx, src[i+2]), digraphs, dilist, cycle)
                
                dest.append(replace_idx)
                i += 2
            else:
                dest.append(src[i])
                i += 1
        
        block_len = len(dest)
        src, dest = dest, src
        cycle += 1
    
    output = bytearray()
    
    output.extend(struct.pack('>I', block_len)[1:])  # Use last 3 bytes
    output.append(cycle)
    output.extend(src[:block_len])
    
    return bytes(output), len(block_data), len(output)
    
def compress(src_file: str, dest_file: str, num_workers: Optional[int] = None) -> None:
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    start_time = time.time()
    
    with open(src_file, 'rb') as srcf, open(dest_file, 'wb') as destf:
        total_read = 0
        total_written = 0
        block_num = 0
        
        if num_workers > 1:
            # Parallel processing for large files
            with mp.Pool(num_workers) as pool:
                while True:
                    # Read multiple blocks
                    blocks = []
                    for _ in range(num_workers * 2):  # Read ahead
                        chunk = srcf.read(BLOCK_SIZE)
                        if not chunk:
                            break
                        blocks.append((chunk, block_num))
                        block_num += 1
                        total_read += len(chunk)
                    
                    if not blocks:
                        break
                    
                    results = pool.starmap(compress_block, blocks)
                    
                    for compressed, orig_size, comp_size in results:
                        destf.write(compressed)
                        total_written += comp_size
                    
                    if block_num % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = total_read / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                        ratio = (total_written / total_read * 100) if total_read > 0 else 0
                        print(f"\rProcessed {block_num} blocks, {total_read // (1024*1024)} MB, "
                              f"{rate:.1f} MB/s, {ratio:.1f}% ratio", end='', flush=True)
        else:
            # Single-threaded for small files
            while True:
                chunk = srcf.read(BLOCK_SIZE)
                if not chunk:
                    break
                
                compressed, orig_size, comp_size = compress_block(chunk, block_num)
                destf.write(compressed)
                
                total_read += orig_size
                total_written += comp_size
                block_num += 1
                
                if block_num % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total_read / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    ratio = (total_written / total_read * 100) if total_read > 0 else 0
                    print(f"\rProcessed {block_num} blocks, {total_read // (1024*1024)} MB, "
                          f"{rate:.1f} MB/s, {ratio:.1f}% ratio", end='', flush=True)
        
        print()  # New line after progress
        elapsed = time.time() - start_time
        rate = total_read / (1024 * 1024 * elapsed) if elapsed > 0 else 0
        ratio = (total_written / total_read * 100) if total_read > 0 else 0
        print(f"Compressed {total_read:,} to {total_written:,} bytes in {block_num} blocks")
        print(f"Ratio: {ratio:.2f}%, Time: {elapsed:.1f}s, Speed: {rate:.1f} MB/s")


def decompress(src_file: str, dest_file: str) -> None:
    start_time = time.time()
    
    with open(src_file, 'rb') as srcf, open(dest_file, 'wb') as destf:
        buffer = [bytearray(), bytearray()]
        src_idx = 0
        dest_idx = 1
        
        block_num = 0
        total_read = 0
        total_written = 0
        
        while True:
            # Read block length (3 bytes)
            block_len_bytes = srcf.read(BUF_BYTES)
            if not block_len_bytes:
                break
            
            # Pad to 4 bytes for unpacking
            block_len = struct.unpack('>I', b'\x00' + block_len_bytes)[0]
            total_read += BUF_BYTES
            
            if block_len > BLOCK_SIZE * 10:  # Sanity check
                print(f"illegal block_len {block_len}", file=sys.stderr)
                return
            
            # Read cycles count
            cycles_byte = srcf.read(1)
            if not cycles_byte:
                print("Unexpected end of file while reading cycles", file=sys.stderr)
                break
            cycles = cycles_byte[0]
            total_read += 1
            
            # Read block data
            src = buffer[src_idx]
            src.clear()
            data = srcf.read(block_len)
            if len(data) != block_len:
                print(f"Unexpected end of file: expected {block_len} bytes, got {len(data)}", file=sys.stderr)
                break
            src.extend(data)
            total_read += block_len
            
            # Safety check: if we have cycles but not enough bytes for header
            if cycles > 0 and len(src) < 3:
                print(f"Error: block {block_num} has {cycles} cycles but only {len(src)} bytes. Skipping decompression.", file=sys.stderr)
                # Skip this block and continue
                block_num += 1
                continue
            
            # Perform decompression cycles
            for cycle in range(cycles):
                dest = buffer[dest_idx]
                dest.clear()
                
                # Additional safety check inside the cycle loop
                if len(src) < 3:
                    print(f"Error: block {block_num} cycle {cycle} has insufficient data. Skipping cycle.", file=sys.stderr)
                    break
                
                hi = src[0]
                lo = src[1]
                replace = src[2]
                
                # Replace bytes
                for i in range(3, block_len):
                    if src[i] == replace:
                        dest.extend([hi, lo])
                    else:
                        dest.append(src[i])
                
                block_len = len(dest)
                src_idx, dest_idx = dest_idx, src_idx
                src = buffer[src_idx]  # Update src reference after swap
            
            # Write decompressed block
            src = buffer[src_idx]
            destf.write(src[:block_len])
            total_written += block_len
            block_num += 1
            
            if block_num % 100 == 0:
                elapsed = time.time() - start_time
                rate = total_written / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                print(f"\rDecompressed {block_num} blocks, {total_written // (1024*1024)} MB, "
                      f"{rate:.1f} MB/s", end='', flush=True)
        
        print()  # New line after progress
        elapsed = time.time() - start_time
        rate = total_written / (1024 * 1024 * elapsed) if elapsed > 0 else 0
        ratio = (total_read / total_written * 100) if total_written > 0 else 0
        print(f"Decompressed {total_read:,} to {total_written:,} bytes in {block_num} blocks")
        print(f"Ratio: {ratio:.2f}%, Time: {elapsed:.1f}s, Speed: {rate:.1f} MB/s")
        
def main():
    if len(sys.argv) < 4:
        print("usage: bpe.py [c|d] src dest [workers]", file=sys.stderr)
        print("  c: compress")
        print("  d: decompress")
        print("  workers: number of parallel workers (default: CPU count - 1)")
        sys.exit(1)
    
    mode = sys.argv[1]
    src_file = sys.argv[2]
    dest_file = sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    if mode == 'c':
        compress(src_file, dest_file, workers)
    elif mode == 'd':
        decompress(src_file, dest_file)
    else:
        print("mode must be 'c' (compress) or 'd' (decompress)", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
