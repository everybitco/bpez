import sys
import struct
import multiprocessing as mp
import os
import array
import zlib
from typing import List, Iterator
from dataclasses import dataclass

BLOCK_SIZE = 1024 * 1024  # 1MB Chunk
MIN_FREQ = 3              # Minimum pair frequency
MAX_CYCLES = 64          # Maximum BPE depth
ZLIB_LEVEL = 9            # compression setting

@dataclass
class Digraph:
    __slots__ = ['di', 'count', 'gen']
    di: int
    count: int
    gen: int

# BPE Core Logic

def inc_digraph(di: int, digraphs: array.array, dilist: List[Digraph], gen: int) -> None:
    """Safely increments a digraph count, adding to the candidate list if needed."""
    if digraphs[di] < MIN_FREQ:
        digraphs[di] += 1
        if digraphs[di] == MIN_FREQ:
            # New candidate found
            dilist.append(Digraph(di, MIN_FREQ - 1, gen))
            digraphs[di] = MIN_FREQ + len(dilist) - 1
    else:
        # Existing candidate
        idx = digraphs[di] - MIN_FREQ
        if idx < len(dilist):
            dilist[idx].count += 1
            dilist[idx].gen = gen

def dec_digraph(di: int, digraphs: array.array, dilist: List[Digraph], gen: int) -> None:
    """Safely decrements a digraph count, handling state synchronization."""
    val = digraphs[di]
    if val >= MIN_FREQ:
        idx = val - MIN_FREQ
        if idx < len(dilist) and dilist[idx].di == di:
            if dilist[idx].count > 0:
                dilist[idx].count -= 1
            dilist[idx].gen = gen
        else:
            # Invalid state reset
            digraphs[di] = 0
    elif val > 0:
        digraphs[di] -= 1

def wsort(dilist: List[Digraph]) -> None:
    """Sorts candidates by frequency (descending)."""
    if not dilist: return
    dilist.sort(key=lambda x: x.count, reverse=True)

# Worker Functions

def compress_chunk(args) -> bytes:
    """
    Worker: Performs BPE reduction followed by Zlib compression on a data chunk.
    """
    chunk_data, _ = args
    src = bytearray(chunk_data)
    if not src: return b''

    block_len = len(src)
    digraphs = array.array('I', [0] * 65536) 
    used = array.array('I', [0] * 256)
    dilist: List[Digraph] = []

    # Initial Frequency Scan
    if block_len > 0:
        used[src[0]] += 1
        for i in range(1, block_len):
            used[src[i]] += 1
            di = ((src[i-1] << 8) | src[i])
            inc_digraph(di, digraphs, dilist, -1)

    cycle = 0
    dest = bytearray()
    
    # BPE Reduction Loop
    while cycle < MAX_CYCLES:
        wsort(dilist)
        
        # Prune and remap indices
        valid_count = 0
        for d in dilist:
            if d.count < MIN_FREQ:
                digraphs[d.di] = max(0, d.count)
            else:
                digraphs[d.di] = MIN_FREQ + valid_count
                dilist[valid_count] = d
                valid_count += 1
        del dilist[valid_count:]
        
        if not dilist: break

        best = dilist[0]
        hi = (best.di >> 8) & 0xFF
        lo = best.di & 0xFF

        # Find unused byte token
        replace_idx = -1
        for i in range(256):
            if used[i] == 0:
                replace_idx = i
                break
        
        if replace_idx == -1: break

        # Prepare Header and Stats
        dest.clear()
        dest.extend((hi, lo, replace_idx))
        used[hi] += 1; used[lo] += 1; used[replace_idx] += 1
        inc_digraph(((hi << 8) | lo), digraphs, dilist, cycle)
        
        # Perform Substitution
        i = 0
        src_len = len(src)
        while i < src_len:
            if i < src_len - 1 and src[i] == hi and src[i+1] == lo:
                used[hi] -= 1; used[lo] -= 1; used[replace_idx] += 1
                dec_digraph(best.di, digraphs, dilist, cycle)
                
                # Update Context
                if len(dest) > 0:
                    prev = dest[-1]
                    dec_digraph(((prev << 8) | hi), digraphs, dilist, cycle)
                    inc_digraph(((prev << 8) | replace_idx), digraphs, dilist, cycle)
                if i < src_len - 2:
                    next_b = src[i+2]
                    dec_digraph(((lo << 8) | next_b), digraphs, dilist, cycle)
                    inc_digraph(((replace_idx << 8) | next_b), digraphs, dilist, cycle)
                
                dest.append(replace_idx)
                i += 2
            else:
                dest.append(src[i])
                i += 1
        
        src[:] = dest
        cycle += 1

    # Pack Payload: [Uncompressed Len(4)] + [Cycles(1)] + [BPE Data]
    payload = struct.pack('>I', block_len) + bytes([cycle]) + src
    
    # Final Stage: Zlib Deflate
    compressed = zlib.compress(payload, level=ZLIB_LEVEL)
    
    # Return: [Block Size(4)] + [Data]
    return struct.pack('>I', len(compressed)) + compressed

def decompress_chunk(zlib_data: bytes) -> bytes:
    try:
        payload = zlib.decompress(zlib_data)
    except Exception:
        return b'' # Corrupt block
    
    if len(payload) < 5: return b''
    
    # Payload: [OrigLen(4)][Cycles(1)][BPE Stream...]
    cycles = payload[4]
    buf = bytearray(payload[5:])
    
    # Reverse BPE (Expansion)
    for _ in range(cycles):
        if len(buf) < 3: break
        
        # Header: [HI, LO, REPL]
        hi, lo, repl = buf[0], buf[1], buf[2]
        
        # Optimized C-level replacement
        current_data = buf[3:]
        buf = current_data.replace(bytes([repl]), bytes([hi, lo]))
        
    return buf

#I/O Managers

def run_compression(src_path: str, dst_path: str, threads: int) -> None:
    file_size = os.path.getsize(src_path)
    print(f"Compressing: '{src_path}' -> '{dst_path}'")
    print(f"Workers: {threads}")
    
    total_written = 0
    
    def stream_source():
        with open(src_path, 'rb') as f:
            idx = 0
            while True:
                chunk = f.read(BLOCK_SIZE)
                if not chunk: break
                yield (chunk, idx)
                idx += 1

    try:
        with open(dst_path, 'wb') as out, mp.Pool(threads) as pool:
            for result in pool.imap(compress_chunk, stream_source()):
                out.write(result)
                total_written += len(result)
                
        ratio = (total_written / file_size) * 100
        print(f"Complete. Size: {total_written:,} bytes (Ratio: {ratio:.2f}%)")
        
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during compression: {e}")
        sys.exit(1)

def run_decompression(src_path: str, dst_path: str, threads: int) -> None:
    print(f"Decompressing: '{src_path}' -> '{dst_path}'")
    
    def stream_compressed():
        with open(src_path, 'rb') as f:
            while True:
                len_bytes = f.read(4)
                if not len_bytes: break
                z_len = struct.unpack('>I', len_bytes)[0]
                z_data = f.read(z_len)
                if len(z_data) != z_len: break
                yield z_data

    try:
        with open(dst_path, 'wb') as out, mp.Pool(threads) as pool:
            for chunk in pool.imap(decompress_chunk, stream_compressed()):
                out.write(chunk)
        
        print("Decompression complete.")
        
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during decompression: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 4:
        print("Usage: python bpe_z.py [c|d] <input> <output> [threads]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    src = sys.argv[2]
    dst = sys.argv[3]
    
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, mp.cpu_count())

    if not os.path.exists(src):
        print(f"Error: Source file '{src}' not found.")
        sys.exit(1)

    if mode == 'c':
        run_compression(src, dst, threads)
    elif mode == 'd':
        run_decompression(src, dst, threads)
    else:
        print("Error: Mode must be 'c' (compress) or 'd' (decompress).")
        sys.exit(1)

if __name__ == '__main__':
    main()
