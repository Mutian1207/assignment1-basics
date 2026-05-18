import regex as re
from collections import defaultdict, Counter
import os
from typing import BinaryIO
# 只加这一行
import multiprocessing
import time
import psutil
import threading

class PerformanceProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # 阶段计时
        self.stage_start = {}
        self.stage_duration = {}
        
        # 内存
        self.peak_rss = 0
        
    def start_stage(self, stage_name):
        """开始一个阶段计时"""
        self.stage_start[stage_name] = time.time()
        self._update_peak()
        print(f"\n📌 [{stage_name}] 开始")
        
    def end_stage(self, stage_name):
        """结束阶段计时"""
        if stage_name not in self.stage_start:
            return
        duration = time.time() - self.stage_start[stage_name]
        self.stage_duration[stage_name] = duration
        self._update_peak()
        print(f"✅ [{stage_name}] 完成 | 耗时: {duration:.2f}s")
    
    def _update_peak(self):
        """更新内存峰值"""
        rss = self.process.memory_info().rss / 1024 / 1024
        if rss > self.peak_rss:
            self.peak_rss = rss
    
    def summary(self):
        """输出完整性能报告"""
        total_time = time.time() - self.start_time
        self._update_peak()
        
        print("\n" + "="*60)
        print("📊 训练性能分析报告")
        print("="*60)
        print(f"总时间                {total_time:.2f} s")
        for stage, t in self.stage_duration.items():
            print(f"├─ {stage:<18} {t:.2f} s  ({t/total_time*100:.1f}%)")
        print(f"峰值内存占用          {self.peak_rss:.1f} MB")
        print(f"当前内存占用          {self.process.memory_info().rss/1024/1024:.1f} MB")
        print(f"CPU 核心数            {os.cpu_count()}")
        print("="*60)

# 全局单例，你直接用
profiler = PerformanceProfiler()
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

class BpeTokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = dict()
        self.reverse_vocab = dict()
        self.token_freqs = Counter()
        self.vocab_index = 0
        self.merges = []
        self.lock = multiprocessing.Lock()  # 进程锁
        self._build_vocab()

    def _split_special_tokens(self, text, special_tokens):
        if not special_tokens:
            return [text]
        escaped_special_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        special_tokens_pattern = re.compile("(" + "|".join(escaped_special_tokens) + ")")
        parts = [part for part in special_tokens_pattern.split(text) if part != ""]
        return parts

    def _pre_tokenize(self, text_parts):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_freqs = Counter()
        for part in text_parts:
            if part in self.special_tokens:
                byte_token = tuple([part.encode("utf-8")])
                token_freqs[byte_token] += 1
                continue
            matches = re.finditer(PAT, part)
            for match in matches:
                token_str = match.group()
                byte_token = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                token_freqs[byte_token] += 1
        return token_freqs

    def _build_vocab(self):
        for token in self.special_tokens:
            byte_token = token.encode("utf-8")
            self.vocab[self.vocab_index] = byte_token
            self.reverse_vocab[byte_token] = self.vocab_index
            self.vocab_index += 1
        for i in range(256):
            byte_token = bytes([i])
            self.vocab[self.vocab_index] = byte_token
            self.reverse_vocab[byte_token] = self.vocab_index
            self.vocab_index += 1
        print(f"初始词汇表构建完成，大小：{self.vocab_index}（特殊token+256字节）")

    def _get_pair_counts(self):
        pair_counts = Counter()
        for token, freq in self.token_freqs.items():
            if len(token) < 2: continue
            for i in range(len(token)-1):
                pair = (token[i], token[i+1])
                pair_counts[pair] += freq
        return pair_counts

    def _merge_pair(self, pair, new_token):
        new_token_freqs = Counter()
        for token, freq in self.token_freqs.items():
            if len(token) < 2:
                new_token_freqs[token] += freq
                continue
            i = 0
            merged_token = []
            while i < len(token)-1:
                if (token[i], token[i+1]) == pair:
                    merged_token.append(new_token)
                    i+=2
                else:
                    merged_token.append(token[i])
                    i+=1
            if i == len(token)-1: merged_token.append(token[-1])
            new_token_freqs[tuple(merged_token)] += freq
        self.token_freqs = new_token_freqs

    def _merge_loop(self):
        while self.vocab_index < self.vocab_size:
            pair_counts = self._get_pair_counts()
            if not pair_counts:
                print("无更多可合并的字节对，终止合并")
                break
            most_common_pair, freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            new_token = most_common_pair[0] + most_common_pair[1]
            self.merges.append(most_common_pair)
            self.vocab[self.vocab_index] = new_token
            self.reverse_vocab[new_token] = self.vocab_index
            self.vocab_index += 1
            self._merge_pair(most_common_pair, new_token)
            if self.vocab_index % 100 == 0:
                print(f"合并进度：词汇表大小 {self.vocab_index}/{self.vocab_size}")

    def _process_worker(self, start, end, return_dict):
        try:
            with open(self.input_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", "ignore")
            clean = self._split_special_tokens(chunk, self.special_tokens)
            counter = self._pre_tokenize(clean)
            return_dict[(start, end)] = counter
            print(f"[进程] 完成块 {start}-{end}")
        except Exception as e:
            print(f"[进程错误] {start}-{end}: {e}")
            return_dict[(start, end)] = Counter()

    def train(self):
        profiler.start_stage("分块")
        with open(self.input_path, 'rb') as f:
            num_workers = 1
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

        chunks = list(zip(boundaries[:-1], boundaries[1:]))
        print(f"\n[主线程] 共 {len(chunks)} 块，使用 {num_workers} 个CPU核心")
        profiler.end_stage("分块")
        
        # 多进程共享字典
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        # 启动进程
        profiler.start_stage("并行预分词")
        for start, end in chunks:
            p = multiprocessing.Process(
                target=self._process_worker,
                args=(start, end, return_dict)
            )
            processes.append(p)
            p.start()

        # 等待全部结束
        for p in processes:
            p.join()
        profiler.end_stage("并行预分词")
        
        # 合并结果
        profiler.start_stage("合并预分词结果")
        for key, counter in return_dict.items():
            self.token_freqs.update(counter)
        profiler.end_stage("合并预分词结果")
        
        print("\n[完成] 预分词结束，开始BPE合并")
        profiler.start_stage("BPE合并")
        self._merge_loop()
        profiler.end_stage("BPE合并")
        print(f"\n训练完成！词汇表大小：{self.vocab_index}")
        
        profiler.summary()
import json  # 必须加

class Tokenizer(BpeTokenizer):
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    @staticmethod
    def from_file(vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            merges = [tuple(line.strip().split()) for line in f]
        return Tokenizer(vocab, merges, special_tokens)

    def _pre_tokenize(self, text_parts) -> list[tuple]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        res = []
        for part in text_parts:
            if not part:  # 修复：过滤空字符串
                continue
            for match in re.finditer(PAT, part):
                token_str = match.group()
                byte_token = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                res.append(byte_token)
        return res

    def _split_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        escaped_special_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
        special_tokens_pattern = re.compile("(" + "|".join(escaped_special_tokens) + ")")
        parts = [part for part in special_tokens_pattern.split(text) if part != ""]
        return parts

    def encode(self, text: str) -> list[int]:
        text_ids = []
        for part in self._split_special_tokens(text):
            if part in self.special_tokens:
                byte_token = part.encode("utf-8")
                text_ids.append(self.reverse_vocab[byte_token])
                continue

            byte_tokens = self._pre_tokenize([part])
            for token_tuple in byte_tokens:
                tokens = list(token_tuple)

                for pair in self.merges:
                    new_tokens = []
                    idx = 0
                    while idx < len(tokens):
                        if idx + 1 < len(tokens) and (tokens[idx], tokens[idx+1]) == pair:
                            new_tokens.append(pair[0] + pair[1])
                            idx += 2
                        else:
                            new_tokens.append(tokens[idx])
                            idx += 1
                    tokens = new_tokens

                # 安全获取 id
                for t in tokens:
                    text_ids.append(self.reverse_vocab[t])
        return text_ids


    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")


    def encode_iterable(self, iterable):
        for line in iterable:
            yield from self.encode(line)