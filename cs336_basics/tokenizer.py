import regex as re
from collections import defaultdict, Counter

class BpeTokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens  # 特殊token列表（如["<|endoftext|>"]）
        self.vocab = dict()  # 索引→字节token（如0: b'<|endoftext|>', 1: b'\x00'）
        self.reverse_vocab = dict()  # 字节token→索引（反向映射，方便查找）
        self.token_freqs = Counter()  # 预分词后的token频率（字节序列→频率）
        self.vocab_index = 0
        self.merges = []  # 记录合并规则：(byte1, byte2) → merged_byte
        self._build_vocab()
        
    def _remove_special_tokens(self, text, special_tokens):
        """移除特殊token并拆分文本，返回非空纯文本片段列表"""
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        special_tokens_pattern = "|".join(escaped_special_tokens)
        split_parts = re.split(special_tokens_pattern, text)
        print("split_parts:", split_parts[0][:100])  # 打印第一个片段的前100字符，检查是否正确拆分
        return split_parts
    
    def _pre_tokenize(self, text_parts):
        """
        预分词核心逻辑：
        1. 对每个纯文本片段，用正则拆分为基础token（单词/数字/标点）
        2. 每个基础token转UTF-8字节序列，统计频率
        返回：{字节序列元组: 频率}（如 (b'h', b'e', b'l', b'l', b'o'): 5）
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_freqs = Counter()
        
        for part in text_parts:
            # 用正则匹配基础token（单词、数字、标点等）
            matches = re.finditer(PAT, part)
            # print(f"正在预分词，当前文本片段前100字符：{part[:1000]}，匹配到的token示例：{[m.group() for m in matches][:100]}")
            for match in matches:
                token_str = match.group()
                # 转字节序列："hello" → (b'h', b'e', b'l', b'l', b'o')
                byte_token = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                token_freqs[byte_token] += 1
        
        self.token_freqs = token_freqs
        print(f"预分词完成，共识别 {len(token_freqs)} 种基础token")
        print("{}".format(token_freqs.most_common(10)))  # 打印最常见的10个token及其频率，检查是否合理
        return token_freqs
    
    def _build_vocab(self):
        """初始化词汇表：先加特殊token，再加256个基础字节"""
        # 1. 添加特殊token
        for token in self.special_tokens:
            byte_token = token.encode("utf-8")
            self.vocab[self.vocab_index] = byte_token
            self.reverse_vocab[byte_token] = self.vocab_index
            self.vocab_index += 1
        
        # 2. 添加0-255的基础字节
        for i in range(256):
            byte_token = bytes([i])
            self.vocab[self.vocab_index] = byte_token
            self.reverse_vocab[byte_token] = self.vocab_index
            self.vocab_index += 1
        
        print(f"初始词汇表构建完成，大小：{self.vocab_index}（特殊token+256字节）")

    def _get_pair_counts(self):
        """统计所有相邻字节对的频率（核心：关联token的总频率）"""
        pair_counts = Counter()
        for token, freq in self.token_freqs.items():
            if len(token) < 2:
                continue
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def _merge_pair(self, pair, new_token):
        """
        合并指定字节对：遍历所有token，将 (byte1, byte2) 替换为 new_token
        """
        new_token_freqs = Counter()
        for token, freq in self.token_freqs.items():
            if len(token) < 2:
                new_token_freqs[token] += freq
                continue
            i = 0
            merged_token = []
            while i < len(token) - 1:
                if (token[i], token[i + 1]) == pair:
                    # 找到匹配的字节对，进行合并
                    merged_token.append(new_token)
                    i+=2
                else:
                    merged_token.append(token[i])
                    i+=1
            if i == len(token) - 1:
                merged_token.append(token[-1])
            
            new_token_freqs[tuple(merged_token)] += freq
        self.token_freqs = new_token_freqs
            


    def _merge_loop(self):
        """BPE核心合并循环：直到词汇表达到目标大小"""
        while self.vocab_index < self.vocab_size:
            # 1. 统计当前所有字节对的频率
            pair_counts = self._get_pair_counts()
            if not pair_counts:  # 无可用合并对，提前终止
                print("无更多可合并的字节对，终止合并")
                break
            
            # 2. 找到频率最高的字节对
            most_common_pair, freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            
            # 3. 生成新token（合并后的字节串）
            new_token = most_common_pair[0] + most_common_pair[1]
            
            # 4. 记录合并规则，更新词汇表
            self.merges.append(most_common_pair)
            self.vocab[self.vocab_index] = new_token
            self.reverse_vocab[new_token] = self.vocab_index
            self.vocab_index += 1
            
            # 5. 合并所有token中的该字节对
            self._merge_pair(most_common_pair, new_token)
            
            # 打印进度（可选）
            if self.vocab_index % 100 == 0:
                print(f"合并进度：词汇表大小 {self.vocab_index}/{self.vocab_size}")

    def train(self):
        """完整训练流程"""
        # 1. 加载文本
        with open(self.input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # 2. 移除特殊token
        clean_text_parts = self._remove_special_tokens(raw_text, self.special_tokens)
        print(f"移除特殊token后，拆分为 {len(clean_text_parts)} 个纯文本片段")
        
        # 3. 预分词：得到字节序列的频率统计
        self._pre_tokenize(clean_text_parts)
        
        # 4. 执行BPE合并循环
        print("开始BPE合并...")
        self._merge_loop()
        
        # 5. 训练完成
        print(f"\n训练完成！最终词汇表大小：{self.vocab_index}")
        print(f"共执行 {len(self.merges)} 次合并")
        print("部分合并规则示例（前10条）：")
        for i, merge in enumerate(self.merges[:100]):
            print(f"  {i+1}: {merge}")