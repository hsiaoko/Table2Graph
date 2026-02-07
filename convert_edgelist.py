#!/usr/bin/env python3
"""
将制表符分隔的(src, 逗号分隔dst)数据转换为标准edgelist格式
输入: src\tdst1,dst2,dst3...
输出: src dst (每行一条边)
"""
import sys

def main():
    # 输出新表头
    print("src dst")
    
    # 跳过原始表头（第一行）
    header_skipped = False
    for line in sys.stdin:
        if not header_skipped:
            header_skipped = True
            continue
            
        line = line.strip()
        if not line:
            continue
            
        # 安全分割：按制表符分割（最多2部分）
        parts = line.split('\t', 1)
        if len(parts) < 2:
            # 尝试用空格分割（兼容部分格式）
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
                
        src = parts[0].strip()
        dst_str = parts[1].strip()
        
        # 处理逗号分隔的dst列表
        for d in dst_str.split(','):
            d_clean = d.strip()
            if d_clean:
                print(f"{src} {d_clean}")

if __name__ == "__main__":
    main()