#!/usr/bin/env python3
"""
将(src, 逗号分隔dst)数据转为标准edgelist
支持制表符/空格分隔，自动跳过表头
"""
import argparse
import sys

def process_file(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件 '{input_path}' 不存在", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    output_lines = ["src dst\n"]  # 新表头 + 换行
    
    # 跳过原始表头（第一行）
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 优先尝试制表符分割（原始数据格式）
        if '\t' in line:
            parts = line.split('\t', 1)
        else:
            # 回退到空格分割（兼容复制粘贴场景）
            parts = line.split(maxsplit=1)
        
        if len(parts) < 2:
            continue
        
        src = parts[0].strip()
        dst_str = parts[1].strip()
        
        for d in dst_str.split(','):
            d_clean = d.strip()
            if d_clean:
                output_lines.append(f"{src} {d_clean}\n")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(output_lines)
        print(f"✅ 转换成功! 共 {len(output_lines)-1} 条边")
        print(f"   输入: {input_path}")
        print(f"   输出: {output_path}")
    except Exception as e:
        print(f"❌ 写入文件失败: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="将(src\\tdst1,dst2)格式转为标准edgelist (每行: src dst)",
        epilog="示例: python convert_edgelist.py -i data.txt -o edges.edgelist"
    )
    parser.add_argument('-i', '--input', required=True, help='输入文件路径 (含表头)')
    parser.add_argument('-o', '--output', default='output.edgelist', help='输出文件路径 (默认: output.edgelist)')
    args = parser.parse_args()
    
    process_file(args.input, args.output)

if __name__ == "__main__":
    main()