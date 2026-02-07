#!/usr/bin/env python3
"""
将 IMDb name.basics.tsv 中的 nconst 和 knownForTitles 
通过映射文件转换为整数 ID，生成 person_id -> movie_id 的 edgelist
"""
import argparse
import sys

def load_mapping(map_file, key_col=0, val_col=1):
    """加载映射文件（TSV），跳过表头，返回 {original_id: int_id_str}"""
    mapping = {}
    try:
        with open(map_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[key_col].strip() and parts[val_col].strip():
                    mapping[parts[key_col].strip()] = parts[val_col].strip()
        print(f"✓ 加载映射: {len(mapping)} 条 | 源文件: {map_file}", file=sys.stderr)
        return mapping
    except Exception as e:
        print(f"❌ 映射文件读取失败 '{map_file}': {e}", file=sys.stderr)
        sys.exit(1)

def process_name_basics(name_file, nconst_map, tconst_map, output_file):
    """处理主数据文件，生成 edgelist"""
    edge_count = 0
    skipped_nconst = 0
    skipped_tconst = 0
    
    try:
        with open(name_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            # 写入新表头
            fout.write("person_id movie_id\n")
            
            # 跳过 name.basics 表头
            header = next(fin, None)
            if header is None:
                raise ValueError("输入文件为空")
            
            for line_num, line in enumerate(fin, start=2):
                line = line.strip()
                if not line:
                    continue
                
                # 严格按制表符分割（IMDb TSV 标准格式）
                fields = line.split('\t')
                if len(fields) < 6:
                    print(f"⚠ 跳过格式异常行 {line_num} (字段数={len(fields)})", file=sys.stderr)
                    continue
                
                nconst = fields[0].strip()
                known_titles = fields[5].strip()
                
                # 映射 person_id
                person_id = nconst_map.get(nconst)
                if not person_id:
                    skipped_nconst += 1
                    continue
                
                # 处理 knownForTitles
                if not known_titles or known_titles == '\\N':
                    continue
                
                for tconst_raw in known_titles.split(','):
                    tconst = tconst_raw.strip()
                    if not tconst:
                        continue
                    movie_id = tconst_map.get(tconst)
                    if movie_id:
                        fout.write(f"{person_id} {movie_id}\n")
                        edge_count += 1
                    else:
                        skipped_tconst += 1
            
        # 汇总报告
        print(f"\n✅ 转换完成!", file=sys.stderr)
        print(f"   总边数: {edge_count}", file=sys.stderr)
        print(f"   跳过: 未映射的人员ID = {skipped_nconst}", file=sys.stderr)
        print(f"   跳过: 未映射的作品ID = {skipped_tconst}", file=sys.stderr)
        print(f"   输出文件: {output_file}", file=sys.stderr)
        return edge_count
    
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理失败: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="将 IMDb name.basics.tsv 映射为整数ID edgelist (person_id movie_id)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python map_imdb_to_edgelist.py \\
    -n name.basics.tsv \\
    -p name_id_map.tsv \\
    -t title_id_map.tsv \\
    -o person_movie_edges.edgelist
""")
    parser.add_argument('-n', '--name_basics', required=True, help='IMDb name.basics.tsv 文件')
    parser.add_argument('-p', '--person_map', required=True, help='nconst -> int_id 映射文件 (文件2)')
    parser.add_argument('-t', '--title_map', required=True, help='tconst -> int_id 映射文件 (文件3)')
    parser.add_argument('-o', '--output', default='person_movie_edges.edgelist', help='输出 edgelist 路径 (默认: person_movie_edges.edgelist)')
    
    args = parser.parse_args()
    
    # 加载映射
    nconst_to_id = load_mapping(args.person_map)
    tconst_to_id = load_mapping(args.title_map)
    
    # 处理主文件
    process_name_basics(args.name_basics, nconst_to_id, tconst_to_id, args.output)

if __name__ == "__main__":
    main()