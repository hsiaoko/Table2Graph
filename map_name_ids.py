import sys
import csv

def load_id_map(filepath, key_col=0, val_col=1, delimiter='\t'):
    """加载 ID 映射表，返回 dict: original_id -> int_id (as int)"""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader, None)  # 跳过标题行（如果存在）
        for row in reader:
            if len(row) <= max(key_col, val_col):
                continue
            orig_id = row[key_col].strip()
            try:
                int_id = int(row[val_col])
                mapping[orig_id] = int_id
            except ValueError:
                continue  # 忽略无效 ID
    return mapping

def main(name_basics_file, name_map_file, title_map_file, output_file):
    # 加载映射表
    name_to_int = load_id_map(name_map_file, key_col=0, val_col=1)
    title_to_int = load_id_map(title_map_file, key_col=0, val_col=1)

    with open(name_basics_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout, delimiter='\t')

        # 写入新标题（可选）
        header = next(reader)  # 假设有标题行
        new_header = ['nconst_int', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession', 'knownForTitles_int']
        writer.writerow(new_header)

        for row in reader:
            if len(row) < 6:
                continue
            nconst, primaryName, birthYear, deathYear, primaryProfession, knownForTitles = row[:6]

            # 映射 nconst
            nconst_int = name_to_int.get(nconst, -1)  # 未找到用 -1 表示

            # 映射 knownForTitles（逗号分隔的 tconst 列表）
            if knownForTitles == '\\N':
                titles_int = []
            else:
                tconsts = knownForTitles.split(',')
                titles_int = [str(title_to_int.get(tid.strip(), -1)) for tid in tconsts]
                titles_int = [tid for tid in titles_int if tid != '-1']  # 可选：过滤掉未映射的

            knownForTitles_int_str = ','.join(titles_int) if titles_int else ''

            # 写出新行
            writer.writerow([
                nconst_int,
                primaryName,
                birthYear,
                deathYear,
                primaryProfession,
                knownForTitles_int_str
            ])

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python map_name_ids.py <name.basics.tsv> <name_id_map.tsv> <title_id_map.tsv> <output.tsv>")
        sys.exit(1)

    name_basics = sys.argv[1]
    name_map = sys.argv[2]
    title_map = sys.argv[3]
    output = sys.argv[4]

    main(name_basics, name_map, title_map, output)