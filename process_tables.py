import csv
import argparse
from pathlib import Path

def build_key_mapping(input_files, key_column_index, delimiter): 
    """ 
    从所有输入文件中遍历指定的键列，为每个唯一的字符串键生成一个从 0 开始的整数 ID。

    Args:
        input_files (list): 输入文件路径列表。
        key_column_index (int): 键所在的列索引。
        delimiter (str): 输入文件的分隔符。

    Returns:
        dict: 字符串键到整数 ID 的映射字典。
    """
    key_mapping = {}
    next_id = 0
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if row:
                    key = row[key_column_index]
                    if key not in key_mapping:
                        key_mapping[key] = next_id
                        next_id += 1
    return key_mapping

def process_and_replace(input_file, output_file, key_mapping, key_column_index, input_delimiter, output_delimiter):
    """
    读取输入文件，使用提供的映射替换键，并用新的分隔符写入输出文件。

    Args:
        input_file (str): 输入文件路径。
        output_file (str): 输出文件路径。
        key_mapping (dict): 字符串键到整数 ID 的映射。
        key_column_index (int): 键所在的列索引。
        input_delimiter (str): 输入文件的分隔符。
        output_delimiter (str): 输出文件的分隔符。
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile, delimiter=input_delimiter)
        writer = csv.writer(outfile, delimiter=output_delimiter)
        for row in reader:
            if row:
                key = row[key_column_index]
                if key in key_mapping:
                    row[key_column_index] = str(key_mapping[key])
                writer.writerow(row)

def main():
    """主函数，用于解析参数并协调处理流程。"""
    parser = argparse.ArgumentParser(description='Process relational tables by replacing string keys with integer IDs and changing delimiters.')
    parser.add_argument('-i', '--input-files', nargs='+', required=True, help='List of input file paths.')
    parser.add_argument('-o', '--output-dir', required=True, help='Directory to save processed files.')
    parser.add_argument('-k', '--key-column', type=int, required=True, help='Index of the key column to be replaced.')
    parser.add_argument('--in-delim', default=',', help='Delimiter for input files.')
    parser.add_argument('--out-delim', default='\t', help='Delimiter for output files.')

    args = parser.parse_args()

    # 确保输出目录存在
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 构建统一的键映射
    print("Building key mapping...")
    mapping = build_key_mapping(args.input_files, args.key_column, args.in_delim)
    print(f"Found {len(mapping)} unique keys.")

    # 2. 处理每个文件
    for input_file in args.input_files:
        input_path = Path(input_file)
        output_file = output_path / input_path.name
        print(f"Processing {input_file} -> {output_file}...")
        process_and_replace(input_file, str(output_file), mapping, args.key_column, args.in_delim, args.out_delim)
    
    print("\nProcessing complete.")

if __name__ == '__main__':
    # 为了演示，我们可以创建一些示例文件
    # 在实际使用中，请通过命令行参数提供您自己的文件
    def create_dummy_files():
        Path("temp_data").mkdir(exist_ok=True)
        with open("temp_data/table1.csv", "w") as f:
            f.write("user_A,item_1,10\n")
            f.write("user_B,item_2,20\n")
        with open("temp_data/table2.csv", "w") as f:
            f.write("user_A,location_X,100\n")
            f.write("user_C,location_Y,200\n")
        print("Created dummy files in 'temp_data' directory.")
        print("Example usage:")
        print("python process_tables.py -i temp_data/table1.csv temp_data/table2.csv -o output_data -k 0 --in-delim ',' --out-delim ' ' ")

    # create_dummy_files()
    main()
