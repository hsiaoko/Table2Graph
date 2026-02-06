"""
IMDb数据去重处理工具
功能：去除dst列中逗号分隔值的重复元素，并支持多种处理模式
"""

import pandas as pd
import numpy as np
from typing import List, Union, Set, Dict, Tuple
import os
import argparse
from collections import Counter
import json


class IMDBDeduplicator:
    """IMDb数据去重处理器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化去重处理器
        
        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """打印日志"""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def remove_duplicates_from_csv_string(self, csv_string: str, delimiter: str = ',') -> str:
        """
        从逗号分隔的字符串中去除重复元素，保持原始顺序
        
        Args:
            csv_string: 逗号分隔的字符串
            delimiter: 分隔符
            
        Returns:
            去除重复后的字符串
        """
        if pd.isna(csv_string) or not str(csv_string).strip():
            return csv_string
        
        try:
            # 分割字符串并清理空白
            elements = [elem.strip() for elem in str(csv_string).split(delimiter) if elem.strip()]
            
            if not elements:
                return csv_string
            
            # 使用有序字典保持顺序
            seen = set()
            unique_elements = []
            
            for elem in elements:
                if elem not in seen:
                    seen.add(elem)
                    unique_elements.append(elem)
            
            return delimiter.join(unique_elements)
            
        except Exception as e:
            self._log(f"处理字符串时出错: {e}", "ERROR")
            return csv_string
    
    def analyze_column_duplicates(self, df: pd.DataFrame, column: str = 'dst', delimiter: str = ',') -> Dict:
        """
        分析指定列的重复情况
        
        Args:
            df: DataFrame
            column: 列名
            delimiter: 分隔符
            
        Returns:
            分析结果字典
        """
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在于DataFrame中")
        
        analysis = {
            'total_rows': len(df),
            'non_empty_rows': 0,
            'total_elements': 0,
            'unique_elements': 0,
            'duplicate_elements': 0,
            'rows_with_duplicates': 0,
            'duplicate_ratio': 0.0,
            'most_duplicated_row': None,
            'most_common_duplicates': []
        }
        
        max_duplicates = 0
        most_duplicated_idx = -1
        all_duplicates_counter = Counter()
        
        for idx, row in df.iterrows():
            value = row[column]
            if pd.notna(value) and str(value).strip():
                analysis['non_empty_rows'] += 1
                
                # 分割元素
                elements = [elem.strip() for elem in str(value).split(delimiter) if elem.strip()]
                analysis['total_elements'] += len(elements)
                
                # 计算唯一元素
                unique_elements = []
                seen = set()
                for elem in elements:
                    if elem not in seen:
                        seen.add(elem)
                        unique_elements.append(elem)
                
                analysis['unique_elements'] += len(unique_elements)
                
                # 计算重复
                duplicates_count = len(elements) - len(unique_elements)
                analysis['duplicate_elements'] += duplicates_count
                
                if duplicates_count > 0:
                    analysis['rows_with_duplicates'] += 1
                    
                    # 更新最大重复行
                    if duplicates_count > max_duplicates:
                        max_duplicates = duplicates_count
                        most_duplicated_idx = idx
                    
                    # 统计重复项
                    element_counter = Counter(elements)
                    for elem, count in element_counter.items():
                        if count > 1:
                            all_duplicates_counter[elem] += (count - 1)
        
        # 计算重复率
        if analysis['total_elements'] > 0:
            analysis['duplicate_ratio'] = analysis['duplicate_elements'] / analysis['total_elements']
        
        # 设置最多重复的行
        if most_duplicated_idx >= 0:
            most_duplicated_value = df.loc[most_duplicated_idx, column]
            elements = [elem.strip() for elem in str(most_duplicated_value).split(delimiter) if elem.strip()]
            element_counter = Counter(elements)
            duplicates = {elem: count for elem, count in element_counter.items() if count > 1}
            
            analysis['most_duplicated_row'] = {
                'row_index': int(most_duplicated_idx),
                'total_elements': len(elements),
                'unique_elements': len(set(elements)),
                'duplicate_count': max_duplicates,
                'duplicates': duplicates
            }
        
        # 设置最常见的重复项
        analysis['most_common_duplicates'] = all_duplicates_counter.most_common(10)
        
        return analysis
    
    def deduplicate_column(self, df: pd.DataFrame, column: str = 'dst', 
                          delimiter: str = ',', inplace: bool = False) -> pd.DataFrame:
        """
        对DataFrame指定列进行去重
        
        Args:
            df: DataFrame
            column: 列名
            delimiter: 分隔符
            inplace: 是否原地修改
            
        Returns:
            处理后的DataFrame
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在于DataFrame中")
        
        # 分析去重前的情况
        before_analysis = self.analyze_column_duplicates(df, column, delimiter)
        
        self._log(f"开始处理列 '{column}' 的去重...")
        self._log(f"处理前行数: {before_analysis['total_rows']}")
        self._log(f"非空行数: {before_analysis['non_empty_rows']}")
        self._log(f"总元素数: {before_analysis['total_elements']}")
        self._log(f"重复元素数: {before_analysis['duplicate_elements']}")
        self._log(f"重复率: {before_analysis['duplicate_ratio']:.2%}")
        
        # 应用去重
        rows_processed = 0
        rows_with_changes = 0
        
        for idx in df.index:
            original_value = df.at[idx, column]
            if pd.notna(original_value):
                processed_value = self.remove_duplicates_from_csv_string(original_value, delimiter)
                
                if str(original_value) != str(processed_value):
                    rows_with_changes += 1
                    df.at[idx, column] = processed_value
                
                rows_processed += 1
        
        # 分析去重后的情况
        after_analysis = self.analyze_column_duplicates(df, column, delimiter)
        
        self._log(f"处理完成！")
        self._log(f"处理行数: {rows_processed}")
        self._log(f"修改行数: {rows_with_changes}")
        self._log(f"去重后总元素数: {after_analysis['total_elements']}")
        self._log(f"去重后重复元素数: {after_analysis['duplicate_elements']}")
        self._log(f"去重后重复率: {after_analysis['duplicate_ratio']:.2%}")
        self._log(f"减少的元素数: {before_analysis['total_elements'] - after_analysis['total_elements']}")
        
        # 显示重复最多的行信息
        if before_analysis['most_duplicated_row']:
            dup_info = before_analysis['most_duplicated_row']
            self._log(f"\n重复最多的行（处理前）:")
            self._log(f"  行号: {dup_info['row_index']}")
            self._log(f"  总元素: {dup_info['total_elements']}")
            self._log(f"  唯一元素: {dup_info['unique_elements']}")
            self._log(f"  重复数: {dup_info['duplicate_count']}")
            
            if dup_info['duplicates']:
                self._log(f"  重复项:")
                for elem, count in dup_info['duplicates'].items():
                    self._log(f"    '{elem}': {count}次")
        
        # 显示最常见的重复元素
        if before_analysis['most_common_duplicates']:
            self._log(f"\n最常见的重复元素（前10）:")
            for elem, count in before_analysis['most_common_duplicates']:
                self._log(f"  '{elem}': 总共重复{count}次")
        
        return df
    
    def expand_edgelist(self, df: pd.DataFrame, src_column: str = 'src',
                       dst_column: str = 'dst', delimiter: str = ',') -> pd.DataFrame:
        """
        将逗号分隔的dst列展开为单一边缘列表
        
        Args:
            df: 输入DataFrame
            src_column: 源节点列名
            dst_column: 目标节点列名
            delimiter: 分隔符
            
        Returns:
            展开后的边缘列表DataFrame
        """
        self._log(f"开始展开边缘列表...")
        self._log(f"原始数据行数: {len(df)}")
        
        expanded_rows = []
        total_edges_before = 0
        total_edges_after = 0
        
        for idx, row in df.iterrows():
            src_value = row[src_column]
            dst_value = row[dst_column]
            
            if pd.notna(dst_value) and str(dst_value).strip():
                # 先去除重复
                unique_dst_string = self.remove_duplicates_from_csv_string(str(dst_value), delimiter)
                dst_elements = [elem.strip() for elem in unique_dst_string.split(delimiter) if elem.strip()]
                
                total_edges_before += len([elem.strip() for elem in str(dst_value).split(delimiter) if elem.strip()])
                total_edges_after += len(dst_elements)
                
                # 为每个目标节点创建一行
                for dst_element in dst_elements:
                    expanded_rows.append({
                        src_column: src_value,
                        dst_column: dst_element
                    })
        
        expanded_df = pd.DataFrame(expanded_rows)
        
        self._log(f"展开完成！")
        self._log(f"展开后行数: {len(expanded_df)}")
        self._log(f"原始总边数: {total_edges_before}")
        self._log(f"去重后总边数: {total_edges_after}")
        self._log(f"去除重复边数: {total_edges_before - total_edges_after}")
        
        if len(df) > 0:
            avg_expansion = len(expanded_df) / len(df)
            self._log(f"平均每行展开为: {avg_expansion:.2f} 条边")
        
        return expanded_df
    
    def save_analysis_report(self, analysis: Dict, output_file: str) -> None:
        """
        保存分析报告到文件
        
        Args:
            analysis: 分析结果字典
            output_file: 输出文件路径
        """
        try:
            # 转换为可序列化的格式
            serializable_analysis = analysis.copy()
            
            # 处理 most_duplicated_row
            if serializable_analysis['most_duplicated_row']:
                dup_row = serializable_analysis['most_duplicated_row']
                if 'duplicates' in dup_row and isinstance(dup_row['duplicates'], dict):
                    dup_row['duplicates'] = dict(dup_row['duplicates'])
            
            # 处理 most_common_duplicates
            if serializable_analysis['most_common_duplicates']:
                serializable_analysis['most_common_duplicates'] = [
                    [str(elem), count] for elem, count in serializable_analysis['most_common_duplicates']
                ]
            
            # 保存为JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_analysis, f, ensure_ascii=False, indent=2)
            
            self._log(f"分析报告已保存到: {output_file}")
            
        except Exception as e:
            self._log(f"保存分析报告时出错: {e}", "ERROR")
    
    def process_file(self, input_file: str, output_file: str = None,
                    src_column: str = 'src', dst_column: str = 'dst',
                    file_delimiter: str = '\t', value_delimiter: str = ',',
                    expand: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        处理整个文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            src_column: 源节点列名
            dst_column: 目标节点列名
            file_delimiter: 文件列分隔符
            value_delimiter: 值内部分隔符
            expand: 是否展开为边缘列表
            
        Returns:
            (deduplicated_df, expanded_df) 元组
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        self._log(f"读取文件: {input_file}")
        
        # 读取文件
        try:
            df = pd.read_csv(input_file, delimiter=file_delimiter)
        except Exception as e:
            raise ValueError(f"读取文件失败: {e}")
        
        self._log(f"文件读取成功，形状: {df.shape}")
        self._log(f"列名: {list(df.columns)}")
        
        # 检查必要列
        if src_column not in df.columns:
            raise ValueError(f"源节点列 '{src_column}' 不存在")
        if dst_column not in df.columns:
            raise ValueError(f"目标节点列 '{dst_column}' 不存在")
        
        # 分析原始数据
        self._log(f"\n分析原始数据重复情况...")
        original_analysis = self.analyze_column_duplicates(df, dst_column, value_delimiter)
        
        # 去重处理
        self._log(f"\n{'='*50}")
        self._log(f"开始去重处理...")
        deduplicated_df = self.deduplicate_column(df, dst_column, value_delimiter, inplace=False)
        
        # 生成输出文件名
        if output_file is None:
            base_name, ext = os.path.splitext(input_file)
            if expand:
                output_file = f"{base_name}_expanded{ext}"
            else:
                output_file = f"{base_name}_deduplicated{ext}"
        
        # 保存去重后的文件
        deduplicated_df.to_csv(output_file, sep=file_delimiter, index=False)
        self._log(f"去重后的数据已保存到: {output_file}")
        
        # 保存分析报告
        report_file = output_file.replace(ext, '_analysis.json')
        self.save_analysis_report(original_analysis, report_file)
        
        # 如果需要展开边缘列表
        expanded_df = None
        if expand:
            self._log(f"\n{'='*50}")
            self._log(f"开始展开边缘列表...")
            expanded_df = self.expand_edgelist(deduplicated_df, src_column, dst_column, value_delimiter)
            
            # 保存展开后的文件
            expanded_file = output_file.replace('_deduplicated', '_expanded').replace('_deduplicated', '_expanded')
            expanded_df.to_csv(expanded_file, sep=file_delimiter, index=False)
            self._log(f"展开后的边缘列表已保存到: {expanded_file}")
        
        return deduplicated_df, expanded_df


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='IMDb数据去重处理工具 - 去除dst列中的重复元素',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本去重处理
  python imdb_deduplicator.py -i input.tsv -o output.tsv
  
  # 处理并展开边缘列表
  python imdb_deduplicator.py -i input.tsv --expand
  
  # 指定自定义分隔符
  python imdb_deduplicator.py -i input.csv --file-delimiter ',' --value-delimiter ';'
  
  # 指定列名
  python imdb_deduplicator.py -i input.tsv --src source --dest destination
  
  # 静默模式
  python imdb_deduplicator.py -i input.tsv --quiet
        """
    )
    
    # 必需参数
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入文件路径'
    )
    
    # 可选参数
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出文件路径（默认：输入文件名_deduplicated.扩展名）'
    )
    
    parser.add_argument(
        '--src',
        type=str,
        default='src',
        help='源节点列名（默认：src）'
    )
    
    parser.add_argument(
        '--dst',
        type=str,
        default='dst',
        help='目标节点列名（默认：dst）'
    )
    
    parser.add_argument(
        '--file-delimiter',
        type=str,
        default='\t',
        help='文件列分隔符（默认：\\t）'
    )
    
    parser.add_argument(
        '--value-delimiter',
        type=str,
        default=',',
        help='值内部分隔符（默认：,）'
    )
    
    parser.add_argument(
        '--expand',
        action='store_true',
        help='展开为边缘列表格式'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，不显示详细日志'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='运行测试示例'
    )
    
    args = parser.parse_args()
    
    # 处理转义字符
    if args.file_delimiter == '\\t':
        args.file_delimiter = '\t'
    
    # 运行测试
    if args.test:
        run_test_example()
        return
    
    # 创建处理器
    deduplicator = IMDBDeduplicator(verbose=not args.quiet)
    
    try:
        # 处理文件
        deduplicated_df, expanded_df = deduplicator.process_file(
            input_file=args.input,
            output_file=args.output,
            src_column=args.src,
            dst_column=args.dst,
            file_delimiter=args.file_delimiter,
            value_delimiter=args.value_delimiter,
            expand=args.expand
        )
        
        print(f"\n{'='*50}")
        print("处理完成！")
        
        if expanded_df is not None:
            print(f"去重后数据行数: {len(deduplicated_df)}")
            print(f"展开后数据行数: {len(expanded_df)}")
        else:
            print(f"最终数据行数: {len(deduplicated_df)}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_test_example():
    """运行测试示例"""
    print("运行测试示例...\n")
    
    # 创建示例数据
    sample_data = {
        'src': [0, 1, 2, 3],
        'dst': [
            '6292913,9822154,9822154,9822154,9822154,9822154,9822154,9822154,10148094,10148094,9822154,9822154',
            '100,200,200,300,300,300',
            '500',
            ''  # 空值
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print("原始数据:")
    print(df)
    print()
    
    # 创建处理器
    deduplicator = IMDBDeduplicator(verbose=True)
    
    # 分析重复情况
    print("=" * 50)
    print("分析重复情况:")
    analysis = deduplicator.analyze_column_duplicates(df, 'dst', ',')
    print(f"总行数: {analysis['total_rows']}")
    print(f"非空行数: {analysis['non_empty_rows']}")
    print(f"总元素数: {analysis['total_elements']}")
    print(f"重复元素数: {analysis['duplicate_elements']}")
    print(f"重复率: {analysis['duplicate_ratio']:.2%}")
    print(f"有重复的行数: {analysis['rows_with_duplicates']}")
    
    if analysis['most_duplicated_row']:
        print(f"\n重复最多的行:")
        dup_info = analysis['most_duplicated_row']
        print(f"  行号: {dup_info['row_index']}")
        print(f"  总元素: {dup_info['total_elements']}")
        print(f"  唯一元素: {dup_info['unique_elements']}")
        print(f"  重复数: {dup_info['duplicate_count']}")
    
    # 去重处理
    print("\n" + "=" * 50)
    print("去重处理:")
    deduplicated_df = deduplicator.deduplicate_column(df, 'dst', ',', inplace=False)
    print("\n去重后的数据:")
    print(deduplicated_df)
    
    # 展开边缘列表
    print("\n" + "=" * 50)
    print("展开边缘列表:")
    expanded_df = deduplicator.expand_edgelist(deduplicated_df, 'src', 'dst', ',')
    print("\n展开后的数据:")
    print(expanded_df)
    
    print("\n" + "=" * 50)
    print("测试完成！")


# 便捷使用函数
def deduplicate_file(input_file: str, output_file: str = None, 
                    src_column: str = 'src', dst_column: str = 'dst',
                    file_delimiter: str = '\t', value_delimiter: str = ',',
                    expand: bool = False, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    便捷函数：处理文件去重
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        src_column: 源节点列名
        dst_column: 目标节点列名
        file_delimiter: 文件列分隔符
        value_delimiter: 值内部分隔符
        expand: 是否展开
        verbose: 是否显示日志
        
    Returns:
        (deduplicated_df, expanded_df) 元组
    """
    deduplicator = IMDBDeduplicator(verbose=verbose)
    return deduplicator.process_file(
        input_file, output_file, src_column, dst_column,
        file_delimiter, value_delimiter, expand
    )


def deduplicate_dataframe(df: pd.DataFrame, column: str = 'dst', 
                         delimiter: str = ',', verbose: bool = True) -> pd.DataFrame:
    """
    便捷函数：处理DataFrame去重
    
    Args:
        df: 输入DataFrame
        column: 目标列名
        delimiter: 分隔符
        verbose: 是否显示日志
        
    Returns:
        去重后的DataFrame
    """
    deduplicator = IMDBDeduplicator(verbose=verbose)
    return deduplicator.deduplicate_column(df, column, delimiter, inplace=False)


if __name__ == "__main__":
    # 直接运行示例或从命令行运行
    import sys
    
    if len(sys.argv) == 1:
        # 没有参数时运行测试
        run_test_example()
    else:
        # 有参数时运行主函数
        sys.exit(main())