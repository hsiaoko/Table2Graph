import pandas as pd
from typing import Dict, Tuple
import os


class IMDbDataProcessor:
    """IMDb数据处理类，用于将tconst和nconst替换为整数ID"""

    def __init__(self, data_dir: str = "."):
        """
        初始化处理器

        Args:
            data_dir: 数据文件所在目录
        """
        self.data_dir = data_dir
        self.tconst_map: Dict[str, int] = {}  # tconst -> 整数ID
        self.nconst_map: Dict[str, int] = {}  # nconst -> 整数ID
        self.next_tconst_id = 0
        self.next_nconst_id = 0

    def _extract_tconst_from_dataframe(self, df: pd.DataFrame, column: str = 'tconst') -> None:
        """
        从DataFrame中提取tconst并建立映射

        Args:
            df: 包含tconst列的DataFrame
            column: tconst列名
        """
        if column in df.columns:
            unique_tconsts = df[column].dropna().unique()
            for tconst in unique_tconsts:
                if tconst not in self.tconst_map:
                    self.tconst_map[tconst] = self.next_tconst_id
                    self.next_tconst_id += 1

    def _extract_nconst_from_dataframe(self, df: pd.DataFrame, column: str = 'nconst') -> None:
        """
        从DataFrame中提取nconst并建立映射

        Args:
            df: 包含nconst列的DataFrame
            column: nconst列名
        """
        if column in df.columns:
            unique_nconsts = df[column].dropna().unique()
            for nconst in unique_nconsts:
                if nconst not in self.nconst_map:
                    self.nconst_map[nconst] = self.next_nconst_id
                    self.next_nconst_id += 1

    def _extract_comma_separated_consts(self, df: pd.DataFrame, column: str) -> None:
        """
        处理逗号分隔的const字段（如directors, writers列）

        Args:
            df: 包含逗号分隔const列的DataFrame
            column: 列名
        """
        if column in df.columns:
            # 分割逗号分隔的值
            all_consts = []
            for value in df[column].dropna():
                if pd.notna(value):
                    # 分割多个const值
                    consts = str(value).split(',')
                    all_consts.extend(consts)

            # 去重并建立映射
            unique_consts = set(all_consts)
            for const in unique_consts:
                if const.startswith('tt'):
                    if const not in self.tconst_map:
                        self.tconst_map[const] = self.next_tconst_id
                        self.next_tconst_id += 1
                elif const.startswith('nm'):
                    if const not in self.nconst_map:
                        self.nconst_map[const] = self.next_nconst_id
                        self.next_nconst_id += 1

    def build_mappings(self) -> None:
        """
        从所有数据文件中构建tconst和nconst的映射关系
        """
        print("开始构建映射关系...")

        # 1. 处理name.basics.tsv
        print("处理name.basics.tsv...")
        name_file = os.path.join(self.data_dir, 'name.basics.tsv')
        if os.path.exists(name_file):
            name_df = pd.read_csv(name_file, sep='\t', low_memory=False)
            self._extract_nconst_from_dataframe(name_df, 'nconst')

        # 2. 处理akas.tsv
        print("处理akas.tsv...")
        akas_file = os.path.join(self.data_dir, 'akas.tsv')
        if os.path.exists(akas_file):
            akas_df = pd.read_csv(akas_file, sep='\t', low_memory=False)
            self._extract_tconst_from_dataframe(akas_df, 'titleId')

        # 3. 处理title.basics.tsv
        print("处理title.basics.tsv...")
        basics_file = os.path.join(self.data_dir, 'title.basics.tsv')
        if os.path.exists(basics_file):
            basics_df = pd.read_csv(basics_file, sep='\t', low_memory=False)
            self._extract_tconst_from_dataframe(basics_df, 'tconst')

        # 4. 处理title.crew.tsv
        print("处理title.crew.tsv...")
        crew_file = os.path.join(self.data_dir, 'title.crew.tsv')
        if os.path.exists(crew_file):
            crew_df = pd.read_csv(crew_file, sep='\t', low_memory=False)
            self._extract_tconst_from_dataframe(crew_df, 'tconst')
            self._extract_comma_separated_consts(crew_df, 'directors')
            self._extract_comma_separated_consts(crew_df, 'writers')

        # 5. 处理title.episode.tsv
        print("处理title.episode.tsv...")
        episode_file = os.path.join(self.data_dir, 'title.episode.tsv')
        if os.path.exists(episode_file):
            episode_df = pd.read_csv(episode_file, sep='\t', low_memory=False)
            self._extract_tconst_from_dataframe(episode_df, 'tconst')
            self._extract_tconst_from_dataframe(episode_df, 'parentTconst')

        # 6. 处理title.principals.tsv
        print("处理title.principals.tsv...")
        principals_file = os.path.join(self.data_dir, 'title.principals.tsv')
        if os.path.exists(principals_file):
            principals_df = pd.read_csv(principals_file, sep='\t', low_memory=False)
            self._extract_tconst_from_dataframe(principals_df, 'tconst')
            self._extract_nconst_from_dataframe(principals_df, 'nconst')

        # 7. 处理title.ratings.tsv
        print("处理title.ratings.tsv...")
        ratings_file = os.path.join(self.data_dir, 'title.ratings.tsv')
        if os.path.exists(ratings_file):
            ratings_df = pd.read_csv(ratings_file, sep='\t', low_memory=False)
            self._extract_tconst_from_dataframe(ratings_df, 'tconst')

        print(f"映射关系构建完成！tconst数量: {len(self.tconst_map)}, nconst数量: {len(self.nconst_map)}")

    def replace_tconst_in_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        替换DataFrame中指定列的tconst值为整数ID

        Args:
            df: 要处理的DataFrame
            column: 包含tconst的列名

        Returns:
            处理后的DataFrame
        """
        if column in df.columns:
            # 创建副本以避免修改原始数据
            result_df = df.copy()

            # 替换函数
            def replace_value(x):
                if pd.isna(x):
                    return x
                if x in self.tconst_map:
                    return self.tconst_map[x]
                return x

            result_df[column] = result_df[column].apply(replace_value)
            return result_df
        return df

    def replace_nconst_in_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        替换DataFrame中指定列的nconst值为整数ID

        Args:
            df: 要处理的DataFrame
            column: 包含nconst的列名

        Returns:
            处理后的DataFrame
        """
        if column in df.columns:
            # 创建副本以避免修改原始数据
            result_df = df.copy()

            # 替换函数
            def replace_value(x):
                if pd.isna(x):
                    return x
                if x in self.nconst_map:
                    return self.nconst_map[x]
                return x

            result_df[column] = result_df[column].apply(replace_value)
            return result_df
        return df

    def replace_comma_separated_consts(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        替换逗号分隔的const列中的值为整数ID

        Args:
            df: 要处理的DataFrame
            column: 包含逗号分隔const的列名

        Returns:
            处理后的DataFrame
        """
        if column in df.columns:
            result_df = df.copy()

            def replace_comma_values(x):
                if pd.isna(x):
                    return x

                consts = str(x).split(',')
                replaced_consts = []
                for const in consts:
                    if const.startswith('tt') and const in self.tconst_map:
                        replaced_consts.append(str(self.tconst_map[const]))
                    elif const.startswith('nm') and const in self.nconst_map:
                        replaced_consts.append(str(self.nconst_map[const]))
                    else:
                        replaced_consts.append(const)

                return ','.join(replaced_consts)

            result_df[column] = result_df[column].apply(replace_comma_values)
            return result_df
        return df

    def process_all_files(self, output_dir: str = "processed") -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        处理所有数据文件，替换tconst和nconst为整数ID

        Args:
            output_dir: 输出目录

        Returns:
            tconst_map, nconst_map 映射字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 先构建映射关系
        self.build_mappings()

        # 保存映射关系
        self.save_mappings(output_dir)

        # 处理每个文件
        self.process_name_basics(output_dir)
        self.process_akas(output_dir)
        self.process_title_basics(output_dir)
        self.process_title_crew(output_dir)
        self.process_title_episode(output_dir)
        self.process_title_principals(output_dir)
        self.process_title_ratings(output_dir)

        return self.tconst_map, self.nconst_map

    def save_mappings(self, output_dir: str) -> None:
        """
        保存映射关系到文件

        Args:
            output_dir: 输出目录
        """
        # 保存tconst映射
        tconst_df = pd.DataFrame(list(self.tconst_map.items()), columns=['original_tconst', 'int_id'])
        tconst_file = os.path.join(output_dir, 'tconst_mapping.tsv')
        tconst_df.to_csv(tconst_file, sep='\t', index=False)
        print(f"tconst映射已保存到: {tconst_file}")

        # 保存nconst映射
        nconst_df = pd.DataFrame(list(self.nconst_map.items()), columns=['original_nconst', 'int_id'])
        nconst_file = os.path.join(output_dir, 'nconst_mapping.tsv')
        nconst_df.to_csv(nconst_file, sep='\t', index=False)
        print(f"nconst映射已保存到: {nconst_file}")

    def process_name_basics(self, output_dir: str) -> None:
        """处理name.basics.tsv文件"""
        input_file = os.path.join(self.data_dir, 'name.basics.tsv')
        output_file = os.path.join(output_dir, 'name.basics_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换nconst列
            df = self.replace_nconst_in_dataframe(df, 'nconst')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")

    def process_akas(self, output_dir: str) -> None:
        """处理akas.tsv文件"""
        input_file = os.path.join(self.data_dir, 'akas.tsv')
        output_file = os.path.join(output_dir, 'akas_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换titleId列（在akas文件中是tconst）
            df = self.replace_tconst_in_dataframe(df, 'titleId')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")

    def process_title_basics(self, output_dir: str) -> None:
        """处理title.basics.tsv文件"""
        input_file = os.path.join(self.data_dir, 'title.basics.tsv')
        output_file = os.path.join(output_dir, 'title.basics_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换tconst列
            df = self.replace_tconst_in_dataframe(df, 'tconst')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")

    def process_title_crew(self, output_dir: str) -> None:
        """处理title.crew.tsv文件"""
        input_file = os.path.join(self.data_dir, 'title.crew.tsv')
        output_file = os.path.join(output_dir, 'title.crew_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换tconst列
            df = self.replace_tconst_in_dataframe(df, 'tconst')

            # 替换directors和writers列中的逗号分隔值
            df = self.replace_comma_separated_consts(df, 'directors')
            df = self.replace_comma_separated_consts(df, 'writers')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")

    def process_title_episode(self, output_dir: str) -> None:
        """处理title.episode.tsv文件"""
        input_file = os.path.join(self.data_dir, 'title.episode.tsv')
        output_file = os.path.join(output_dir, 'title.episode_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换tconst和parentTconst列
            df = self.replace_tconst_in_dataframe(df, 'tconst')
            df = self.replace_tconst_in_dataframe(df, 'parentTconst')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")

    def process_title_principals(self, output_dir: str) -> None:
        """处理title.principals.tsv文件"""
        input_file = os.path.join(self.data_dir, 'title.principals.tsv')
        output_file = os.path.join(output_dir, 'title.principals_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换tconst和nconst列
            df = self.replace_tconst_in_dataframe(df, 'tconst')
            df = self.replace_nconst_in_dataframe(df, 'nconst')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")

    def process_title_ratings(self, output_dir: str) -> None:
        """处理title.ratings.tsv文件"""
        input_file = os.path.join(self.data_dir, 'title.ratings.tsv')
        output_file = os.path.join(output_dir, 'title.ratings_processed.tsv')

        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            df = pd.read_csv(input_file, sep='\t', low_memory=False)

            # 替换tconst列
            df = self.replace_tconst_in_dataframe(df, 'tconst')

            # 保存处理后的文件
            df.to_csv(output_file, sep='\t', index=False)
            print(f"已保存到 {output_file}")


def save_const_mappings(tconst_map: Dict[str, int], nconst_map: Dict[str, int],
                        output_dir: str = ".") -> None:
    """
    单独保存const映射关系

    Args:
        tconst_map: tconst映射字典
        nconst_map: nconst映射字典
        output_dir: 输出目录
    """
    import pandas as pd
    import os

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存tconst映射
    tconst_df = pd.DataFrame([
        {"original_id": k, "int_id": v, "type": "tconst"}
        for k, v in tconst_map.items()
    ])

    # 保存nconst映射
    nconst_df = pd.DataFrame([
        {"original_id": k, "int_id": v, "type": "nconst"}
        for k, v in nconst_map.items()
    ])

    # 也可以保存为单独的文件
    tconst_df.to_csv(os.path.join(output_dir, "tconst_mapping.tsv"), sep='\t', index=False)
    nconst_df.to_csv(os.path.join(output_dir, "nconst_mapping.tsv"), sep='\t', index=False)

    # 或者保存为合并的文件
    combined_df = pd.concat([tconst_df, nconst_df], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, "const_mappings.tsv"), sep='\t', index=False)

    print(f"映射关系已保存到 {output_dir} 目录")

def main():
    """主函数"""
    # 创建处理器实例
    processor = IMDbDataProcessor(data_dir="./imdb_data")  # 假设数据在imdb_data目录下

    # 处理所有文件
    tconst_map, nconst_map = processor.process_all_files(output_dir="./processed_imdb")

    print("处理完成！")
    print(f"共处理 {len(tconst_map)} 个tconst和 {len(nconst_map)} 个nconst")


if __name__ == "__main__":
    main()