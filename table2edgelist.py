"""
IMDb图数据生成器
基于已处理的IMDb数据文件生成图数据的边列表(edgelist)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
import os
import argparse


class IMDbGraphGenerator:
    """IMDb图数据生成器类"""
    
    def __init__(self, data_dir: str = "."):
        """
        初始化图数据生成器
        
        Args:
            data_dir: 包含已处理数据文件的目录
        """
        self.data_dir = data_dir
        self.nconst_map: Dict[int, str] = {}  # int_id -> original_nconst
        self.tconst_map: Dict[int, str] = {}  # int_id -> original_tconst
        
    def load_mappings(self) -> None:
        """
        加载nconst和tconst的映射文件
        """
        print("加载映射文件...")
        
        # 加载nconst映射
        nconst_file = os.path.join(self.data_dir, 'nconst_mapping.tsv')
        if os.path.exists(nconst_file):
            nconst_df = pd.read_csv(nconst_file, sep='\t')
            self.nconst_map = dict(zip(nconst_df['int_id'], nconst_df['original_nconst']))
            print(f"  加载了 {len(self.nconst_map)} 个nconst映射")
        else:
            print(f"  警告: nconst_mapping.tsv 文件不存在于 {self.data_dir}")
        
        # 加载tconst映射
        tconst_file = os.path.join(self.data_dir, 'tconst_mapping.tsv')
        if os.path.exists(tconst_file):
            tconst_df = pd.read_csv(tconst_file, sep='\t')
            self.tconst_map = dict(zip(tconst_df['int_id'], tconst_df['original_tconst']))
            print(f"  加载了 {len(self.tconst_map)} 个tconst映射")
        else:
            print(f"  警告: tconst_mapping.tsv 文件不存在于 {self.data_dir}")
    
    def generate_actor_movie_edges(self) -> List[Tuple[int, int, str]]:
        """
        从title.principals_processed.tsv生成演员-电影边
        
        Returns:
            边的列表，每个元素为(src, dst, relationship_type)
        """
        edges = []
        principals_file = os.path.join(self.data_dir, 'title.principals_processed.tsv')
        
        if not os.path.exists(principals_file):
            print(f"  警告: {principals_file} 文件不存在")
            return edges
        
        print("处理title.principals_processed.tsv...")
        try:
            principals_df = pd.read_csv(principals_file, sep='\t', low_memory=False)
            
            # 确保列名正确
            if 'nconst' not in principals_df.columns or 'tconst' not in principals_df.columns:
                print(f"  错误: 文件缺少必要列，实际列: {list(principals_df.columns)}")
                return edges
            
            for _, row in principals_df.iterrows():
                try:
                    src = int(row['nconst'])  # nconst int_id
                    dst = int(row['tconst'])  # tconst int_id
                    category = str(row['category']) if pd.notna(row.get('category')) else "unknown"
                    
                    edges.append((src, dst, f"acted_in_{category}"))
                except (ValueError, TypeError) as e:
                    # 跳过无效数据
                    continue
                    
            print(f"  生成了 {len(edges)} 条演员-电影边")
            
        except Exception as e:
            print(f"  处理文件时出错: {e}")
        
        return edges
    
    def generate_director_movie_edges(self) -> List[Tuple[int, int, str]]:
        """
        从title.crew_processed.tsv生成导演-电影边
        
        Returns:
            边的列表，每个元素为(src, dst, relationship_type)
        """
        edges = []
        crew_file = os.path.join(self.data_dir, 'title.crew_processed.tsv')
        
        if not os.path.exists(crew_file):
            print(f"  警告: {crew_file} 文件不存在")
            return edges
        
        print("处理title.crew_processed.tsv...")
        try:
            crew_df = pd.read_csv(crew_file, sep='\t', low_memory=False)
            
            if 'tconst' not in crew_df.columns or 'directors' not in crew_df.columns:
                print(f"  错误: 文件缺少必要列，实际列: {list(crew_df.columns)}")
                return edges
            
            for _, row in crew_df.iterrows():
                try:
                    dst = int(row['tconst'])  # 电影ID
                    directors_str = row['directors']
                    
                    if pd.notna(directors_str):
                        # 处理逗号分隔的导演ID
                        director_ids = str(directors_str).split(',')
                        for director_id in director_ids:
                            if director_id.strip():  # 非空字符串
                                src = int(director_id.strip())
                                edges.append((src, dst, "directed"))
                except (ValueError, TypeError) as e:
                    # 跳过无效数据
                    continue
            
            print(f"  生成了 {len(edges)} 条导演-电影边")
            
        except Exception as e:
            print(f"  处理文件时出错: {e}")
        
        return edges
    
    def generate_writer_movie_edges(self) -> List[Tuple[int, int, str]]:
        """
        从title.crew_processed.tsv生成编剧-电影边
        
        Returns:
            边的列表，每个元素为(src, dst, relationship_type)
        """
        edges = []
        crew_file = os.path.join(self.data_dir, 'title.crew_processed.tsv')
        
        if not os.path.exists(crew_file):
            return edges
        
        print("处理title.crew_processed.tsv中的编剧关系...")
        try:
            crew_df = pd.read_csv(crew_file, sep='\t', low_memory=False)
            
            if 'tconst' not in crew_df.columns or 'writers' not in crew_df.columns:
                return edges
            
            for _, row in crew_df.iterrows():
                try:
                    dst = int(row['tconst'])  # 电影ID
                    writers_str = row['writers']
                    
                    if pd.notna(writers_str):
                        # 处理逗号分隔的编剧ID
                        writer_ids = str(writers_str).split(',')
                        for writer_id in writer_ids:
                            if writer_id.strip():  # 非空字符串
                                src = int(writer_id.strip())
                                edges.append((src, dst, "wrote"))
                except (ValueError, TypeError):
                    continue
            
            print(f"  生成了 {len(edges)} 条编剧-电影边")
            
        except Exception as e:
            print(f"  处理文件时出错: {e}")
        
        return edges
    
    def generate_episode_edges(self) -> List[Tuple[int, int, str]]:
        """
        从title.episode_processed.tsv生成剧集-系列边
        
        Returns:
            边的列表，每个元素为(src, dst, relationship_type)
        """
        edges = []
        episode_file = os.path.join(self.data_dir, 'title.episode_processed.tsv')
        
        if not os.path.exists(episode_file):
            print(f"  警告: {episode_file} 文件不存在")
            return edges
        
        print("处理title.episode_processed.tsv...")
        try:
            episode_df = pd.read_csv(episode_file, sep='\t', low_memory=False)
            
            if 'tconst' not in episode_df.columns or 'parentTconst' not in episode_df.columns:
                print(f"  错误: 文件缺少必要列，实际列: {list(episode_df.columns)}")
                return edges
            
            for _, row in episode_df.iterrows():
                try:
                    src = int(row['tconst'])  # 剧集ID
                    parent_id = row['parentTconst']
                    
                    if pd.notna(parent_id):
                        dst = int(parent_id)  # 系列/季ID
                        edges.append((src, dst, "episode_of"))
                except (ValueError, TypeError):
                    continue
            
            print(f"  生成了 {len(edges)} 条剧集-系列边")
            
        except Exception as e:
            print(f"  处理文件时出错: {e}")
        
        return edges
    
    def generate_all_edges(self) -> pd.DataFrame:
        """
        生成所有类型的边并合并为一个DataFrame
        
        Returns:
            包含所有边的DataFrame
        """
        print("开始生成图数据边列表...")
        
        # 加载映射
        self.load_mappings()
        
        # 生成各种类型的边
        all_edges = []
        
        # 1. 演员-电影边
        actor_edges = self.generate_actor_movie_edges()
        all_edges.extend(actor_edges)
        
        # 2. 导演-电影边
        director_edges = self.generate_director_movie_edges()
        all_edges.extend(director_edges)
        
        # 3. 编剧-电影边
        writer_edges = self.generate_writer_movie_edges()
        all_edges.extend(writer_edges)
        
        # 4. 剧集-系列边
        episode_edges = self.generate_episode_edges()
        all_edges.extend(episode_edges)
        
        # 转换为DataFrame
        edges_df = pd.DataFrame(all_edges, columns=['src', 'dst', 'relationship'])
        
        # 添加原始ID列（可选）
        if self.nconst_map and self.tconst_map:
            print("添加原始ID信息...")
            edges_df['src_original'] = edges_df['src'].map(self.nconst_map)
            edges_df['dst_original'] = edges_df['dst'].map(self.tconst_map)
        
        print(f"总共生成了 {len(edges_df)} 条边")
        
        # 统计关系类型
        if not edges_df.empty:
            print("\n边类型统计:")
            rel_counts = edges_df['relationship'].value_counts()
            for rel, count in rel_counts.items():
                print(f"  {rel}: {count} 条边")
        
        return edges_df
    
    def save_edgelist(self, edges_df: pd.DataFrame, output_file: str, 
                     include_original_ids: bool = False) -> None:
        """
        保存边列表到文件
        
        Args:
            edges_df: 边数据DataFrame
            output_file: 输出文件路径
            include_original_ids: 是否包含原始ID
        """
        if edges_df.empty:
            print("警告: 边数据为空，不保存文件")
            return
        
        print(f"保存边列表到 {output_file}...")
        
        # 选择要保存的列
        if include_original_ids and 'src_original' in edges_df.columns and 'dst_original' in edges_df.columns:
            columns_to_save = ['src', 'dst', 'relationship', 'src_original', 'dst_original']
        else:
            columns_to_save = ['src', 'dst', 'relationship']
        
        # 保存为TSV文件
        edges_df[columns_to_save].to_csv(output_file, sep='\t', index=False)
        
        print(f"已保存 {len(edges_df)} 条边到 {output_file}")
        
        # 统计信息
        unique_src = edges_df['src'].nunique()
        unique_dst = edges_df['dst'].nunique()
        print(f"  唯一源节点(nconst): {unique_src}")
        print(f"  唯一目标节点(tconst): {unique_dst}")
    
    def save_multiple_formats(self, edges_df: pd.DataFrame, output_dir: str, 
                             base_name: str = "imdb_edgelist") -> None:
        """
        保存多种格式的图数据文件
        
        Args:
            edges_df: 边数据DataFrame
            output_dir: 输出目录
            base_name: 基础文件名
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存完整的TSV文件（包含关系类型和原始ID）
        full_file = os.path.join(output_dir, f"{base_name}_full.tsv")
        self.save_edgelist(edges_df, full_file, include_original_ids=True)
        
        # 2. 保存简单的边列表格式（src dst，用于大多数图分析工具）
        simple_file = os.path.join(output_dir, f"{base_name}_simple.tsv")
        edges_df[['src', 'dst']].to_csv(simple_file, sep='\t', index=False, header=False)
        print(f"保存简单边列表到 {simple_file}")
        
        # 3. 保存带权重的边列表（按关系类型统计）
        weighted_file = os.path.join(output_dir, f"{base_name}_weighted.tsv")
        weight_df = edges_df.groupby(['src', 'dst']).size().reset_index(name='weight')
        weight_df.to_csv(weighted_file, sep='\t', index=False)
        print(f"保存带权重边列表到 {weighted_file}")
        
        # 4. 保存邻接表格式
        adjacency_file = os.path.join(output_dir, f"{base_name}_adjacency.tsv")
        adjacency_data = edges_df.groupby('src')['dst'].apply(list).reset_index()
        adjacency_data['dst'] = adjacency_data['dst'].apply(lambda x: ','.join(map(str, x)))
        adjacency_data.to_csv(adjacency_file, sep='\t', index=False)
        print(f"保存邻接表到 {adjacency_file}")
        
        # 5. 保存节点元数据
        self.save_node_metadata(edges_df, output_dir)
    
    def save_node_metadata(self, edges_df: pd.DataFrame, output_dir: str) -> None:
        """
        保存节点元数据文件
        
        Args:
            edges_df: 边数据DataFrame
            output_dir: 输出目录
        """
        print("生成节点元数据...")
        
        # 收集所有节点
        all_nodes = set(edges_df['src'].unique()).union(set(edges_df['dst'].unique()))
        
        # 区分节点类型
        nconst_nodes = set(edges_df['src'].unique())  # nconst节点
        tconst_nodes = set(edges_df['dst'].unique())  # tconst节点
        
        # 加载name.basics数据获取人员信息
        name_file = os.path.join(self.data_dir, 'name.basics_processed.tsv')
        name_info = {}
        if os.path.exists(name_file):
            try:
                name_df = pd.read_csv(name_file, sep='\t', low_memory=False, nrows=10000)  # 只读取部分用于示例
                if 'nconst' in name_df.columns and 'primaryName' in name_df.columns:
                    for _, row in name_df.iterrows():
                        node_id = row['nconst']
                        if isinstance(node_id, (int, np.integer)):
                            name_info[int(node_id)] = {
                                'name': row.get('primaryName', ''),
                                'type': 'person'
                            }
            except Exception as e:
                print(f"  加载人员信息时出错: {e}")
        
        # 加载title.basics数据获取电影信息
        title_file = os.path.join(self.data_dir, 'title.basics_processed.tsv')
        title_info = {}
        if os.path.exists(title_file):
            try:
                title_df = pd.read_csv(title_file, sep='\t', low_memory=False, nrows=10000)  # 只读取部分
                if 'tconst' in title_df.columns and 'primaryTitle' in title_df.columns:
                    for _, row in title_df.iterrows():
                        node_id = row['tconst']
                        if isinstance(node_id, (int, np.integer)):
                            title_info[int(node_id)] = {
                                'title': row.get('primaryTitle', ''),
                                'type': 'title'
                            }
            except Exception as e:
                print(f"  加载电影信息时出错: {e}")
        
        # 创建节点元数据DataFrame
        nodes_data = []
        for node_id in all_nodes:
            node_type = 'person' if node_id in nconst_nodes else 'title'
            
            # 尝试获取名称/标题
            name = ''
            if node_id in name_info:
                name = name_info[node_id]['name']
            elif node_id in title_info:
                name = title_info[node_id]['title']
            
            # 添加原始ID
            original_id = ''
            if node_type == 'person' and self.nconst_map:
                original_id = self.nconst_map.get(node_id, '')
            elif node_type == 'title' and self.tconst_map:
                original_id = self.tconst_map.get(node_id, '')
            
            nodes_data.append({
                'node_id': node_id,
                'node_type': node_type,
                'name': name,
                'original_id': original_id
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_file = os.path.join(output_dir, 'imdb_nodes_metadata.tsv')
        nodes_df.to_csv(nodes_file, sep='\t', index=False)
        print(f"保存节点元数据到 {nodes_file} ({len(nodes_df)} 个节点)")
        
        # 统计节点类型
        if not nodes_df.empty:
            node_type_counts = nodes_df['node_type'].value_counts()
            print(f"  人员节点: {node_type_counts.get('person', 0)}")
            print(f"  电影节点: {node_type_counts.get('title', 0)}")


def main():
    """主函数 - 参数化版本"""
    parser = argparse.ArgumentParser(
        description='IMDb图数据生成器 - 从已处理的IMDb数据生成图边列表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python imdb_graph_generator.py --input ./processed_data --output ./graph_data
  python imdb_graph_generator.py -i ./processed -o ./graph --formats all
  python imdb_graph_generator.py --input ./data --output ./output --simple-only
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='已处理数据文件目录路径（必须）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./imdb_graph',
        help='输出目录路径，默认为 ./imdb_graph'
    )
    
    parser.add_argument(
        '--formats',
        type=str,
        choices=['all', 'simple', 'full', 'weighted'],
        default='all',
        help='输出文件格式：all(全部), simple(简单边列表), full(完整边列表), weighted(带权重)'
    )
    
    parser.add_argument(
        '--simple-only',
        action='store_true',
        help='只生成简单边列表格式(src dst)，忽略其他格式'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细处理信息'
    )
    
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='不生成节点元数据文件'
    )
    
    parser.add_argument(
        '--edge-types',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'actor', 'director', 'writer', 'episode'],
        help='指定生成的边类型'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 验证输入目录是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.verbose:
        print(f"输入目录: {args.input}")
        print(f"输出目录: {args.output}")
        print(f"输出格式: {args.formats}")
        print(f"简单模式: {args.simple_only}")
        print(f"边类型: {args.edge_types}")
        print("-" * 50)
    
    # 创建图生成器
    generator = IMDbGraphGenerator(data_dir=args.input)
    
    # 生成边数据
    edges_df = generator.generate_all_edges()
    
    if edges_df.empty:
        print("错误: 未生成任何边数据")
        return
    
    # 根据参数保存不同格式
    if args.simple_only:
        # 只保存简单格式
        simple_file = os.path.join(args.output, 'imdb_edgelist.tsv')
        edges_df[['src', 'dst']].to_csv(simple_file, sep='\t', index=False, header=False)
        print(f"保存简单边列表到 {simple_file}")
    elif args.formats == 'all':
        # 保存所有格式
        generator.save_multiple_formats(edges_df, args.output)
    elif args.formats == 'simple':
        simple_file = os.path.join(args.output, 'imdb_edgelist_simple.tsv')
        edges_df[['src', 'dst']].to_csv(simple_file, sep='\t', index=False, header=False)
        print(f"保存简单边列表到 {simple_file}")
    elif args.formats == 'full':
        full_file = os.path.join(args.output, 'imdb_edgelist_full.tsv')
        generator.save_edgelist(edges_df, full_file, include_original_ids=True)
    elif args.formats == 'weighted':
        weighted_file = os.path.join(args.output, 'imdb_edgelist_weighted.tsv')
        weight_df = edges_df.groupby(['src', 'dst']).size().reset_index(name='weight')
        weight_df.to_csv(weighted_file, sep='\t', index=False)
        print(f"保存带权重边列表到 {weighted_file}")
    
    # 生成节点元数据（除非指定不生成）
    if not args.no_metadata and not args.simple_only:
        generator.save_node_metadata(edges_df, args.output)
    
    print(f"\n图数据生成完成！文件保存在: {args.output}")
    
    # 显示输出文件列表
    if args.verbose:
        print("\n生成的文件:")
        for file in sorted(os.listdir(args.output)):
            file_path = os.path.join(args.output, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {file}: {size:,} bytes")


def quick_generate(input_dir: str, output_dir: str = None) -> pd.DataFrame:
    """
    快速生成图数据的便捷函数
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选，如果不提供则只返回DataFrame）
        
    Returns:
        边数据的DataFrame
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'graph_output')
    
    generator = IMDbGraphGenerator(data_dir=input_dir)
    edges_df = generator.generate_all_edges()
    
    if not edges_df.empty:
        generator.save_edgelist(edges_df, os.path.join(output_dir, 'imdb_edgelist.tsv'))
    
    return edges_df


if __name__ == "__main__":
    main()