import unittest
import os
import shutil
import csv
from pathlib import Path
from process_tables import build_key_mapping, process_and_replace

class TestProcessTables(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy files for testing."""
        self.test_dir = Path("temp_test_data")
        self.output_dir = self.test_dir / "output"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.file1_path = self.test_dir / "table1.csv"
        self.file2_path = self.test_dir / "table2.csv"

        with open(self.file1_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['user_A', 'item_1', '10'])
            writer.writerow(['user_B', 'item_2', '20'])

        with open(self.file2_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['user_A', 'location_X', '100'])
            writer.writerow(['user_C', 'location_Y', '200'])
            
        self.input_files = [str(self.file1_path), str(self.file2_path)]

    def tearDown(self):
        """Clean up the temporary directory and files after tests."""
        shutil.rmtree(self.test_dir)

    def test_build_key_mapping(self):
        """Test if the key mapping is built correctly across multiple files."""
        key_mapping = build_key_mapping(self.input_files, key_column_index=0, delimiter=',')
        
        # Should find 3 unique keys: user_A, user_B, user_C
        self.assertEqual(len(key_mapping), 3)
        self.assertIn('user_A', key_mapping)
        self.assertIn('user_B', key_mapping)
        self.assertIn('user_C', key_mapping)
        
        # Check if IDs are unique and start from 0
        self.assertEqual(sorted(key_mapping.values()), [0, 1, 2])

    def test_process_and_replace(self):
        """Test if keys are replaced and delimiter is changed correctly."""
        key_mapping = {'user_A': 0, 'user_B': 1, 'user_C': 2}
        output_file_path = self.output_dir / "output_table1.tsv"
        
        process_and_replace(
            input_file=str(self.file1_path), 
            output_file=str(output_file_path), 
            key_mapping=key_mapping, 
            key_column_index=0, 
            input_delimiter=',', 
            output_delimiter='\t'
        )

        self.assertTrue(output_file_path.exists())

        with open(output_file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            # Check if user_A was replaced by 0
            self.assertEqual(rows[0], ['0', 'item_1', '10'])
            # Check if user_B was replaced by 1
            self.assertEqual(rows[1], ['1', 'item_2', '20'])

if __name__ == '__main__':
    unittest.main()
