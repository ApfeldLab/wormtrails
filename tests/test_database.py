import unittest
import tempfile
import os
import pandas as pd
from wormtrails.database import (
    create_database, write_measurements, read_measurements,
    add_recording, list_tables, SCHEMA,
)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        os.unlink(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_database(self):
        create_database(self.db_path)
        tables = list_tables(self.db_path)
        for name in SCHEMA:
            self.assertIn(name, tables)

    def test_create_database_overwrite(self):
        create_database(self.db_path)
        # should raise by default
        with self.assertRaises(FileExistsError):
            create_database(self.db_path)
        # should succeed with overwrite
        create_database(self.db_path, overwrite=True)

    def test_write_and_read_measurements_known_table(self):
        create_database(self.db_path)
        df = pd.DataFrame({
            'worm_id': [1, 2],
            'distance': [10.0, 20.0],
            'area': [100, 200],
        })
        write_measurements(df, self.db_path, 'trail_measurements')
        result = read_measurements(self.db_path, 'trail_measurements')
        self.assertEqual(len(result), 2)

    def test_write_measurements_unknown_table_raises(self):
        create_database(self.db_path)
        df = pd.DataFrame({'x': [1]})
        with self.assertRaises(KeyError):
            write_measurements(df, self.db_path, 'nonexistent_table')

    def test_add_recording_and_read(self):
        create_database(self.db_path)
        rid = add_recording(
            self.db_path,
            source_file='test.avi',
            pixels_per_mm=10.0,
            frames_per_second=30.0,
        )
        self.assertIsInstance(rid, int)
        self.assertGreater(rid, 0)

    def test_list_tables_empty_db(self):
        create_database(self.db_path)
        tables = list_tables(self.db_path)
        self.assertIn('recordings', tables)

    def test_read_measurements_empty_table(self):
        create_database(self.db_path)
        result = read_measurements(self.db_path, 'chemotaxis_results')
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()
