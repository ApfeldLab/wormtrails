"""SQLite storage for wormtrails measurement results.

Uses only stdlib ``sqlite3`` and ``pandas`` (already a dependency).
No additional database drivers or ORMs required.

Schema templates
----------------
The :func:`create_database` function creates a SQLite file with these tables:

``recordings``
    Metadata about each video recording analysed.
``chemotaxis_results``
    Per-worm chemotaxis measurements from :func:`~wormtrails.measure_chemotaxis`.
``trail_measurements``
    Worm trail distances and areas from :func:`~wormtrails.count_simple`.
``healthspan_counts``
    Roaming / quiescent worm counts from :func:`~wormtrails.count_video`.
"""

import sqlite3
from pathlib import Path
import pandas as pd

__all__ = [
    'create_database',
    'write_measurements',
    'read_measurements',
    'add_recording',
    'list_tables',
    'SCHEMA',
]


# ---------------------------------------------------------------------------
# Schema templates — used by create_database() and write_measurements()
# ---------------------------------------------------------------------------

SCHEMA = {
    'recordings': """
        CREATE TABLE IF NOT EXISTS recordings (
            recording_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            timestamp TEXT,
            pixels_per_mm REAL,
            frames_per_second REAL,
            notes TEXT
        )
    """,
    'chemotaxis_results': """
        CREATE TABLE IF NOT EXISTS chemotaxis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER DEFAULT NULL,
            time INTEGER,
            label_id INTEGER,
            y REAL,
            x REAL,
            direction_y REAL,
            direction_x REAL,
            speed REAL,
            speed_mm_s REAL,
            trail_radius_mm REAL,
            worm_radius_mm REAL,
            r REAL,
            r_mm REAL,
            theta REAL,
            relative_angle REAL,
            FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
        )
    """,
    'trail_measurements': """
        CREATE TABLE IF NOT EXISTS trail_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER DEFAULT NULL,
            worm_id INTEGER,
            distance REAL,
            distance_mm REAL,
            area REAL,
            area_mm2 REAL,
            FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
        )
    """,
    'healthspan_counts': """
        CREATE TABLE IF NOT EXISTS healthspan_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER DEFAULT NULL,
            n_roaming INTEGER,
            n_quiescent INTEGER,
            n_total INTEGER,
            FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
        )
    """,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_database(db_path, overwrite=False):
    """Create a new SQLite database with the wormtrails schema tables.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file to create.
    overwrite : bool
        If True and *db_path* already exists, delete it first.
        If False (default) and *db_path* exists, raises ``FileExistsError``.

    Raises
    ------
    FileExistsError
        If *db_path* exists and *overwrite* is False.
    """
    db_path = Path(db_path)
    if db_path.exists():
        if overwrite:
            db_path.unlink()
        else:
            raise FileExistsError(
                f"Database already exists: {db_path}. "
                "Use overwrite=True to replace it."
            )
    conn = sqlite3.connect(str(db_path))
    try:
        for ddl in SCHEMA.values():
            conn.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def write_measurements(df, db_path, table_name, if_exists='append', **kwargs):
    """Write a measurement DataFrame to a SQLite table.

    If *table_name* matches one of the predefined schema tables
    and that table does not yet exist, it is created automatically
    from the schema template.  Otherwise the table is created on the
    fly with columns matching the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Measurement data, typically returned by
        :func:`~wormtrails.measure_chemotaxis`,
        :func:`~wormtrails.count_simple`, or similar.
    db_path : str or Path
        Path to a wormtrails SQLite database.
    table_name : str
        Target table (e.g. ``'chemotaxis_results'``,
        ``'trail_measurements'``, ``'healthspan_counts'``).
    if_exists : str
        Behaviour when the table already exists:
        ``'append'`` (default), ``'replace'``, or ``'fail'``.
    **kwargs
        Extra arguments forwarded to :meth:`pandas.DataFrame.to_sql`.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_table(conn, table_name)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False, **kwargs)
        conn.commit()
    finally:
        conn.close()


def read_measurements(db_path, table_name, **kwargs):
    """Read a measurement table into a DataFrame.

    Parameters
    ----------
    db_path : str or Path
        Path to a wormtrails SQLite database.
    table_name : str
        Table name to query.
    **kwargs
        Extra arguments forwarded to :func:`pandas.read_sql`.

    Returns
    -------
    pandas.DataFrame
    """
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn, **kwargs)
    finally:
        conn.close()


def add_recording(db_path, source_file=None, timestamp=None,
                  pixels_per_mm=None, frames_per_second=None, notes=None):
    """Insert a new recording entry and return its ``recording_id``.

    Parameters
    ----------
    db_path : str or Path
        Path to a wormtrails SQLite database.
    source_file : str, optional
        Original video file name or path.
    timestamp : str, optional
        ISO-formatted timestamp or other date string.
    pixels_per_mm : float, optional
        Calibration factor used for this recording.
    frames_per_second : float, optional
        Frame rate of the original video.
    notes : str, optional
        Free-text notes.

    Returns
    -------
    int
        The ``recording_id`` of the newly inserted row.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_table(conn, 'recordings')
        cur = conn.execute(
            """INSERT INTO recordings
               (source_file, timestamp, pixels_per_mm, frames_per_second, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (source_file, timestamp, pixels_per_mm, frames_per_second, notes),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_tables(db_path):
    """Return the list of user table names in the database.

    Parameters
    ----------
    db_path : str or Path
        Path to a SQLite database.

    Returns
    -------
    list of str
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_table(conn, table_name):
    """Create *table_name* from the schema template if it doesn't exist.

    Raises a KeyError if *table_name* is not a recognised schema table and
    does not already exist, to prevent silent misconfiguration.
    """
    if table_name in SCHEMA:
        conn.execute(SCHEMA[table_name])
    else:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if not cur.fetchone():
            raise KeyError(
                f"Unknown table '{table_name}'. "
                f"Predefined tables: {', '.join(SCHEMA.keys())}. "
                "To create an ad-hoc table, call pandas.to_sql directly on the connection."
            )
