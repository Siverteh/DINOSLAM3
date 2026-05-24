from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class SequenceMetric:
    sequence: str
    status: str
    ate_rmse: float
    rpe_trans_rmse: float
    rpe_rot_rmse: float
    coverage: float
    ate_rmse_associated: float = math.nan


class ExperimentStore:
    """Small SQLite logger for experiment lineage and metrics."""

    def __init__(self, db_path: str | Path) -> None:
        dbp = Path(db_path).expanduser()
        if not dbp.is_absolute():
            dbp = (Path.cwd() / dbp).resolve()
        self.db_path = dbp
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    run_name TEXT,
                    prompt_tag TEXT,
                    parent_id TEXT,
                    git_commit TEXT,
                    config_hash TEXT,
                    created_utc TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    epoch INTEGER,
                    path TEXT NOT NULL,
                    selected_flag INTEGER NOT NULL DEFAULT 0,
                    selection_score REAL,
                    created_utc TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_exp ON checkpoints(experiment_id);

                CREATE TABLE IF NOT EXISTS sequence_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ate_rmse REAL,
                    ate_rmse_associated REAL,
                    rpe_trans_rmse REAL,
                    rpe_rot_rmse REAL,
                    coverage REAL,
                    created_utc TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_seqmetrics_exp ON sequence_metrics(experiment_id);

                CREATE TABLE IF NOT EXISTS aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    created_utc TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_aggregates_exp ON aggregates(experiment_id);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_exp ON artifacts(experiment_id);
                """
            )
            # Lightweight schema migrations for existing DBs.
            cols = {
                row[1]: row for row in conn.execute("PRAGMA table_info(sequence_metrics)").fetchall()
            }
            if "ate_rmse_associated" not in cols:
                conn.execute("ALTER TABLE sequence_metrics ADD COLUMN ate_rmse_associated REAL")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            fv = float(v)
        except Exception:
            return None
        if not math.isfinite(fv):
            return None
        return fv

    def upsert_experiment(
        self,
        *,
        experiment_id: str,
        run_name: str,
        prompt_tag: str,
        parent_id: Optional[str],
        git_commit: str,
        config_hash: str,
        notes: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, run_name, prompt_tag, parent_id,
                    git_commit, config_hash, created_utc, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    run_name=excluded.run_name,
                    prompt_tag=excluded.prompt_tag,
                    parent_id=excluded.parent_id,
                    git_commit=excluded.git_commit,
                    config_hash=excluded.config_hash,
                    notes=excluded.notes
                """,
                (
                    str(experiment_id),
                    str(run_name),
                    str(prompt_tag),
                    None if parent_id is None else str(parent_id),
                    str(git_commit),
                    str(config_hash),
                    self._now(),
                    None if notes is None else str(notes),
                ),
            )

    def ensure_experiment(
        self,
        *,
        experiment_id: str,
        run_name: str = "unknown",
        prompt_tag: str = "unknown",
        parent_id: Optional[str] = None,
        git_commit: str = "unknown",
        config_hash: str = "unknown",
        notes: Optional[str] = None,
    ) -> None:
        """Insert experiment row if missing, but never overwrite existing metadata."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO experiments (
                    experiment_id, run_name, prompt_tag, parent_id,
                    git_commit, config_hash, created_utc, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(experiment_id),
                    str(run_name),
                    str(prompt_tag),
                    None if parent_id is None else str(parent_id),
                    str(git_commit),
                    str(config_hash),
                    self._now(),
                    None if notes is None else str(notes),
                ),
            )

    def log_checkpoint(
        self,
        *,
        experiment_id: str,
        stage: str,
        epoch: Optional[int],
        path: str | Path,
        selected_flag: bool,
        selection_score: Optional[float],
    ) -> None:
        cp = Path(path).expanduser()
        if not cp.is_absolute():
            cp = (Path.cwd() / cp).resolve()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (
                    experiment_id, stage, epoch, path,
                    selected_flag, selection_score, created_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(experiment_id),
                    str(stage),
                    None if epoch is None else int(epoch),
                    str(cp),
                    1 if bool(selected_flag) else 0,
                    self._to_float(selection_score),
                    self._now(),
                ),
            )

    def log_sequence_metrics(
        self,
        *,
        experiment_id: str,
        stage: str,
        rows: Iterable[SequenceMetric | Dict[str, Any]],
    ) -> None:
        payload = []
        for r in rows:
            if isinstance(r, SequenceMetric):
                seq = r.sequence
                st = r.status
                ate = r.ate_rmse
                ate_assoc = r.ate_rmse_associated
                rpe_t = r.rpe_trans_rmse
                rpe_r = r.rpe_rot_rmse
                cov = r.coverage
            else:
                seq = str(r.get("sequence", ""))
                st = str(r.get("status", ""))
                ate = r.get("ate_rmse")
                ate_assoc = r.get("ate_rmse_associated")
                rpe_t = r.get("rpe_trans_rmse")
                rpe_r = r.get("rpe_rot_rmse")
                cov = r.get("coverage")
            if not seq:
                continue
            payload.append(
                (
                    str(experiment_id),
                    str(stage),
                    seq,
                    st,
                    self._to_float(ate),
                    self._to_float(ate_assoc),
                    self._to_float(rpe_t),
                    self._to_float(rpe_r),
                    self._to_float(cov),
                    self._now(),
                )
            )
        if not payload:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO sequence_metrics (
                    experiment_id, stage, sequence, status,
                    ate_rmse, ate_rmse_associated,
                    rpe_trans_rmse, rpe_rot_rmse, coverage, created_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def log_aggregate(
        self,
        *,
        experiment_id: str,
        metric_name: str,
        metric_value: Optional[float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO aggregates (
                    experiment_id, metric_name, metric_value, created_utc
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(experiment_id),
                    str(metric_name),
                    self._to_float(metric_value),
                    self._now(),
                ),
            )

    def log_artifact(self, *, experiment_id: str, artifact_type: str, path: str | Path) -> None:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (experiment_id, type, path, created_utc)
                VALUES (?, ?, ?, ?)
                """,
                (str(experiment_id), str(artifact_type), str(p), self._now()),
            )
