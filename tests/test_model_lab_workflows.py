import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import pandas as pd

from app import server
from core.database import connect_existing_database
from processing import process_seasons_data as processing_workflow
from processing.core import database_connection
from storage import import_fixtures
from training import prepare_dataset


class ModelLabWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.jobs = {}
        self.jobs_patch = patch.object(server, "MODEL_JOBS", self.jobs)
        self.jobs_patch.start()
        self.addCleanup(self.jobs_patch.stop)

    @staticmethod
    def handler(payload=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        handler = SimpleNamespace(
            headers={"Content-Length": str(len(body))},
            rfile=io.BytesIO(body),
            responses=[],
        )
        handler._send_json = lambda data, status=200: handler.responses.append((status, data))
        handler._query_int = server.DashboardHandler._query_int.__get__(handler)
        return handler

    def test_batch_request_orders_actions_and_snapshots_options(self):
        handler = self.handler({
            "action": "batch", "actions": ["train-neural", "fetch", "prepare"],
            "league_id": 1,
        })

        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        with patch.object(server, "_read_league", return_value={"seasons": [2024]}), \
             patch.object(server.threading, "Thread", ImmediateThread), \
             patch.object(server, "_run_model_workflow_batch") as run:
            server.DashboardHandler._start_model_lab_job(handler)
        self.assertEqual(handler.responses[0][0], 202)
        job = self.jobs[handler.responses[0][1]["job_id"]]
        self.assertEqual(job["actions"], ["fetch", "prepare", "train-neural"])
        self.assertEqual(run.call_args.args[1:], (job["actions"], 1, 2024, True, True))
        self.assertFalse(server.MODEL_LOCK.locked())

    def test_invalid_batch_selections_are_rejected(self):
        for actions in ([], ["fetch", "fetch"], ["unknown"], "fetch", [None]):
            with self.subTest(actions=actions):
                handler = self.handler({"action": "batch", "actions": actions, "league_id": 1, "season": 2024})
                server.DashboardHandler._start_model_lab_job(handler)
                self.assertEqual(handler.responses[0][0], 400)
        self.assertEqual(self.jobs, {})

    def test_batch_stops_at_failure_and_records_successful_results(self):
        actions = ["fetch", "process", "prepare"]
        self.jobs["job"] = {
            "status": "running", "workflow_states": dict.fromkeys(actions, "queued"),
            "workflow_results": {},
        }
        calls = []

        def run(job_id, action, league_id, season, **kwargs):
            calls.append(action)
            if action == "process":
                server._update_model_job(job_id, message="Failed to process")
                return False
            server._update_model_job(job_id, result={"message": "Fetched"})
            return True

        with patch.object(server, "_run_model_job", side_effect=run):
            server._run_model_workflow_batch("job", actions, 1, 2024, False, False)
        job = self.jobs["job"]
        self.assertEqual(calls, ["fetch", "process"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["current_step"], 2)
        self.assertEqual(job["workflow_states"], {
            "fetch": "completed", "process": "failed", "prepare": "queued",
        })
        self.assertEqual(job["workflow_results"], {"fetch": {"message": "Fetched"}})

    def test_batch_success_marks_every_step_complete(self):
        actions = ["fetch", "prepare", "train-neural"]
        self.jobs["job"] = {
            "status": "running", "workflow_states": dict.fromkeys(actions, "queued"),
            "workflow_results": {},
        }

        def run(job_id, action, league_id, season, **kwargs):
            server._update_model_job(job_id, result={"message": action})
            return True

        with patch.object(server, "_run_model_job", side_effect=run):
            server._run_model_workflow_batch("job", actions, 1, 2024, False, False)
        job = self.jobs["job"]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["workflow_states"], dict.fromkeys(actions, "completed"))
        self.assertEqual(list(job["workflow_results"]), actions)
        self.assertEqual(job["current_step"], 3)

    def test_run_all_cards_match_backend_workflow_order(self):
        class WorkflowParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.actions = []

            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if tag == "input" and "model-select" in attributes.get("class", "").split():
                    self.actions.append(attributes["data-action"])

        parser = WorkflowParser()
        template = Path(server.__file__).parent / "templates" / "index.html"
        parser.feed(template.read_text(encoding="utf-8"))
        self.assertEqual(parser.actions, list(server.MODEL_WORKFLOW_ACTIONS))
        self.assertEqual(len(parser.actions), 6)

    def test_workflow_history_is_scoped_and_queued_does_not_hide_completion(self):
        self.jobs.update({
            "old": {"league_id": 1, "season": 2024, "action": "fetch", "status": "completed"},
            "other": {"league_id": 2, "season": 2025, "action": "prepare", "status": "completed"},
            "batch": {
                "league_id": 1, "season": 2024, "action": "batch", "status": "running",
                "workflow_states": {"fetch": "queued", "process": "running"},
            },
        })
        handler = self.handler()
        server.DashboardHandler._serve_model_workflows(handler, {"league_id": ["1"], "season": ["2024"]})
        self.assertEqual(handler.responses[0], (200, {
            "states": {"fetch": "completed", "process": "running"}, "active_job_id": "batch",
        }))

    def test_progress_reports_counts_and_zero_total(self):
        self.jobs["job"] = {}
        server._report_model_job_progress("job", 1, 4, "Working")
        self.assertEqual(self.jobs["job"], {"completed": 1, "total": 4, "progress": 25, "message": "Working"})
        server._report_model_job_progress("job", 0, 0, "Starting")
        self.assertEqual(self.jobs["job"]["progress"], 0)
        self.assertEqual(self.jobs["job"]["total"], 0)

    def test_neural_workflow_trains_and_reports_outcome_networks(self):
        from training.deep_learning import train_match_networks, train_sequence_models, train_outcome_networks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.ddb"
            with duckdb.connect(str(dataset)):
                pass
            self.jobs["job"] = {}
            with patch.object(server, "get_database_path", return_value=root / "leagues.ddb"), \
                 patch.object(server, "_read_league", return_value={"seasons": [2024], "country_name": "England"}), \
                 patch.object(server, "get_country_database_path", return_value=root / "country.ddb"), \
                 patch.object(server, "get_dataset_database_path", return_value=dataset), \
                 patch.object(server, "_has_all_season_dataset", return_value=True), \
                 patch.object(train_match_networks, "run_experiment", return_value=(pd.DataFrame([{"outcome_accuracy": 0.5}]), None)), \
                 patch.object(train_sequence_models, "run_experiment", return_value=(pd.DataFrame([{"model": "cnn"}, {"model": "rnn"}]), None)), \
                 patch.object(train_outcome_networks, "run_experiment", return_value=(pd.DataFrame([{"model": "lstm"}, {"model": "autoencoder"}]), None)) as train:
                self.assertTrue(server._run_model_job("job", "train-neural", 39, 2024))
            train.assert_called_once()
            self.assertEqual(
                [row["model"] for row in self.jobs["job"]["result"]["metrics"]],
                ["MLP", "CNN", "RNN", "LSTM", "AUTOENCODER"],
            )
            self.assertEqual(self.jobs["job"]["progress"], 100)

    def test_existing_database_connection_does_not_create_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.ddb"
            with self.assertRaises(FileNotFoundError):
                connect_existing_database(path)
            self.assertFalse(path.exists())

    def test_fetch_and_dataset_readers_share_writer_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leagues.ddb"
            with duckdb.connect(str(path)) as writer:
                writer.execute(
                    "CREATE TABLE leagues (league_id INTEGER, country_name VARCHAR, seasons INTEGER[])"
                )
                writer.execute("INSERT INTO leagues VALUES (39, 'England', [2024])")
                with patch.object(import_fixtures, "LEAGUES_DATABASE", path):
                    self.assertEqual(import_fixtures.get_country("39-2024-fixtures"), "England")
                self.assertEqual(prepare_dataset.get_league_details(39, path), ([2024], "England"))
                with database_connection(path, read_only=True) as reader:
                    self.assertEqual(reader.execute("SELECT COUNT(*) FROM leagues").fetchone()[0], 1)

    def test_status_and_processing_can_overlap_in_both_connection_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            league_db = root / "leagues.ddb"
            country_db = root / "leagues" / "England.ddb"
            dataset_db = root / "datasets" / "England.ddb"
            country_db.parent.mkdir()
            dataset_db.parent.mkdir()
            with duckdb.connect(str(league_db)) as connection:
                connection.execute(
                    "CREATE TABLE leagues (league_id INTEGER, league_name VARCHAR, "
                    "country_name VARCHAR, seasons INTEGER[])"
                )
                connection.execute("INSERT INTO leagues VALUES (39, 'League', 'England', [2024])")
            with duckdb.connect(str(country_db)) as connection:
                connection.execute(
                    'CREATE TABLE "39-2024-fixtures" AS SELECT 1 AS fixture_id, '
                    "TIMESTAMP '2024-01-01' AS date, 'A' AS home_team, 'B' AS away_team, "
                    "'Round 1' AS round, 'FT' AS status, 1 AS home_goals, 0 AS away_goals"
                )
            with duckdb.connect(str(dataset_db)):
                pass

            status_open = threading.Event()
            allow_status = threading.Event()
            processing_open = threading.Event()
            allow_processing = threading.Event()
            original_fixture_table = server._fixture_table

            def wait_in_status(connection, table_name):
                if not status_open.is_set():
                    status_open.set()
                    if not allow_status.wait(15):
                        raise TimeoutError("Status reader was not released")
                return original_fixture_table(connection, table_name)

            def wait_in_processing(**kwargs):
                processing_open.set()
                if not allow_processing.wait(15):
                    raise TimeoutError("Processing writer was not released")

            with ExitStack() as stack:
                stack.enter_context(patch.object(server, "_fixture_table", side_effect=wait_in_status))
                stack.enter_context(patch.object(processing_workflow, "process_goals_data", side_effect=wait_in_processing))
                for name in (
                    "process_points_data", "process_xg_data", "process_rho_data", "process_fair_odds_data",
                    "process_outcome_probability_data", "process_recent_goals_data",
                    "process_recent_points_data", "process_recent_xg_data",
                    "process_recent_fair_odds_data", "process_head_to_head_data",
                ):
                    stack.enter_context(patch.object(processing_workflow, name))
                league_writer = stack.enter_context(duckdb.connect(str(league_db)))
                league_writer.execute("SELECT 1")
                dataset_writer = stack.enter_context(duckdb.connect(str(dataset_db)))
                dataset_writer.execute("SELECT 1")
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
                try:
                    status = pool.submit(server.get_model_lab_status, 39, 2024, league_db)
                    self.assertTrue(status_open.wait(15), "Status did not open country database")
                    processing = pool.submit(
                        processing_workflow.process_seasons_data,
                        39, [2024], league_db, country_db,
                    )
                    self.assertTrue(processing_open.wait(15), "Processing did not open country database")
                    allow_status.set()
                    result = status.result(timeout=15)
                    self.assertEqual(result["feature_tables"]["available"], 0)
                    allow_processing.set()
                    processing.result(timeout=15)
                finally:
                    allow_status.set()
                    allow_processing.set()


if __name__ == "__main__":
    unittest.main()
