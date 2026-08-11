from __future__ import annotations

import copy
import os
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch


DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from app.backend import OPENAI_TOOLS, dispatch_tool
from app.backend_factory import create_backend
from app.dashboard import _decision_snapshot_token, _has_map_coordinate
from app.demo_backend import DemoBackend
from app.guide_service import create_notification_draft, plan_work


class DScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = DemoBackend(today=date(2026, 8, 12), storage_days=12)
        self.farm_id = self.backend.FARM_ID

    def test_calendar_preserves_forecast_resolution(self) -> None:
        result = self.backend.get_risk_calendar(self.farm_id, 7, None)
        self.assertEqual(result["status"], "ok")
        block_items = [
            item for item in result["data"]["items"] if item["resolution"] == "3h"
        ]
        daily_items = [
            item for item in result["data"]["items"] if item["resolution"] == "day"
        ]
        self.assertEqual(len(block_items), 24)
        self.assertEqual(len(daily_items), 4)
        self.assertTrue(all(item["block"] is None for item in daily_items))
        self.assertTrue(all(item["start"] is None for item in daily_items))

    def test_backend_factory_and_status_do_not_require_dashboard_edits(self) -> None:
        with patch.dict(
            os.environ,
            {"D_BACKEND_FACTORY": "app.demo_backend:DemoBackend"},
        ):
            backend = create_backend(storage_days=9)
        status = backend.get_system_status()
        self.assertEqual(status["data"]["default_farm_id"], backend.FARM_ID)
        self.assertEqual(backend.get_storage_days(backend.FARM_ID)["data"]["days"], 9)

    def test_snapshot_token_changes_with_model_or_forecast_update(self) -> None:
        status = self.backend.get_system_status()
        calendar = self.backend.get_risk_calendar(self.farm_id, 7, None)
        original = _decision_snapshot_token(status, calendar)
        updated = copy.deepcopy(calendar)
        updated["data"]["items"][0]["model_version"] = "new-model"
        self.assertNotEqual(original, _decision_snapshot_token(status, updated))
        updated = copy.deepcopy(calendar)
        updated["data"]["items"][0]["forecast_issued_at"] = (
            "2026-08-12T07:00:00+09:00"
        )
        self.assertNotEqual(original, _decision_snapshot_token(status, updated))

    def test_map_coordinate_filter_keeps_missing_rows_out_of_map_only(self) -> None:
        self.assertTrue(_has_map_coordinate({"latitude": 35.9, "longitude": 127.0}))
        self.assertFalse(_has_map_coordinate({"latitude": None, "longitude": 127.0}))
        self.assertFalse(_has_map_coordinate({"latitude": 95.0, "longitude": 127.0}))

    def test_sensor_snapshot_matches_canonical_contract(self) -> None:
        requested_at = "2026-08-12T19:30:00+09:00"
        result = self.backend.get_sensor_snapshot(at=requested_at)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["requested_at"], requested_at)
        self.assertEqual(result["source"]["data_as_of"], "2026-08-11")
        stations = result["data"]["stations"]
        self.assertGreaterEqual(len(stations), 4)
        self.assertEqual(
            len({item["station_id"] for item in stations}), len(stations)
        )
        required = {
            "observation_id",
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "observed_at",
            "ingested_at",
            "h2s_ppm",
            "nh3_ppm",
            "tvoc_ppm",
            "complex_odor_value",
            "complex_odor_unit",
            "temperature_c",
            "humidity_pct",
            "wind_direction_text",
            "wind_speed_ms",
            "record_qc",
            "quality_flags",
            "data_version",
        }
        for station in stations:
            self.assertTrue(required.issubset(station))
            self.assertTrue(-90 <= station["latitude"] <= 90)
            self.assertTrue(-180 <= station["longitude"] <= 180)
            observed = datetime.fromisoformat(station["observed_at"])
            ingested = datetime.fromisoformat(station["ingested_at"])
            self.assertIsNotNone(observed.utcoffset())
            self.assertGreaterEqual(ingested, observed)
            self.assertIsNone(station["complex_odor_unit"])
        missing = next(item for item in stations if item["record_qc"] == "missing")
        self.assertIsNone(missing["h2s_ppm"])
        self.assertIsNone(missing["complex_odor_value"])

    def test_plan_is_deterministic_and_returns_top_three(self) -> None:
        first = plan_work(self.backend, self.farm_id, "분뇨제거").to_dict()
        second = plan_work(self.backend, self.farm_id, "분뇨제거").to_dict()
        self.assertEqual(first, second)
        self.assertEqual(len(first["recommended"]), 3)
        self.assertEqual(len(first["avoid"]), 3)
        self.assertLessEqual(
            first["recommended"][0]["recommendation_score"],
            first["recommended"][1]["recommendation_score"],
        )
        self.assertGreaterEqual(
            first["avoid"][0]["recommendation_score"],
            first["avoid"][1]["recommendation_score"],
        )

    def test_notification_is_draft_only(self) -> None:
        guide = plan_work(self.backend, self.farm_id, "청소").to_dict()
        draft = create_notification_draft(
            self.backend, self.farm_id, "청소", guide["recommended"][0]
        ).to_dict()
        self.assertTrue(draft["audience_is_mock"])
        self.assertFalse(draft["approved"])
        self.assertFalse(draft["sent"])
        self.assertEqual(draft["plume_status"], "unverified")

    def test_openai_tools_use_strict_schemas_and_allowlist(self) -> None:
        for tool in OPENAI_TOOLS:
            self.assertTrue(tool["strict"])
            parameters = tool["parameters"]
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(
                set(parameters["properties"]), set(parameters["required"])
            )
        denied = dispatch_tool(self.backend, "delete_everything", {})
        self.assertEqual(denied["status"], "unavailable")

    def test_all_tool_schemas_match_backend_argument_names(self) -> None:
        arguments = {
            "get_risk_calendar": {
                "farm_id": self.farm_id,
                "days": 3,
                "work_type": "분뇨제거",
            },
            "get_forecast": {"farm_id": self.farm_id, "days": 3},
            "get_storage_days": {"farm_id": self.farm_id},
            "search_rag": {
                "question": "분뇨제거 관리 기준",
                "query_type": "분뇨제거",
            },
            "get_farm_config": {"farm_id": self.farm_id},
            "get_plume_assessment": {
                "farm_id": self.farm_id,
                "when": "2026-08-13T09:00:00+09:00",
            },
        }
        for name, values in arguments.items():
            with self.subTest(tool=name):
                result = dispatch_tool(self.backend, name, values)
                self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
