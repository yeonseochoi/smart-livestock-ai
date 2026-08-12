from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from agents.fixture_provider import FixtureProvider
from agents.notify_draft import approve_for_demo, create_draft
from agents.provider import create_provider
from agents.tools_schema import AgentTools, OPENAI_TOOLS, dispatch_tool
from agents.work_guide import plan_work


class PlumeCountingProvider(FixtureProvider):
    def __init__(self) -> None:
        super().__init__(storage_days=12)
        self.plume_calls = 0

    def get_plume_assessment(self, farm_id: str, when: str):
        self.plume_calls += 1
        return super().get_plume_assessment(farm_id, when)


class DAgentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FixtureProvider(storage_days=12)
        self.farm_id = self.provider.FARM_ID

    def test_fixture_sensor_contract_separates_times_and_unknown_unit(self) -> None:
        result = self.provider.get_sensor_snapshot()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"]["state"], "fixture")
        first = result["data"]["stations"][0]
        self.assertNotEqual(first["observed_at"], first["ingested_at"])
        self.assertIsNone(first["complex_odor_unit"])

    def test_calendar_preserves_forecast_resolution(self) -> None:
        items = self.provider.get_risk_calendar(self.farm_id, 7)["data"]["items"]
        self.assertTrue(all(item["resolution"] == "3h" for item in items if item["horizon"] == "D+1~3"))
        self.assertTrue(all(item["resolution"] == "day" and item["block"] is None
                            for item in items if item["horizon"] == "D+4~7"))

    def test_plan_returns_top_three_and_does_not_use_plume_for_ranking(self) -> None:
        provider = PlumeCountingProvider()
        guide = plan_work(provider, provider.FARM_ID, "분뇨제거")
        self.assertEqual(len(guide.recommended), 3)
        self.assertEqual(len(guide.avoid), 3)
        scores = [item.recommendation_score for item in guide.recommended]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(provider.plume_calls, 0)

    def test_notification_is_draft_only_and_human_approval_does_not_send(self) -> None:
        guide = plan_work(self.provider, self.farm_id, "분뇨제거")
        draft = create_draft(
            self.provider, self.farm_id, "분뇨제거",
            guide.recommended[0].to_dict(),
        )
        self.assertFalse(draft.approved)
        self.assertFalse(draft.sent)
        approved = approve_for_demo(draft)
        self.assertTrue(approved["approved"])
        self.assertFalse(approved["sent"])
        self.assertEqual(approved["approval_scope"], "streamlit_session_demo_only")

    def test_openai_tool_schemas_match_agent_tools_and_are_strict(self) -> None:
        for schema in OPENAI_TOOLS:
            self.assertTrue(hasattr(AgentTools, schema["name"]))
            self.assertTrue(schema["strict"])
            self.assertFalse(schema["parameters"]["additionalProperties"])
            self.assertEqual(
                set(schema["parameters"]["required"]),
                set(schema["parameters"]["properties"]),
            )

    def test_dispatch_rejects_unknown_tool(self) -> None:
        result = dispatch_tool(AgentTools(self.provider), "__dict__", {})
        self.assertEqual(result["status"], "unavailable")

    def test_bad_external_factory_is_not_silently_replaced_by_fixture(self) -> None:
        with patch.dict(os.environ, {"D_PROVIDER_FACTORY": "bad-format"}, clear=False):
            with self.assertRaises(ValueError):
                create_provider(storage_days=12)

    def test_dashboard_is_thin_and_imports_agents_not_team_data_layers(self) -> None:
        source = (DEMO_DIR / "app" / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("from agents.work_guide import", source)
        self.assertNotIn("from ops", source)
        self.assertNotIn("from rag", source)
        self.assertNotIn("from scoring", source)

        guide_source = (DEMO_DIR / "agents" / "work_guide.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from agents.scoring_policy import", guide_source)
        self.assertNotIn("from scoring", guide_source)


if __name__ == "__main__":
    unittest.main()
