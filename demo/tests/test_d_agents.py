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
from agents.gemini_explainer import SYSTEM_PROMPT
from agents.provider import create_provider
from agents.rag_adapter import RagYujinAdapter
from agents.tools_schema import AgentTools, TOOL_SPECS, dispatch_tool
from agents.work_guide import _build_windows, plan_work, run


class _Document:
    def __init__(self, text: str, metadata: dict) -> None:
        self.page_content = text
        self.metadata = metadata


class _Retriever:
    def __init__(self, documents: list[_Document]) -> None:
        self.documents = documents

    def invoke(self, query: str) -> list[_Document]:
        return self.documents


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

    def test_calendar_preserves_hourly_and_daily_resolution(self) -> None:
        items = self.provider.get_risk_calendar(self.farm_id, 7)["data"]["items"]
        self.assertTrue(all(
            item["resolution"] == "1h" and item["hour"] is not None
            for item in items if item["horizon"] == "D+1~3"
        ))
        self.assertTrue(all(
            item["resolution"] == "day" and item["hour"] is None
            for item in items if item["horizon"] == "D+4~7"
        ))

    def test_six_hour_window_uses_each_hour_weight_directly(self) -> None:
        items = []
        date_text = self.provider.today.isoformat()
        for hour, score in enumerate((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
            items.append({
                "resolution": "1h",
                "start": f"{date_text}T{hour:02d}:00:00+09:00",
                "risk_score": score,
                "risk_grade": "낮음",
            })
        windows = _build_windows(items, "청소", None)
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0].window_risk, 0.30)

    def test_plan_returns_top_three(self) -> None:
        guide = plan_work(self.provider, self.farm_id, "분뇨제거")
        self.assertEqual(len(guide.recommended), 3)
        self.assertEqual(len(guide.avoid), 3)
        scores = [item.recommendation_score for item in guide.recommended]
        self.assertEqual(scores, sorted(scores))

    def test_tool_specs_are_provider_neutral_and_limited_to_four(self) -> None:
        names = {schema["name"] for schema in TOOL_SPECS}
        self.assertEqual(names, {
            "get_risk_calendar", "get_storage_days", "search_rag", "get_farm_config"
        })
        for schema in TOOL_SPECS:
            self.assertTrue(hasattr(AgentTools, schema["name"]))
            self.assertFalse(schema["parameters"]["additionalProperties"])

    def test_dispatch_rejects_unknown_tool(self) -> None:
        result = dispatch_tool(AgentTools(self.provider), "__dict__", {})
        self.assertEqual(result["status"], "unavailable")

    def test_bad_external_factory_is_not_silently_replaced_by_fixture(self) -> None:
        with patch.dict(os.environ, {"D_PROVIDER_FACTORY": "bad-format"}, clear=False):
            with self.assertRaises(ValueError):
                create_provider(storage_days=12)

    def test_gemini_prompt_freezes_decision_fields(self) -> None:
        for word in ("시간", "점수", "등급", "변경", "source_file"):
            self.assertIn(word, SYSTEM_PROMPT)

    def test_gemini_explanation_is_appended_after_canonical_plan(self) -> None:
        env = {"D_PROVIDER_MODE": "fixture", "GOOGLE_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=False):
            with patch("agents.gemini_explainer.compose", return_value="근거 기반 설명"):
                result = run(FixtureProvider.FARM_ID, "청소")
        self.assertTrue(result.startswith("[추천 창]"))
        self.assertIn("\n\n[AI 설명]\n근거 기반 설명", result)

    def test_rag_adapter_returns_structured_source_metadata(self) -> None:
        adapter = RagYujinAdapter.__new__(RagYujinAdapter)
        adapter.manual_retriever = _Retriever([
            _Document("실무 근거", {
                "source_file": "manual.pdf", "unit": "제1장", "page": 3,
                "doc_type": "manual",
            })
        ])
        adapter.law_retriever = _Retriever([
            _Document("법령 근거", {
                "source_file": "law.pdf", "unit": "제2조", "page": 1,
                "doc_type": "law",
            })
        ])
        result = adapter.search("청소 기준")
        self.assertEqual(result["backend"], "rag_yujin-chroma")
        self.assertEqual(result["results"][0]["source_file"], "manual.pdf")
        self.assertEqual(result["results"][1]["unit"], "제2조")

    def test_dashboard_uses_gemini_and_excludes_notification_feature(self) -> None:
        source = (DEMO_DIR / "app" / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("from agents.gemini_explainer import", source)
        self.assertNotIn("agents.notify_draft", source)
        self.assertNotIn("openai_explainer", source)


if __name__ == "__main__":
    unittest.main()
