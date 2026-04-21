import unittest
import os

from pdf_web.main import analyze_scope_data, analyze_scope_data_with_gpt


class ScopeAnalysisTests(unittest.TestCase):
    def test_detects_scope_mentions_and_metrics(self):
        text = """
        Scope 1 emissions 2024: 1200 tCO2e
        Scope 2 market-based emissions 2024: 900 tCO2e
        Scope 3 business travel emissions 2024: 300 tCO2e
        We target a 30% reduction by 2030 from a 2020 baseline.
        """
        result = analyze_scope_data(text)

        self.assertTrue(result["scope_presence"]["scope_1"]["found"])
        self.assertTrue(result["scope_presence"]["scope_2"]["found"])
        self.assertTrue(result["scope_presence"]["scope_3"]["found"])
        self.assertIn("2024", result["reporting_years"])
        self.assertIn("2030", result["reporting_years"])
        self.assertGreaterEqual(len(result["target_statements"]), 1)

    def test_handles_empty_input(self):
        result = analyze_scope_data("")

        self.assertFalse(result["scope_presence"]["scope_1"]["found"])
        self.assertFalse(result["scope_presence"]["scope_2"]["found"])
        self.assertFalse(result["scope_presence"]["scope_3"]["found"])
        self.assertEqual(result["reporting_years"], [])
        self.assertEqual(result["target_statements"], [])

    def test_gpt_fallback_exposes_missing_key_reason(self):
        previous_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = analyze_scope_data_with_gpt("Scope 1 emissions 2024: 100 tCO2e")
            troubleshooting = result.get("troubleshooting", {})

            self.assertEqual(result.get("analysis_method"), "heuristic_fallback")
            self.assertFalse(troubleshooting.get("used_gpt"))
            self.assertIn("OPENAI_API_KEY", troubleshooting.get("reason", ""))
        finally:
            if previous_key is not None:
                os.environ["OPENAI_API_KEY"] = previous_key

    def test_gpt_fallback_exposes_empty_text_reason(self):
        previous_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = analyze_scope_data_with_gpt("")
            troubleshooting = result.get("troubleshooting", {})

            self.assertEqual(result.get("analysis_method"), "heuristic_fallback")
            self.assertFalse(troubleshooting.get("input_has_text"))
            self.assertIn("No extracted text", troubleshooting.get("reason", ""))
        finally:
            if previous_key is not None:
                os.environ["OPENAI_API_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
