import unittest
import os

from pdf_web.main import analyze_scope_data, analyze_scope_data_with_gpt


class ScopeAnalysisTests(unittest.TestCase):
    def test_detects_reported_scope_emissions(self):
        text = """
        Scope 1 emissions 2024: 1200 tCO2e
        Scope 2 market-based emissions 2024: 900 tCO2e
        Scope 3 business travel emissions 2024: 300 tCO2e
        """

        result = analyze_scope_data(text)

        self.assertTrue(result["scope_1"]["reported_emissions_found"])
        self.assertTrue(result["scope_2"]["reported_emissions_found"])
        self.assertTrue(result["scope_3"]["reported_emissions_found"])
        self.assertIn("2024", result["reporting_years"])

    def test_handles_empty_input(self):
        result = analyze_scope_data("")

        self.assertFalse(result["scope_1"]["reported_emissions_found"])
        self.assertFalse(result["scope_1"]["activity_data_found"])
        self.assertFalse(result["scope_2"]["reported_emissions_found"])
        self.assertFalse(result["scope_2"]["activity_data_found"])
        self.assertFalse(result["scope_3"]["reported_emissions_found"])
        self.assertFalse(result["scope_3"]["activity_data_found"])
        self.assertEqual(result["reporting_years"], [])

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

    def test_activity_data_detection_and_estimation(self):
        text = """
        Electricity consumption in 2024 was 12000 kWh.
        Gas usage in 2024 was 5000 kWh.
        """

        result = analyze_scope_data(text)

        self.assertTrue(result["scope_2"]["activity_data_found"])
        self.assertTrue(result["scope_1"]["activity_data_found"])
        self.assertTrue(result["scope_2"]["estimated_emissions_possible"])
        self.assertTrue(result["scope_1"]["estimated_emissions_possible"])
        self.assertIsNotNone(result["scope_2"]["estimated_emissions_tco2e"])
        self.assertIsNotNone(result["scope_1"]["estimated_emissions_tco2e"])
        self.assertGreater(len(result["scope_2"]["activity_items"]), 0)
        self.assertGreater(len(result["scope_1"]["activity_items"]), 0)

    def test_scope_3_without_factor_marks_not_estimable(self):
        text = "Business travel and logistics data mentioned in 2024 without any CO2e values."
        result = analyze_scope_data(text)

        self.assertFalse(result["scope_3"]["estimated_emissions_possible"])


if __name__ == "__main__":
    unittest.main()