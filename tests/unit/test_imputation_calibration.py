"""An unassessed item must be imputed from context, never defaulted to zero."""

from __future__ import annotations

import unittest

from psyvec.evaluation.calibration import TotalScoreCalibrator, fit_calibrator
from psyvec.evaluation.imputation import (
    ImputationModel,
    fit_from_references,
    fit_from_run,
)

ITEMS = ("a", "b", "c")
# Three participants spanning the severity range, with items that move together
# the way PHQ-8 items actually do.
REFERENCES = [
    {"a": 0, "b": 0, "c": 0},
    {"a": 1, "b": 1, "c": 2},
    {"a": 3, "b": 3, "c": 3},
    {"a": 2, "b": 2, "c": 2},
    {"a": 0, "b": 1, "c": 0},
]


class FitFromReferencesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = fit_from_references(REFERENCES, ITEMS)

    def test_fits_every_item(self) -> None:
        self.assertEqual(set(self.model.fits), set(ITEMS))

    def test_a_severe_participant_is_not_imputed_at_zero(self) -> None:
        # The v2 rule scored this 0; the whole point is that it must not.
        estimate = self.model.impute({"a": 3, "b": 3}, ["c"])["c"]
        self.assertGreater(estimate, 1.5)

    def test_a_minimal_participant_is_imputed_low(self) -> None:
        self.assertLess(self.model.impute({"a": 0, "b": 0}, ["c"])["c"], 1.0)

    def test_with_nothing_assessed_it_falls_back_to_the_base_rate(self) -> None:
        estimate = self.model.impute({}, ["c"])["c"]
        self.assertAlmostEqual(estimate, self.model.fits["c"].base_rate, places=6)

    def test_estimates_stay_inside_the_rubric_range(self) -> None:
        extreme = self.model.impute({"a": 3, "b": 3}, ["c"])["c"]
        self.assertGreaterEqual(extreme, 0.0)
        self.assertLessEqual(extreme, 3.0)

    def test_an_unknown_item_still_returns_a_usable_estimate(self) -> None:
        self.assertIn("zzz", self.model.impute({"a": 1}, ["zzz"]))

    def test_round_trips_through_json(self) -> None:
        restored = ImputationModel.from_json(self.model.to_json())
        self.assertAlmostEqual(
            restored.impute({"a": 2}, ["c"])["c"],
            self.model.impute({"a": 2}, ["c"])["c"],
        )

    def test_empty_training_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_from_references([], ITEMS)


class FitFromRunTest(unittest.TestCase):
    def test_conditions_on_what_the_pipeline_actually_assessed(self) -> None:
        assessed = [{"a": 0}, {"a": 1}, {"a": 3}, {"a": 2}, {"a": 0}]
        model = fit_from_run(assessed, REFERENCES, ITEMS)
        self.assertGreater(
            model.impute({"a": 3}, ["c"])["c"], model.impute({"a": 0}, ["c"])["c"]
        )

    def test_misaligned_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_from_run([{"a": 1}], REFERENCES, ITEMS)

    def test_an_item_with_too_few_abstentions_falls_back_to_its_base_rate(self) -> None:
        # 'c' is assessed for everyone, so there is nothing to fit it on.
        assessed = [dict(reference) for reference in REFERENCES]
        model = fit_from_run(assessed, REFERENCES, ITEMS)
        self.assertEqual(model.fits["c"].slope, 0.0)


class CalibrationTest(unittest.TestCase):
    PREDICTED = [2.0, 4.0, 6.0, 8.0, 10.0]
    REFERENCE = [0.0, 4.0, 8.0, 12.0, 16.0]

    def test_mae_objective_recovers_an_exact_affine_relationship(self) -> None:
        calibrator = fit_calibrator(self.PREDICTED, self.REFERENCE, objective="mae")
        self.assertAlmostEqual(calibrator.apply(6.0), 8.0, places=1)

    def test_spread_objective_matches_the_reference_spread(self) -> None:
        calibrator = fit_calibrator(self.PREDICTED, self.REFERENCE, objective="spread")
        calibrated = [calibrator.apply(value) for value in self.PREDICTED]
        mean = sum(calibrated) / len(calibrated)
        spread = (sum((v - mean) ** 2 for v in calibrated) / len(calibrated)) ** 0.5
        self.assertAlmostEqual(spread, 5.6568, places=2)

    def test_identity_objective_is_a_no_op_for_ablations(self) -> None:
        calibrator = fit_calibrator(
            self.PREDICTED, self.REFERENCE, objective="identity"
        )
        self.assertEqual(calibrator.apply(7.0), 7.0)

    def test_output_is_clipped_to_the_phq8_range(self) -> None:
        calibrator = TotalScoreCalibrator(intercept=0.0, slope=10.0, objective="test")
        self.assertEqual(calibrator.apply(100.0), 24.0)
        self.assertEqual(calibrator.apply(-100.0), 0.0)

    def test_round_trips_through_json(self) -> None:
        calibrator = fit_calibrator(self.PREDICTED, self.REFERENCE, objective="mae")
        restored = TotalScoreCalibrator.from_json(calibrator.to_json())
        self.assertAlmostEqual(restored.apply(5.0), calibrator.apply(5.0))

    def test_misaligned_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_calibrator([1.0], [1.0, 2.0])

    def test_unknown_objective_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_calibrator(
                self.PREDICTED,
                self.REFERENCE,
                objective="nonsense",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
