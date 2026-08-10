from __future__ import annotations

import unittest

from vifinqa.utils.viet_text import (
    FUZZY_SCORER_BACKEND,
    FUZZY_SCORER_VERSION,
    fuzz_partial,
    fuzz_token_set,
    fuzzy_scorer_provenance,
    label_metric_score,
)


class FuzzyScorerReproducibilityTests(unittest.TestCase):
    def test_provenance_is_explicit_and_versioned(self):
        self.assertEqual(FUZZY_SCORER_BACKEND, "difflib.SequenceMatcher")
        self.assertEqual(FUZZY_SCORER_VERSION, "1")
        self.assertEqual(fuzzy_scorer_provenance(), {
            "backend": "difflib.SequenceMatcher",
            "version": "1",
        })

    def test_scores_match_kaggle_p2_1_difflib_contract(self):
        self.assertAlmostEqual(
            fuzz_token_set(
                "loi nhuan sau thue",
                "loi nhuan sau thue chua phan phoi",
            ),
            70.58823529411765,
        )
        self.assertAlmostEqual(
            label_metric_score(
                "loi nhuan sau thue",
                "loi nhuan sau thue chua phan phoi",
            ),
            62.52100840336135,
        )
        self.assertAlmostEqual(
            fuzz_partial("do la my usd", "ngoai te usd"),
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
