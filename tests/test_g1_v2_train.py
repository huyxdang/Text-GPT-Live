from __future__ import annotations

import unittest

from scripts.g1_v2_train import _promotion_gate


class G1V2PromotionTests(unittest.TestCase):
    def test_promotion_requires_candidate_and_retained_demo_gates(self) -> None:
        candidate = {"gates": {"passed": True}}
        retained = {
            "demo-1": {"kind_accuracy": 0.99, "strict_accuracy": 0.84},
            "demo-2": {"kind_accuracy": 0.99, "strict_accuracy": 0.89},
            "demo-4": {"kind_accuracy": 1.0, "strict_accuracy": 0.74},
        }
        self.assertTrue(_promotion_gate(candidate, retained)["passed"])

        retained["demo-2"]["kind_accuracy"] = 0.97
        failed = _promotion_gate(candidate, retained)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["retained_demo_results"]["demo-2"]["kind_accuracy"])

    def test_promotion_fails_when_candidate_gates_fail(self) -> None:
        candidate = {"gates": {"passed": False}}
        retained = {
            "demo-1": {"kind_accuracy": 1.0, "strict_accuracy": 1.0},
            "demo-2": {"kind_accuracy": 1.0, "strict_accuracy": 1.0},
            "demo-4": {"kind_accuracy": 1.0, "strict_accuracy": 1.0},
        }
        result = _promotion_gate(candidate, retained)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["candidate_gates"])


if __name__ == "__main__":
    unittest.main()
