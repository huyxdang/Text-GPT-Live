import unittest

from fastapi.testclient import TestClient

from app.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_and_index(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["tick_ms"], 650)
        self.assertIn("model_name", health.json())
        self.assertIn("model_status", health.json())
        self.assertEqual(health.json()["stream_format"], "flat")

        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("Text GPT-Live", index.text)

    def test_session_tick_and_reset(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={"instruction": "Highlight every animal I mention."},
        ).json()
        session_id = created["id"]
        self.assertEqual(created["instruction"], "Highlight every animal I mention.")
        self.assertIn("highlight", created["permissions"]["ui"])
        self.assertEqual(created["highlights"], [])

        tick = self.client.post(
            f"/api/sessions/{session_id}/tick",
            json={"text": "hello there", "state": "active"},
        )
        self.assertEqual(tick.status_code, 200)
        body = tick.json()
        self.assertEqual(body["turn"]["event"]["index"], 1)
        self.assertEqual(body["turn"]["action"]["kind"], "idle")
        self.assertEqual(len(body["session"]["history"]), 1)
        self.assertEqual(body["session"]["instruction"], "Highlight every animal I mention.")

        reset = self.client.post(f"/api/sessions/{session_id}/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["history"], [])
        self.assertEqual(reset.json()["next_index"], 1)

    def test_empty_text_tick_is_recorded_as_a_real_event(self) -> None:
        created = self.client.post(
            "/api/sessions",
            json={"instruction": "Remain present during silence."},
        ).json()
        session_id = created["id"]

        tick = self.client.post(
            f"/api/sessions/{session_id}/tick",
            json={"text": "", "state": "idle"},
        )

        self.assertEqual(tick.status_code, 200)
        event = tick.json()["turn"]["event"]
        self.assertEqual(event["content"], "")
        self.assertEqual(event["state"], "idle")
        self.assertEqual(tick.json()["session"]["history"][0]["event"], event)

    def test_unknown_session_is_404(self) -> None:
        response = self.client.get("/api/sessions/not-a-session")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
