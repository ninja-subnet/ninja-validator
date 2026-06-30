import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from github_miner import CommitCandidate, CommitFile
from task_generation import generate_task_description


def _candidate() -> CommitCandidate:
    return CommitCandidate(
        repo_full_name="owner/repo",
        repo_clone_url="https://github.com/owner/repo.git",
        commit_sha="abcdef123456",
        parent_sha="parent",
        message="Improve widgets",
        html_url="https://github.com/owner/repo/commit/abcdef123456",
        author_name=None,
        event_id="event",
        files=[
            CommitFile(
                filename="src/app.py",
                status="modified",
                additions=120,
                deletions=5,
                changes=125,
                patch="@@ -1 +1 @@\n-old\n+new",
            ),
        ],
    )


class TaskGenerationTest(unittest.TestCase):
    def test_http_status_error_uses_fallback_task(self):
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(400, request=request, text='{"error":"bad request"}')
        error = httpx.HTTPStatusError("bad request", request=request, response=response)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("task_generation._run_claude", side_effect=error):
                task = generate_task_description(
                    candidate=_candidate(),
                    prompt_dir=Path(tmp),
                    model="Qwen/Qwen3-32B",
                    timeout=10,
                    openrouter_api_key="key",
                )

        self.assertEqual(task.title, "Improve widgets")
        self.assertIn("HTTP 400", task.raw_output)
        self.assertIn("src/app.py", task.prompt_text)


if __name__ == "__main__":
    unittest.main()
