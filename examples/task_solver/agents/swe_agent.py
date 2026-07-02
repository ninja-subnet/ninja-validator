"""Minimal *real* sample agent: calls the model through the validator proxy.

Demonstrates the secure LLM path (sandbox -> proxy -> upstream): it only ever sees
``api_base`` (the proxy URL) and ``api_key`` (the per-solve token), never the real
upstream key. It is NOT a competitive SWE agent — it asks the model for a short fix
and writes it, producing a diff and spending real tokens against whatever
``LLM_PROVIDER`` the worker is configured for. The sandbox image ships ``openai``.

``seed_duel.py`` replaces ``MARKER`` per submission.
"""

MARKER = "agent"


def solve(repo_path, issue, model, api_base, api_key):  # noqa: ANN001, ANN201
    import pathlib

    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a terse software engineer."},
            {"role": "user", "content": f"Propose a minimal fix for this task in <=5 lines:\n\n{issue}"},
        ],
    )
    note = (resp.choices[0].message.content or "").strip()
    out = pathlib.Path(repo_path) / "SOLUTION.md"
    out.write_text(f"# Fix proposed by {MARKER}\n\n{note}\n", encoding="utf-8")
    return {"success": True, "patch": ""}
