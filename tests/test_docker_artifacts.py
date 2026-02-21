from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exists_and_has_expected_runtime_contract() -> None:
    dockerfile = PROJECT_ROOT / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile is required for Step 5 productionization."
    text = dockerfile.read_text(encoding="utf-8")

    assert "FROM python:3.13-slim" in text
    assert "USER agri" in text
    assert "ENTRYPOINT [\"python\", \"-m\", \"agri_auditor\"]" in text
    assert "CMD [\"process\"" in text
    assert "--data-dir\", \"/data\"" in text


def test_dockerignore_excludes_local_artifacts_and_env_files() -> None:
    dockerignore = PROJECT_ROOT / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore is required for clean production builds."
    text = dockerignore.read_text(encoding="utf-8")

    expected_entries = [
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        "artifacts/",
        ".env",
    ]
    for entry in expected_entries:
        assert entry in text, f"Missing .dockerignore entry: {entry}"


def test_ci_workflow_includes_required_gemini_live_gate() -> None:
    workflow = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "CI workflow is required for mandatory gemini_live gating."
    text = workflow.read_text(encoding="utf-8")
    assert "gemini-live:" in text
    assert "GEMINI_API_KEY" in text
    assert "-m gemini_live" in text
