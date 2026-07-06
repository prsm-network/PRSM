"""Local compute job store.

Keeps a JSON file at ~/.prsm/compute_jobs.json so both the P2P node
and the CLI can share job state without an API server.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


JOBS_FILE = Path.home() / ".prsm" / "compute_jobs.json"


@dataclass
class JobRecord:
    job_id: str
    prompt: str
    model: str = "nwtn"
    status: str = "pending"          # pending | running | completed | failed | cancelled
    max_tokens: int = 1000
    budget: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


def _load() -> Dict[str, dict]:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: Dict[str, dict]) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(data, indent=2, default=str))


def list_jobs(limit: int = 20) -> List[dict]:
    jobs = list(_load().values())
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return jobs[:limit]


def get_job(job_id: str) -> Optional[dict]:
    return _load().get(job_id)


def create_job(
    prompt: str,
    model: str = "nwtn",
    max_tokens: int = 1000,
    budget: Optional[float] = None,
    job_id: Optional[str] = None,
    status: str = "pending",
) -> dict:
    import uuid
    # sp1391 — allow pinning the id to the daemon's job_id so the CLI cache and the node agree.
    job = JobRecord(
        job_id=job_id or uuid.uuid4().hex[:12],
        prompt=prompt,
        model=model,
        status=status,
        max_tokens=max_tokens,
        budget=budget,
    )
    data = _load()
    data[job.job_id] = asdict(job)
    _save(data)
    return asdict(job)


def update_job(job_id: str, **kw: Any) -> Optional[dict]:
    data = _load()
    if job_id not in data:
        return None
    data[job_id].update(kw)
    _save(data)
    return data[job_id]
