"""
BitTorrent API Router
=====================

FastAPI router for BitTorrent torrent management.
All endpoints require JWT authentication.
"""

import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from prsm.interface.auth import get_current_user
from prsm.core.bittorrent_client import (
    BitTorrentClient,
    TorrentInfo,
)

router = APIRouter(prefix="/api/v1/torrents", tags=["BitTorrent"])


# ── sp1269 — path confinement (audit round 5) ──────────────────────────────────
# create/add seeded an arbitrary local content_path to the PUBLIC swarm (exfiltrate the
# node's private keys); add/download wrote to an arbitrary save_path. Confine both to an
# allowed torrent data root, rejecting traversal / symlink escape.

def _bt_data_roots() -> List[Path]:
    """Allowed roots for torrent content + downloads. Override via PRSM_BT_DATA_ROOT
    (os.pathsep-separated); else the node's <data_dir>/torrents; else ~/.prsm/torrents."""
    roots: List[Path] = []
    for part in os.getenv("PRSM_BT_DATA_ROOT", "").split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser())
    if not roots:
        try:
            from prsm.node.node import get_node
            node = get_node()
            data_dir = getattr(getattr(node, "config", None), "data_dir", None)
            if data_dir:
                roots.append(Path(data_dir) / "torrents")
        except Exception:  # noqa: BLE001 — node not up / no config → fall through to default
            pass
    if not roots:
        roots.append(Path.home() / ".prsm" / "torrents")
    return roots


def _safe_bt_path(user_path: str, *, must_exist: bool) -> Path:
    """Resolve ``user_path`` (FOLLOWING symlinks) and confine it to an allowed BT data root.

    Raises HTTP 403 on traversal / symlink escape outside every allowed root, 400 when
    ``must_exist`` and the resolved path is absent. realpath collapses ``..`` and resolves
    symlinks, so a symlink planted inside a root that points outside it is rejected.
    """
    if not user_path:
        raise HTTPException(status_code=400, detail="path is required")
    roots = [Path(os.path.realpath(r)) for r in _bt_data_roots()]
    for r in roots:
        try:
            r.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 — best-effort; confinement check still runs
            pass
    resolved = Path(os.path.realpath(Path(user_path).expanduser()))
    if not any(resolved == r or r in resolved.parents for r in roots):
        raise HTTPException(
            status_code=403,
            detail="path must be within an allowed torrent data root (PRSM_BT_DATA_ROOT)")
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=400, detail="path does not exist")
    return resolved


# ── Request/Response Models ──────────────────────────────────────────────────

class CreateTorrentRequest(BaseModel):
    """Request to create a new torrent from content."""
    content_path: str = Field(..., description="Path to file or directory to torrent")
    name: Optional[str] = Field(None, description="Optional name for the torrent")
    piece_length: int = Field(262144, description="Size of each piece in bytes")
    provenance_id: Optional[str] = Field(None, description="PRSM provenance ID")


class AddTorrentRequest(BaseModel):
    """Request to add an existing torrent."""
    source: str = Field(..., description="Magnet URI or path to .torrent file")
    save_path: Optional[str] = Field(None, description="Where to save downloaded files")
    seed_mode: bool = Field(False, description="True if we already have the data")


class DownloadRequest(BaseModel):
    """Request to download a torrent."""
    infohash: str = Field(..., description="Infohash of torrent to download")
    save_path: str = Field(..., description="Where to save downloaded files")
    timeout: float = Field(3600.0, description="Maximum time to wait in seconds")


class TorrentResponse(BaseModel):
    """Response for a single torrent."""
    infohash: str
    name: str
    size_bytes: int
    piece_length: int
    num_pieces: int
    progress: float
    state: str
    seeders: int
    leechers: int
    download_rate: float
    upload_rate: float
    bytes_downloaded: int
    bytes_uploaded: int
    eta_seconds: float
    error: Optional[str] = None


class TorrentListResponse(BaseModel):
    """Response for listing torrents."""
    torrents: List[TorrentResponse]
    total: int


class DownloadStatusResponse(BaseModel):
    """Response for download status."""
    request_id: str
    infohash: str
    status: str
    progress: float
    bytes_downloaded: int
    total_bytes: int
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """Response for BitTorrent statistics."""
    available: bool
    provider: Optional[Dict[str, Any]] = None
    requester: Optional[Dict[str, Any]] = None


# ── Helper Functions ────────────────────────────────────────────────────────

def _get_bt_client() -> BitTorrentClient:
    """Get the BitTorrent client instance."""
    from prsm.node.node import get_node
    node = get_node()
    if not node or not node.bt_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BitTorrent client not available"
        )
    return node.bt_client


def _torrent_to_response(info: TorrentInfo) -> TorrentResponse:
    """Convert TorrentInfo to response model."""
    return TorrentResponse(
        infohash=info.infohash,
        name=info.name,
        size_bytes=info.size_bytes,
        piece_length=info.piece_length,
        num_pieces=info.num_pieces,
        progress=info.progress,
        state=info.state.value,
        seeders=info.seeders,
        leechers=info.leechers,
        download_rate=info.download_rate,
        upload_rate=info.upload_rate,
        bytes_downloaded=info.bytes_downloaded,
        bytes_uploaded=info.bytes_uploaded,
        eta_seconds=info.eta_seconds,
        error=info.error,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/create", response_model=TorrentResponse)
async def create_torrent(
    request: CreateTorrentRequest,
    user: Dict = Depends(get_current_user),
) -> TorrentResponse:
    """
    Create a new torrent from local content and begin seeding.

    Requires authentication. The content must exist on the local filesystem.
    """
    from prsm.node.node import get_node

    node = get_node()
    if not node or not node.bt_provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BitTorrent provider not available"
        )

    manifest = await node.bt_provider.seed_content(
        path=_safe_bt_path(request.content_path, must_exist=True),  # sp1269 — confine + no escape
        name=request.name,
        provenance_id=request.provenance_id,
        piece_length=request.piece_length,
    )

    if not manifest:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create torrent"
        )

    # Get the status
    status_info = await node.bt_client.get_status(manifest.infohash)
    if isinstance(status_info, TorrentInfo):
        return _torrent_to_response(status_info)

    # Return basic info if status not available yet
    return TorrentResponse(
        infohash=manifest.infohash,
        name=manifest.name,
        size_bytes=manifest.total_size,
        piece_length=manifest.piece_length,
        num_pieces=manifest.num_pieces,
        progress=0.0,
        state="seeding",
        seeders=0,
        leechers=0,
        download_rate=0.0,
        upload_rate=0.0,
        bytes_downloaded=0,
        bytes_uploaded=0,
        eta_seconds=0.0,
    )


@router.post("/add", response_model=TorrentResponse)
async def add_torrent(
    request: AddTorrentRequest,
    user: Dict = Depends(get_current_user),
) -> TorrentResponse:
    """
    Add an existing torrent by magnet URI or .torrent file path.
    """
    bt_client = _get_bt_client()

    source = request.source
    if source.startswith("magnet:"):
        # Magnet URI — bt_client.add_torrent accepts it directly
        pass  # intentional: source remains the magnet URI string
    else:
        # Assume it's a .torrent file path — confine it (sp1269: was an arbitrary-file read)
        safe_source = _safe_bt_path(source, must_exist=True)
        try:
            with open(safe_source, "rb") as f:
                source = f.read()
        except HTTPException:
            raise
        except Exception:
            # sp1269 — do NOT echo file contents / raw errors back to the caller
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to read torrent file"
            )

    save_path = _safe_bt_path(request.save_path, must_exist=False) if request.save_path else None

    result = await bt_client.add_torrent(
        source=source,
        save_path=save_path,
        seed_mode=request.seed_mode,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to add torrent"
        )

    # Get the status
    status_info = await bt_client.get_status(result.infohash)
    if isinstance(status_info, TorrentInfo):
        return _torrent_to_response(status_info)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Added torrent but could not get status"
    )


@router.get("", response_model=TorrentListResponse)
async def list_torrents(
    user: Dict = Depends(get_current_user),
) -> TorrentListResponse:
    """List all active torrents (seeding and downloading)."""
    bt_client = _get_bt_client()

    statuses = await bt_client.get_status()

    if isinstance(statuses, list):
        torrents = [_torrent_to_response(s) for s in statuses]
        return TorrentListResponse(torrents=torrents, total=len(torrents))

    return TorrentListResponse(torrents=[], total=0)


@router.get("/{infohash}", response_model=TorrentResponse)
async def get_torrent(
    infohash: str,
    user: Dict = Depends(get_current_user),
) -> TorrentResponse:
    """Get detailed status for a specific torrent."""
    bt_client = _get_bt_client()

    status_info = await bt_client.get_status(infohash)

    if isinstance(status_info, TorrentInfo):
        return _torrent_to_response(status_info)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Torrent not found"
    )


@router.post("/{infohash}/seed", response_model=TorrentResponse)
async def start_seeding(
    infohash: str,
    user: Dict = Depends(get_current_user),
) -> TorrentResponse:
    """Start seeding a torrent (resume if paused)."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENT,
        detail="Seeding control not yet implemented"
    )


@router.delete("/{infohash}/seed")
async def stop_seeding(
    infohash: str,
    user: Dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Stop seeding a torrent."""
    from prsm.node.node import get_node

    node = get_node()
    if not node or not node.bt_provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BitTorrent provider not available"
        )

    success = await node.bt_provider.stop_seeding(infohash)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not stop seeding (not found or minimum seed time not met)"
        )

    return {"success": True, "infohash": infohash}


@router.post("/{infohash}/download", response_model=DownloadStatusResponse)
async def start_download(
    infohash: str,
    request: DownloadRequest,
    user: Dict = Depends(get_current_user),
) -> DownloadStatusResponse:
    """Begin downloading a torrent."""
    from prsm.node.node import get_node

    node = get_node()
    if not node or not node.bt_requester:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BitTorrent requester not available"
        )

    result = await node.bt_requester.request_content(
        infohash=infohash,
        save_path=_safe_bt_path(request.save_path, must_exist=False),  # sp1269 — confine writes
        timeout=request.timeout,
    )

    return DownloadStatusResponse(
        request_id=result.request_id,
        infohash=result.infohash,
        status="completed" if result.success else "failed",
        progress=1.0 if result.success else 0.0,
        bytes_downloaded=result.bytes_downloaded,
        total_bytes=result.bytes_downloaded,
        error=result.error,
    )


@router.get("/{infohash}/download/{request_id}", response_model=DownloadStatusResponse)
async def get_download_status(
    infohash: str,
    request_id: str,
    user: Dict = Depends(get_current_user),
) -> DownloadStatusResponse:
    """Poll the status of a download."""
    from prsm.node.node import get_node

    node = get_node()
    if not node or not node.bt_requester:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BitTorrent requester not available"
        )

    download = node.bt_requester.get_download_status(request_id)

    if not download:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Download request not found"
        )

    return DownloadStatusResponse(
        request_id=download.request_id,
        infohash=download.infohash,
        status=download.status,
        progress=download.bytes_downloaded / max(1, download.total_bytes),
        bytes_downloaded=download.bytes_downloaded,
        total_bytes=download.total_bytes,
        error=download.error,
    )


@router.get("/{infohash}/peers")
async def list_peers(
    infohash: str,
    user: Dict = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List peers connected for a torrent."""
    bt_client = _get_bt_client()

    peers = await bt_client.get_peers(infohash)

    return [
        {
            "peer_id": p.peer_id,
            "ip": p.ip,
            "port": p.port,
            "client": p.client,
            "downloaded": p.downloaded,
            "uploaded": p.uploaded,
            "is_seed": p.is_seed,
        }
        for p in peers
    ]


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    user: Dict = Depends(get_current_user),
) -> StatsResponse:
    """Get aggregate BitTorrent statistics for this node."""
    from prsm.node.node import get_node

    node = get_node()
    if not node:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Node not available"
        )

    return StatsResponse(
        available=node.bt_client.available if node.bt_client else False,
        provider=node.bt_provider.get_stats() if node.bt_provider else None,
        requester=node.bt_requester.get_stats() if node.bt_requester else None,
    )
