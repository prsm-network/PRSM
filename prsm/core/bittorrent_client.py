"""
BitTorrent Client
=================

Core libtorrent wrapper for PRSM's BitTorrent integration.
Provides async interface for creating, seeding, and downloading torrents.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Check if libtorrent is available
try:
    import libtorrent as lt
    LT_AVAILABLE = True
except ImportError:
    LT_AVAILABLE = False
    lt = None
    logger.warning(
        "libtorrent not available — BitTorrent features disabled. "
        "Install with: pip install libtorrent>=2.0.9"
    )


class TorrentState(str, Enum):
    """Torrent download/seeding state."""
    QUEUED = "queued"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    SEEDING = "seeding"
    PAUSED = "paused"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class BitTorrentConfig:
    """Configuration for BitTorrent client."""
    port_range_start: int = 6881
    port_range_end: int = 6891
    dht_enabled: bool = True
    dht_bootstrap_nodes: List[str] = field(default_factory=lambda: [
        "router.bittorrent.com:6881",
        "router.utorrent.com:6881",
        "dht.transmissionbt.com:6881",
    ])
    download_dir: str = "~/.prsm/torrents"
    max_uploads: int = 4
    max_connections: int = 50
    upload_rate_limit: int = 0  # 0 = unlimited
    download_rate_limit: int = 0  # 0 = unlimited
    alert_poll_interval: float = 0.1  # seconds
    piece_length: int = 262144  # 256KB default

    def get_download_dir(self) -> Path:
        """Get expanded download directory path."""
        return Path(self.download_dir).expanduser()


@dataclass
class BitTorrentResult:
    """Result of a BitTorrent operation."""
    success: bool
    infohash: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileEntry:
    """File within a torrent."""
    path: str
    size_bytes: int
    offset_in_torrent: int = 0


@dataclass
class PeerInfo:
    """Information about a connected peer."""
    peer_id: str
    ip: str
    port: int
    client: str = ""
    downloaded: int = 0
    uploaded: int = 0
    is_seed: bool = False


@dataclass
class TorrentInfo:
    """Live status information for a torrent."""
    infohash: str
    name: str
    size_bytes: int
    piece_length: int
    num_pieces: int
    files: List[FileEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    seeders: int = 0
    leechers: int = 0
    download_rate: float = 0.0
    upload_rate: float = 0.0
    progress: float = 0.0
    state: TorrentState = TorrentState.UNKNOWN
    bytes_downloaded: int = 0
    bytes_uploaded: int = 0
    eta_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "infohash": self.infohash,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "piece_length": self.piece_length,
            "num_pieces": self.num_pieces,
            "files": [{"path": f.path, "size_bytes": f.size_bytes} for f in self.files],
            "created_at": self.created_at,
            "seeders": self.seeders,
            "leechers": self.leechers,
            "download_rate": self.download_rate,
            "upload_rate": self.upload_rate,
            "progress": self.progress,
            "state": self.state.value,
            "bytes_downloaded": self.bytes_downloaded,
            "bytes_uploaded": self.bytes_uploaded,
            "eta_seconds": self.eta_seconds,
            "error": self.error,
        }


class BitTorrentClient:
    """
    Async wrapper around libtorrent for BitTorrent operations.

    All libtorrent calls are wrapped in run_in_executor to keep
    everything non-blocking. Alert polling runs as a background task.
    """

    def __init__(self, config: Optional[BitTorrentConfig] = None):
        self.config = config or BitTorrentConfig()
        self._session: Optional[Any] = None
        self._torrents: Dict[str, Any] = {}  # infohash -> torrent_handle
        self._alert_callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._alert_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False

    @property
    def available(self) -> bool:
        """Check if BitTorrent is available."""
        return LT_AVAILABLE and self._initialized

    async def initialize(self) -> bool:
        """
        Initialize the libtorrent session.

        Returns True if successful, False otherwise.
        """
        if not LT_AVAILABLE:
            logger.warning("Cannot initialize BitTorrent: libtorrent not installed")
            return False

        if self._initialized:
            return True

        try:
            self._loop = asyncio.get_event_loop()

            # Sprint 179 — libtorrent 2.0.12+ removed `settings_pack`
            # from Python bindings; use the dict-based `default_settings()`
            # API instead. The old settings_pack path is retained as a
            # legacy branch for libtorrent <2.0.12.
            if hasattr(lt, "default_settings"):
                settings = lt.default_settings()
                settings["listen_interfaces"] = (
                    f"0.0.0.0:{self.config.port_range_start}"
                )
                if self.config.dht_enabled:
                    settings["enable_dht"] = True
                settings["upload_rate_limit"] = (
                    self.config.upload_rate_limit * 1024
                )
                settings["download_rate_limit"] = (
                    self.config.download_rate_limit * 1024
                )
                self._session = lt.session(settings)
            else:
                settings = lt.settings_pack()
                settings.set_int(lt.settings_pack.listen_interfaces,
                               f"0.0.0.0:{self.config.port_range_start}")
                if self.config.dht_enabled:
                    settings.set_bool(lt.settings_pack.enable_dht, True)
                settings.set_int(lt.settings_pack.upload_rate_limit,
                               self.config.upload_rate_limit * 1024)
                settings.set_int(lt.settings_pack.download_rate_limit,
                               self.config.download_rate_limit * 1024)
                self._session = lt.session(settings)

            # DHT bootstrap nodes (same on both API paths).
            # Sprint 179 — libtorrent 2.0.12+ moved add_dht_router from
            # module-level to session method; older versions exposed
            # both. Prefer the method form when available.
            if self.config.dht_enabled and self._session is not None:
                for node in self.config.dht_bootstrap_nodes:
                    host, port = node.rsplit(":", 1)
                    if hasattr(self._session, "add_dht_router"):
                        self._session.add_dht_router(host, int(port))
                    elif hasattr(lt, "add_dht_router"):
                        lt.add_dht_router(self._session, host, int(port))

            # Ensure download directory exists
            download_dir = self.config.get_download_dir()
            download_dir.mkdir(parents=True, exist_ok=True)

            self._initialized = True
            self._running = True

            # Start alert polling
            self._alert_task = asyncio.create_task(self._poll_alerts())

            logger.info(
                f"BitTorrent client initialized on port {self.config.port_range_start}, "
                f"download_dir={download_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BitTorrent client: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the BitTorrent client."""
        if not self._initialized:
            return

        self._running = False

        # Stop alert polling
        if self._alert_task:
            self._alert_task.cancel()
            try:
                await self._alert_task
            except asyncio.CancelledError:
                pass

        # Pause all torrents
        for infohash, handle in list(self._torrents.items()):
            try:
                handle.pause()
            except Exception:
                pass

        # Save resume data and close session
        if self._session:
            try:
                # Save resume data for each torrent
                for handle in self._torrents.values():
                    if handle.is_valid() and handle.has_metadata():
                        handle.save_resume_data()
                # Give time for resume data to save
                await asyncio.sleep(0.5)
                del self._session
            except Exception as e:
                logger.debug(f"Error during shutdown: {e}")

        self._torrents.clear()
        self._initialized = False
        logger.info("BitTorrent client shutdown complete")

    async def create_torrent(
        self,
        path: Path,
        piece_length: int = 262144,
        comment: str = "",
        private: bool = False,
    ) -> BitTorrentResult:
        """
        Create a .torrent file from a file or directory.

        Args:
            path: Path to file or directory to torrent
            piece_length: Size of each piece in bytes (default 256KB)
            comment: Optional comment to embed in torrent
            private: If True, disable DHT/PEX

        Returns:
            BitTorrentResult with infohash and torrent bytes
        """
        if not self.available:
            return BitTorrentResult(success=False, error="BitTorrent client not initialized")

        path = Path(path).expanduser()
        if not path.exists():
            return BitTorrentResult(success=False, error=f"Path does not exist: {path}")

        try:
            def _create():
                # Create file storage
                fs = lt.file_storage()

                if path.is_file():
                    # sp1071 — add by BASENAME (not str(path)). With a full path,
                    # libtorrent builds a MULTI-file torrent named after the parent
                    # dir (e.g. the staging-dir basename), so the infohash depended on
                    # the operator's filesystem layout AND couldn't be reproduced by a
                    # non-libtorrent node. The basename yields a clean single-file
                    # torrent whose v1 infohash == prsm.core.torrent_infohash's
                    # pure-Python value (golden-verified). set_piece_hashes below reads
                    # the bytes from path.parent, so the file still resolves.
                    fs.add_file(path.name, path.stat().st_size)
                else:
                    # Add directory recursively
                    for f in sorted(path.rglob("*")):
                        if f.is_file():
                            rel_path = f.relative_to(path)
                            fs.add_file(str(rel_path), f.stat().st_size)

                # Create the torrent. sp1071 — pin to v1_only so the infohash is the
                # canonical BEP-3 v1 SHA-1 that prsm.core.torrent_infohash computes in
                # pure Python (libtorrent 2.x otherwise defaults to HYBRID v1+v2, whose
                # info dict carries v2 file-tree/piece-layers fields → a different
                # infohash a non-libtorrent node can't reproduce). v1_only keeps every
                # node — libtorrent or not — assigning the SAME CID to the same bytes.
                # Older libtorrent (1.x) is v1-only already and lacks the flag.
                _v1_only = getattr(lt.create_torrent, "v1_only", None)
                if _v1_only is not None:
                    t = lt.create_torrent(fs, piece_size=piece_length, flags=_v1_only)
                else:
                    t = lt.create_torrent(fs, piece_size=piece_length)

                # Set comment if provided
                if comment:
                    t.set_comment(comment)

                # `set_priv` was removed from create_torrent in
                # libtorrent 2.0.12+. Older API takes a flag here;
                # newer API exposes `priv` directly. Both fail-soft.
                if private:
                    if hasattr(t, "set_priv"):
                        t.set_priv(True)
                    elif hasattr(t, "priv"):
                        try:
                            t.priv = True
                        except Exception:
                            pass

                # Read files and compute hashes
                lt.set_piece_hashes(t, str(path.parent if path.is_file() else path))

                # Generate the .torrent entry
                torrent_bytes = lt.bencode(t.generate())

                # Sprint 179 — libtorrent 2.0.12+ removed
                # `create_torrent.get_torrent_info()`. The new API
                # path is to load the bencoded data back into a
                # `torrent_info` object. We try the old method first
                # for back-compat.
                if hasattr(t, "get_torrent_info"):
                    info = t.get_torrent_info()
                else:
                    info = lt.torrent_info(lt.bdecode(torrent_bytes))
                infohash = str(info.info_hash())

                return infohash, torrent_bytes, info

            # Run in executor to avoid blocking
            infohash, torrent_bytes, info = await self._loop.run_in_executor(None, _create)

            return BitTorrentResult(
                success=True,
                infohash=infohash,
                metadata={
                    "torrent_bytes": torrent_bytes,
                    "name": info.name() if info else path.name,
                    "total_size": info.total_size() if info else 0,
                    "piece_length": info.piece_length() if info else piece_length,
                    "num_pieces": info.num_pieces() if info else 0,
                }
            )

        except Exception as e:
            logger.error(f"Failed to create torrent: {e}")
            return BitTorrentResult(success=False, error=str(e))

    async def add_torrent(
        self,
        source: Union[bytes, str],
        save_path: Optional[Path] = None,
        seed_mode: bool = False,
    ) -> BitTorrentResult:
        """
        Add a torrent from .torrent bytes or magnet URI.

        Args:
            source: Either raw .torrent file bytes or a magnet URI string
            save_path: Where to save/download files (default: config download_dir)
            seed_mode: If True, assume we already have all data (seeding only)

        Returns:
            BitTorrentResult with infohash
        """
        if not self.available:
            return BitTorrentResult(success=False, error="BitTorrent client not initialized")

        save_path = save_path or self.config.get_download_dir()

        try:
            def _add():
                params = lt.add_torrent_params()
                params.save_path = str(save_path)

                if seed_mode:
                    params.flags |= lt.torrent_flags.seed_mode

                if isinstance(source, bytes):
                    # Raw torrent bytes
                    params.ti = lt.torrent_info(source)
                elif isinstance(source, str) and source.startswith("magnet:"):
                    # Magnet URI
                    params = lt.parse_magnet_uri(source)
                    params.save_path = str(save_path)
                    if seed_mode:
                        params.flags |= lt.torrent_flags.seed_mode
                else:
                    raise ValueError("source must be .torrent bytes or magnet URI")

                handle = self._session.add_torrent(params)
                return str(handle.info_hash()), handle

            infohash, handle = await self._loop.run_in_executor(None, _add)
            self._torrents[infohash] = handle

            logger.info(f"Added torrent: {infohash[:16]}... to {save_path}")
            return BitTorrentResult(success=True, infohash=infohash)

        except Exception as e:
            logger.error(f"Failed to add torrent: {e}")
            return BitTorrentResult(success=False, error=str(e))

    async def remove_torrent(
        self,
        infohash: str,
        delete_files: bool = False,
    ) -> BitTorrentResult:
        """
        Remove a torrent from the session.

        Args:
            infohash: Infohash of torrent to remove
            delete_files: If True, also delete downloaded files

        Returns:
            BitTorrentResult indicating success
        """
        if not self.available:
            return BitTorrentResult(success=False, error="BitTorrent client not initialized")

        handle = self._torrents.get(infohash)
        if not handle:
            return BitTorrentResult(success=False, error="Torrent not found")

        try:
            self._session.remove_torrent(handle, delete_files)
            del self._torrents[infohash]

            logger.info(f"Removed torrent: {infohash[:16]}... (delete_files={delete_files})")
            return BitTorrentResult(success=True, infohash=infohash)

        except Exception as e:
            logger.error(f"Failed to remove torrent: {e}")
            return BitTorrentResult(success=False, error=str(e))

    async def get_status(
        self,
        infohash: Optional[str] = None,
    ) -> Union[TorrentInfo, List[TorrentInfo]]:
        """
        Get status for one or all active torrents.

        Args:
            infohash: Specific torrent to get status for, or None for all

        Returns:
            TorrentInfo for a single torrent, or list of TorrentInfo for all
        """
        if not self.available:
            if infohash:
                return TorrentInfo(infohash=infohash, name="", size_bytes=0,
                                  piece_length=0, num_pieces=0, state=TorrentState.ERROR,
                                  error="Client not initialized")
            return []

        if infohash:
            return await self._get_single_status(infohash)

        # Return all torrents
        results = []
        for ih in list(self._torrents.keys()):
            status = await self._get_single_status(ih)
            results.append(status)
        return results

    async def _get_single_status(self, infohash: str) -> TorrentInfo:
        """Get status for a single torrent."""
        handle = self._torrents.get(infohash)
        if not handle or not handle.is_valid():
            return TorrentInfo(
                infohash=infohash,
                name="",
                size_bytes=0,
                piece_length=0,
                num_pieces=0,
                state=TorrentState.ERROR,
                error="Invalid or unknown torrent"
            )

        try:
            status = handle.status()
            torrent_info = handle.get_torrent_info() if handle.has_metadata() else None

            # Map libtorrent state to our enum
            state_map = {
                lt.torrent_status.checking_files: TorrentState.CHECKING,
                lt.torrent_status.checking_resume_data: TorrentState.CHECKING,
                lt.torrent_status.downloading: TorrentState.DOWNLOADING,
                lt.torrent_status.finished: TorrentState.SEEDING,
                lt.torrent_status.seeding: TorrentState.SEEDING,
                lt.torrent_status.allocating: TorrentState.QUEUED,
                lt.torrent_status.queued_for_checking: TorrentState.QUEUED,
            }

            state = state_map.get(status.state, TorrentState.UNKNOWN)
            if status.paused:
                state = TorrentState.PAUSED

            # Get file list
            files = []
            if torrent_info:
                for i in range(torrent_info.num_files()):
                    f = torrent_info.file_at(i)
                    files.append(FileEntry(
                        path=f.path,
                        size_bytes=f.size,
                        offset_in_torrent=f.offset,
                    ))

            # Calculate ETA
            eta = 0.0
            if status.download_rate > 0 and status.total_wanted > status.total_wanted_done:
                remaining = status.total_wanted - status.total_wanted_done
                eta = remaining / status.download_rate

            return TorrentInfo(
                infohash=infohash,
                name=status.name if status.name else (torrent_info.name() if torrent_info else ""),
                size_bytes=status.total_wanted if status.total_wanted > 0 else (torrent_info.total_size() if torrent_info else 0),
                piece_length=torrent_info.piece_length() if torrent_info else 0,
                num_pieces=torrent_info.num_pieces() if torrent_info else 0,
                files=files,
                seeders=status.num_seeds,
                leechers=status.num_peers - status.num_seeds,
                download_rate=status.download_rate,
                upload_rate=status.upload_rate,
                progress=status.progress,
                state=state,
                bytes_downloaded=status.total_done,
                bytes_uploaded=status.total_upload,
                eta_seconds=eta,
                error=status.error if status.error else None,
            )

        except Exception as e:
            logger.error(f"Error getting torrent status: {e}")
            return TorrentInfo(
                infohash=infohash,
                name="",
                size_bytes=0,
                piece_length=0,
                num_pieces=0,
                state=TorrentState.ERROR,
                error=str(e)
            )

    async def wait_for_completion(
        self,
        infohash: str,
        timeout: float = 3600.0,
        progress_callback: Optional[Callable[[str, float, Dict], None]] = None,
    ) -> BitTorrentResult:
        """
        Wait for a torrent to finish downloading.

        Args:
            infohash: Torrent to wait for
            timeout: Maximum time to wait in seconds
            progress_callback: Optional callback(infohash, progress, stats)

        Returns:
            BitTorrentResult indicating success or timeout/error
        """
        if not self.available:
            return BitTorrentResult(success=False, error="BitTorrent client not initialized")

        handle = self._torrents.get(infohash)
        if not handle:
            return BitTorrentResult(success=False, error="Torrent not found")

        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self._get_single_status(infohash)

            if status.state == TorrentState.SEEDING or status.progress >= 1.0:
                return BitTorrentResult(
                    success=True,
                    infohash=infohash,
                    metadata={"bytes_downloaded": status.bytes_downloaded}
                )

            if status.state == TorrentState.ERROR:
                return BitTorrentResult(
                    success=False,
                    infohash=infohash,
                    error=status.error or "Unknown error"
                )

            if progress_callback:
                try:
                    progress_callback(infohash, status.progress, status.to_dict())
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")

            await asyncio.sleep(1.0)

        return BitTorrentResult(
            success=False,
            infohash=infohash,
            error=f"Timeout after {timeout} seconds"
        )

    async def get_peers(self, infohash: str) -> List[PeerInfo]:
        """
        Get list of peers connected for a torrent.

        Args:
            infohash: Torrent to get peers for

        Returns:
            List of PeerInfo for connected peers
        """
        if not self.available:
            return []

        handle = self._torrents.get(infohash)
        if not handle or not handle.is_valid():
            return []

        try:
            peer_infos = handle.get_peer_info()
            result = []
            for p in peer_infos:
                result.append(PeerInfo(
                    peer_id=p.pid.to_string() if p.pid else "",
                    ip=p.ip[0] if p.ip else "",
                    port=p.ip[1] if p.ip else 0,
                    client=p.client if hasattr(p, 'client') else "",
                    downloaded=p.total_download,
                    uploaded=p.total_upload,
                    is_seed=p.flags & lt.peer_info.seed,
                ))
            return result
        except Exception as e:
            logger.debug(f"Error getting peers: {e}")
            return []

    def on_alert(self, alert_type: str, callback: Callable) -> None:
        """
        Register a callback for a specific alert type.

        Args:
            alert_type: Alert type name (e.g., "torrent_finished", "piece_finished")
            callback: Function to call when alert is received
        """
        self._alert_callbacks.setdefault(alert_type, []).append(callback)

    async def _poll_alerts(self) -> None:
        """Background task to poll libtorrent alerts."""
        while self._running and self._session:
            try:
                alerts = self._session.pop_alerts()
                for alert in alerts:
                    await self._handle_alert(alert)
            except Exception as e:
                if self._running:
                    logger.debug(f"Alert polling error: {e}")

            await asyncio.sleep(self.config.alert_poll_interval)

    async def _handle_alert(self, alert: Any) -> None:
        """Process a single libtorrent alert."""
        alert_type = type(alert).__name__

        # Get infohash if applicable
        infohash = None
        if hasattr(alert, 'handle'):
            try:
                infohash = str(alert.handle.info_hash())
            except Exception:
                pass

        # Call registered callbacks
        callbacks = self._alert_callbacks.get(alert_type, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(alert, infohash)
                else:
                    cb(alert, infohash)
            except Exception as e:
                logger.debug(f"Alert callback error ({alert_type}): {e}")

        # Log important alerts
        if alert_type == "torrent_finished" and infohash:
            logger.info(f"Torrent finished: {infohash[:16]}...")
        elif alert_type == "torrent_error" and infohash:
            logger.error(f"Torrent error ({infohash[:16]}...): {alert.error if hasattr(alert, 'error') else alert}")


# Convenience function for getting a client instance
_bittorrent_client: Optional[BitTorrentClient] = None


def get_bittorrent_client(config: Optional[BitTorrentConfig] = None) -> BitTorrentClient:
    """Get or create a global BitTorrent client instance."""
    global _bittorrent_client
    if _bittorrent_client is None:
        _bittorrent_client = BitTorrentClient(config)
    return _bittorrent_client
