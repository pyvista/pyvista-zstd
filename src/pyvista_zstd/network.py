"""
Read individual compressed field arrays over strict, version-pinned HTTP ranges.

Install ``pyvista-zstd[network]`` to use this optional module. A producer exports
the small index with :func:`build_range_index`; a host serves that JSON and the
original container with byte-range support and the matching ETag. No full-file
download fallback is permitted.
"""

from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import zstandard

from pyvista_zstd import _capi

if TYPE_CHECKING:
    import httpx
    from numpy.typing import NDArray


def file_etag(path: Path | str) -> str:
    """Return a quoted file-revision validator without reading container payloads."""
    stat = Path(path).stat()
    return f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'


def build_range_index(filename: Path | str) -> dict[str, Any]:
    """
    Describe field-array byte ranges without decompressing their payloads.

    The native core validates the source container. This function reads only
    compressed array headers and the native index. The exported index contains
    two compressed frames per field: its header and its payload. Remote readers
    reassemble those frames as a minimal container and use the same native
    decoder, including byte shuffle, dtype validation, and bounds checks.

    Parameters
    ----------
    filename : pathlib.Path or str
        Committed local ``.pv`` file. Do not modify it during index generation.

    Returns
    -------
    dict
        JSON-serializable sidecar, containing a source ETag and named ranges.

    """
    path = Path(filename)
    etag = file_etag(path)
    fields: dict[str, str] = {}
    ranges: dict[str, dict[str, int]] = {}
    with _capi.CoreReader(path) as core, path.open("rb") as source:
        sizes, compressed = core.frame_sizes()
        names = core.names()
        for name in core.field_array_names():
            index = core.find_field(name)
            if index is not None:
                fields[names[index]] = name
        ends = np.cumsum(compressed, dtype=np.uint64)
        decoder = zstandard.ZstdDecompressor()
        for frame in range(0, len(compressed), 2):
            start = int(ends[frame - 1]) if frame else 0
            source.seek(start)
            header = decoder.decompress(source.read(int(compressed[frame])), max_output_size=int(sizes[frame]))
            name_length = struct.unpack_from("<I", header)[0]
            full_name = header[4 : 4 + name_length].decode("utf-8")
            if full_name in fields:
                ranges[fields[full_name]] = {
                    "start": start,
                    "end": int(ends[frame + 1]),
                    "header_compressed": int(compressed[frame]),
                    "header_size": int(sizes[frame]),
                    "payload_size": int(sizes[frame + 1]),
                }
    if file_etag(path) != etag:
        msg = "Container changed while generating the range index"
        raise RuntimeError(msg)
    return {"schema": 1, "etag": etag, "size": path.stat().st_size, "arrays": ranges}


class HTTPRangeReader:
    """
    Fetch and decode one field array without reading any other result blocks.

    Parameters
    ----------
    url : str
        Container URL. The host must implement Range and If-Match.
    index : dict
        Sidecar produced by :func:`build_range_index` for that exact revision.
    client : httpx.Client
        Caller-owned client, allowing connection reuse and host authentication.

    Notes
    -----
    The file must be immutable for the sidecar's lifetime. An append replaces
    the file revision; refresh the sidecar before opening a newer result set.
    Responses must be 206 with identity content encoding, the exact byte range,
    and the same ETag. A 200 response is refused before its body is read.

    """

    def __init__(self, url: str, index: dict[str, Any], *, client: httpx.Client) -> None:
        """Bind the reader to a specific committed container revision."""
        if index.get("schema") != 1 or not index.get("etag") or not isinstance(index.get("arrays"), dict):
            msg = "Unsupported or incomplete range index"
            raise ValueError(msg)
        self.url = url
        self.index = index
        self.client = client
        self.bytes_received = 0

    def read_array(self, name: str) -> NDArray[Any]:
        """Fetch only the named field's compressed frames and decode them natively."""
        selected = self.index["arrays"][name]
        start, end = selected["start"], selected["end"]
        if not 0 <= start < end <= self.index["size"]:
            msg = "Array range lies outside the indexed container"
            raise ValueError(msg)
        headers = {
            "Range": f"bytes={start}-{end - 1}",
            "If-Match": self.index["etag"],
            "Accept-Encoding": "identity",
        }
        with self.client.stream("GET", self.url, headers=headers) as response:
            self._validate_response(response, start, end)
            payload = bytearray()
            for chunk in response.iter_raw(chunk_size=64 * 1024):
                if len(payload) + len(chunk) > end - start:
                    msg = "HTTP response exceeded the requested byte range"
                    raise ValueError(msg)
                payload.extend(chunk)
                self.bytes_received += len(chunk)
        if len(payload) != end - start:
            msg = "HTTP response was shorter than the requested byte range"
            raise ValueError(msg)
        header_end = selected["header_compressed"]
        if not 0 < header_end < len(payload):
            msg = "Indexed header boundary lies outside its array range"
            raise ValueError(msg)
        payload.extend(
            struct.pack("<QQQQQ", header_end, selected["header_size"], end - start, selected["payload_size"], 2)
        )
        with _capi.CoreReader(buffer=payload) as core:
            return core.read_at(0)

    def _validate_response(self, response: httpx.Response, start: int, end: int) -> None:
        if response.status_code != HTTPStatus.PARTIAL_CONTENT:
            msg = f"Expected HTTP 206 for a partial read; received {response.status_code}"
            raise ValueError(msg)
        expected = f"bytes {start}-{end - 1}/{self.index['size']}"
        if response.headers.get("Content-Range") != expected:
            msg = "HTTP Content-Range does not match the requested array"
            raise ValueError(msg)
        if response.headers.get("ETag") != self.index["etag"]:
            msg = "Remote container revision does not match the range index"
            raise ValueError(msg)
        if response.headers.get("Content-Encoding", "identity") != "identity":
            msg = "Byte-range responses must use identity content encoding"
            raise ValueError(msg)
        declared_size = response.headers.get("Content-Length")
        if declared_size is not None and int(declared_size) != end - start:
            msg = "HTTP Content-Length does not match the requested byte range"
            raise ValueError(msg)


def write_range_index(filename: Path | str, destination: Path | str) -> None:
    """Export a local container's compressed field ranges as a JSON sidecar."""
    Path(destination).write_text(json.dumps(build_range_index(filename)) + "\n", encoding="utf-8")
