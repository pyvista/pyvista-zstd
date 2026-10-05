"""Exercise strict partial HTTP reads against independently compressed fields."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd
from pyvista_zstd.network import HTTPRangeReader
from pyvista_zstd.network import build_range_index
from pyvista_zstd.network import write_range_index


@pytest.mark.parametrize("shuffle", [False, True, "auto"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_reads_only_selected_range(tmp_path, shuffle, dtype) -> None:
    """Decode shuffled and complex arrays with a single bounded HTTPS request."""
    path = tmp_path / "results.pv"
    pyvista_zstd.write(pv.Sphere(), path)
    generator = np.random.default_rng(4)
    arrays = {f"u_{i}": generator.standard_normal((4000, 3)).astype(dtype) for i in range(8)}
    if dtype == np.complex128:
        arrays = {key: value + 1j * value[::-1] for key, value in arrays.items()}
    pyvista_zstd.append_arrays(path, arrays, shuffle=shuffle)
    index = build_range_index(path)
    requests = []

    def serve(request) -> httpx.Response:
        requests.append(request)
        start, end = (int(value) for value in request.headers["Range"].removeprefix("bytes=").split("-"))
        with path.open("rb") as source:
            source.seek(start)
            body = source.read(end - start + 1)
        return httpx.Response(
            206,
            headers={"ETag": index["etag"], "Content-Range": f"bytes {start}-{end}/{index['size']}"},
            stream=httpx.ByteStream(body),
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        reader = HTTPRangeReader("https://example.test/results.pv", index, client=client)
        np.testing.assert_array_equal(reader.read_array("u_3"), arrays["u_3"])
        assert len(requests) == 1
        assert reader.bytes_received < path.stat().st_size / 4
        assert requests[0].headers["If-Match"] == index["etag"]
        assert requests[0].headers["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("failure", ["full", "etag", "range", "encoding", "length", "oversized", "truncated"])
def test_refuses_unpinned_or_full_responses(tmp_path, failure) -> None:
    """Refuse full responses, stale revisions, and incorrect range semantics."""
    path = tmp_path / "results.pv"
    pyvista_zstd.write(pv.Sphere(), path)
    pyvista_zstd.append_arrays(path, {"u": np.arange(300, dtype=float).reshape(100, 3)})
    index = build_range_index(path)

    def serve(request) -> httpx.Response:
        start, end = (int(value) for value in request.headers["Range"].removeprefix("bytes=").split("-"))
        headers = {"ETag": index["etag"], "Content-Range": f"bytes {start}-{end}/{index['size']}"}
        with path.open("rb") as source:
            source.seek(start)
            body = source.read(end - start + 1)
        if failure == "etag":
            headers["ETag"] = '"changed"'
        elif failure == "range":
            headers["Content-Range"] = "bytes 0-1/2"
        elif failure == "encoding":
            headers["Content-Encoding"] = "br"
        elif failure == "truncated":
            body = body[:-1]
        elif failure == "length":
            headers["Content-Length"] = str(len(body) + 1)
        elif failure == "oversized":
            body += b"extra"
        return httpx.Response(200 if failure == "full" else 206, headers=headers, stream=httpx.ByteStream(body))

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        reader = HTTPRangeReader("https://example.test/results.pv", index, client=client)
        with pytest.raises(ValueError, match=r"HTTP|Remote|Byte-range"):
            reader.read_array("u")
        if failure in {"full", "etag", "range", "encoding", "length"}:
            assert reader.bytes_received == 0


def test_sidecar_and_index_boundaries(tmp_path) -> None:
    """Keep the sidecar small and refuse ranges outside its pinned file."""
    path = tmp_path / "results.pv"
    sidecar = tmp_path / "results.pv.index.json"
    pyvista_zstd.write(pv.Sphere(), path)
    pyvista_zstd.append_arrays(path, {"u": np.arange(300, dtype=float).reshape(100, 3)})
    write_range_index(path, sidecar)
    index = json.loads(sidecar.read_text())
    assert list(index["arrays"]) == ["u"]
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="index"):
            HTTPRangeReader("https://example.test/data.pv", {"schema": 2}, client=client)
        index["arrays"]["u"]["start"] = -1
        reader = HTTPRangeReader("https://example.test/data.pv", index, client=client)
        with pytest.raises(ValueError, match="outside"):
            reader.read_array("u")
