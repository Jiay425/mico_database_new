"""Download mysql:8.0 from Docker Hub and create a docker-load tar archive."""
from __future__ import annotations

import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "mysql_8_0_image.tar"
REGISTRY = "https://registry-1.docker.io/v2/library/mysql"


def fetch(url: str, headers: dict[str, str] | None = None):
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    return response


def access_token() -> str:
    return requests.get(
        "https://auth.docker.io/token",
        params={"service": "registry.docker.io", "scope": "repository:library/mysql:pull"}, timeout=30,
    ).json()["token"]


def main() -> None:
    token = access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
    }
    index = fetch(f"{REGISTRY}/manifests/8.0", headers).json()
    descriptor = next(
        item for item in index["manifests"]
        if item.get("platform", {}).get("os") == "linux" and item.get("platform", {}).get("architecture") == "amd64"
    )
    manifest = fetch(f"{REGISTRY}/manifests/{descriptor['digest']}", headers).json()
    config_digest = manifest["config"]["digest"].split(":", 1)[1]
    config = fetch(f"{REGISTRY}/blobs/{manifest['config']['digest']}", {"Authorization": f"Bearer {token}"}).content
    layers = manifest["layers"]
    print(f"Preparing mysql:8.0 image with {len(layers)} layers", flush=True)
    with tarfile.open(OUT, "w") as archive:
        info = tarfile.TarInfo(f"{config_digest}.json"); info.size = len(config)
        archive.addfile(info, io.BytesIO(config))
        names = []
        for index, layer in enumerate(layers):
            directory = f"layer{index}"
            names.append(f"{directory}/layer.tar")
            print(f"Downloading layer {index + 1}/{len(layers)}", flush=True)
            # Large layers can take longer than Docker Hub's short-lived token.
            response = fetch(f"{REGISTRY}/blobs/{layer['digest']}", {"Authorization": f"Bearer {access_token()}"})
            raw = io.BytesIO()
            with gzip.GzipFile(fileobj=response.raw) as source:
                shutil.copyfileobj(source, raw, length=1024 * 1024)
            payload = raw.getvalue()
            layer_info = tarfile.TarInfo(names[-1]); layer_info.size = len(payload)
            archive.addfile(layer_info, io.BytesIO(payload))
            version = b"1.0"
            version_info = tarfile.TarInfo(f"{directory}/VERSION"); version_info.size = len(version)
            archive.addfile(version_info, io.BytesIO(version))
        manifest_payload = json.dumps([{"Config": f"{config_digest}.json", "RepoTags": ["mysql:8.0"], "Layers": names}]).encode()
        manifest_info = tarfile.TarInfo("manifest.json"); manifest_info.size = len(manifest_payload)
        archive.addfile(manifest_info, io.BytesIO(manifest_payload))
    print(OUT, OUT.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
