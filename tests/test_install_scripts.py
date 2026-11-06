"""Regression tests for the root bootstrap shell installers.

These scripts normally run as root from cloud-init and write to absolute paths under /etc and
/opt, which a developer account cannot reproduce. Two strips are testable without privileges:

- ``install-nginx-auth.sh`` is executed end to end against stub package-manager, nginx,
  systemctl and openssl binaries, with ``KEY_FILE``/``CONF_FILE`` redirected into a temporary
  directory. That covers the file modes and the script's own output, which is what leaked.
- ``install-llmfit.sh``/``install-drivers.sh`` need git, cargo and dpkg, so their temporary-path
  handling is asserted statically. The defect they guard against (a fixed path in world-writable
  /tmp that another local user can pre-create) has no portable unprivileged reproduction.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
NGINX_AUTH = SCRIPTS / "install-nginx-auth.sh"
FAKE_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to exercise the installers"
)


def _stub_bin(directory: Path) -> None:
    """Create no-op binaries for everything install-nginx-auth.sh calls."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("apt-get", "dnf", "yum", "nginx", "systemctl"):
        stub = directory / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    openssl = directory / "openssl"
    openssl.write_text(f"#!/bin/sh\nprintf '%s\\n' {FAKE_KEY}\n")
    openssl.chmod(0o755)


@pytest.fixture()
def rendered(tmp_path: Path) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run the real auth installer against a temporary key file and config path."""
    stubs = tmp_path / "bin"
    _stub_bin(stubs)
    key_file = tmp_path / "opt" / "api-key"
    conf_dir = tmp_path / "nginx"
    conf_dir.mkdir()
    conf_file = conf_dir / "turboinference.conf"

    env = {
        "PATH": f"{stubs}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "KEY_FILE": str(key_file),
        "CONF_FILE": str(conf_file),
        "PROXY_PORT": "8443",
        "API_PORT": "8080",
    }
    result = subprocess.run(
        ["bash", str(NGINX_AUTH)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, key_file, conf_file


class TestNginxAuthInstaller:
    def test_installer_succeeds_against_stubs(self, rendered) -> None:
        result, key_file, conf_file = rendered
        assert result.returncode == 0, result.stderr
        assert key_file.exists() and conf_file.exists()

    def test_nginx_config_is_not_world_readable(self, rendered) -> None:
        """The config embeds the API key, so it must be as restricted as the key file."""
        _, key_file, conf_file = rendered
        conf_mode = stat.S_IMODE(conf_file.stat().st_mode)
        key_mode = stat.S_IMODE(key_file.stat().st_mode)
        assert key_mode == 0o600
        assert conf_mode == 0o600
        assert conf_mode & 0o077 == 0

    def test_nginx_config_still_carries_the_key_and_directives(self, rendered) -> None:
        """The permission change must not cost the proxy its authentication or routing."""
        _, _, conf_file = rendered
        body = conf_file.read_text()
        assert f'"Bearer {FAKE_KEY}"        1;' in body
        assert "listen 8443;" in body
        assert "proxy_pass http://localhost:8080;" in body

    def test_api_key_is_never_printed(self, rendered) -> None:
        """The banner may only print the command that reads the key, never the key."""
        result, _, _ = rendered
        output = result.stdout + result.stderr
        assert FAKE_KEY not in output
        assert "Bearer $(cat " in result.stdout


class TestRootInstallerTemporaryPaths:
    @pytest.mark.parametrize(
        ("script", "fixed_paths"),
        [
            ("install-llmfit.sh", ('LLMFIT_DIR="/tmp/llmfit-build"', "/tmp/llmfit-build")),
            ("install-drivers.sh", ("-O /tmp/cuda-keyring.deb", "/tmp/cuda-keyring.deb")),
        ],
    )
    def test_root_installers_use_mktemp_paths(self, script, fixed_paths) -> None:
        """A predictable path in /tmp must not be trusted by a root build or install."""
        body = (SCRIPTS / script).read_text()
        assert "mktemp" in body
        for fixed in fixed_paths:
            assert fixed not in body
        assert "trap " in body
