"""Regression checks for the bounded P1-025 MCP-Atlas runtime recovery."""

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "src" / "agent_environment" / "mcp_server_template.json"
MANIFEST_PATH = ROOT / "dev_scripts" / "p1_025_recovery_python_tools.json"
INSTALL_SCRIPT_PATH = ROOT / "dev_scripts" / "install_mcp_packages.sh"


def test_recovery_manifest_matches_runtime_configuration():
    manifest = json.loads(MANIFEST_PATH.read_text())
    config = json.loads(CONFIG_PATH.read_text())

    for server_id, expected in manifest["servers"].items():
        runtime = config["mcpServers"][server_id]
        assert runtime["command"] == expected["command"]
        assert runtime.get("args", []) == expected["args"]


def test_recovery_install_script_uses_exact_mcp_sdk_and_pubmed_source():
    manifest = json.loads(MANIFEST_PATH.read_text())
    install_script = INSTALL_SCRIPT_PATH.read_text()
    sdk_version = manifest["mcp_python_sdk"]
    python_version = manifest["python"]

    assert f'MCP_PYTHON_SDK_VERSION="{sdk_version}"' in install_script
    assert f'MCP_TOOL_PYTHON_VERSION="{python_version}"' in install_script
    for server_id, expected in manifest["servers"].items():
        if server_id == "pubmed":
            assert expected["source"] in install_script
            assert expected["environment"] in install_script
        else:
            assert expected["package"] in install_script
            assert (
                'uv tool install --python "$MCP_TOOL_PYTHON_VERSION" '
                '--with "$MCP_PYTHON_SDK_VERSION"'
            ) in install_script
