from __future__ import annotations

import shutil
import subprocess  # nosec B404 - local validator shells out to trusted CLI tools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INFRA_ROOT = REPO_ROOT / "infra"
PARAM_FILES = (
    "infra/main.dev.bicepparam",
    "infra/main.staging.bicepparam",
    "infra/main.prod.bicepparam",
)
PSRULE_CONFIG = REPO_ROOT / "ps-rule.yaml"
MINIMUM_MODULE_VERSION = "1.29.0"
SECURITY_BASELINE = "Azure.Pillar.Security.L1"


def resolve_command(command: str) -> str | None:
    candidates = [command]
    if sys.platform.startswith("win"):
        candidates.extend([f"{command}.cmd", f"{command}.exe"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def run_pwsh(
    pwsh_command: str,
    command: str,
    *,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(  # nosec B603 - command is built from fixed args and static scripts
        [pwsh_command, "-NoLogo", "-NoProfile", "-Command", command],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def ensure_psrule_module(pwsh_command: str) -> None:
    install_command = f"""
$ErrorActionPreference = 'Stop'
if (-not (Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue)) {{
    Register-PSRepository -Default -ErrorAction Stop
}}
$module = Get-Module -ListAvailable -Name PSRule.Rules.Azure |
    Sort-Object Version -Descending |
    Select-Object -First 1
if ($null -eq $module -or $module.Version -lt [Version]'{MINIMUM_MODULE_VERSION}') {{
    Install-Module `
        -Name PSRule.Rules.Azure `
        -Repository PSGallery `
        -Scope CurrentUser `
        -Force `
        -AllowClobber
}}
"""
    run_pwsh(pwsh_command, install_command)


def resolve_bicep_path() -> str:
    bicep_command = resolve_command("bicep")
    if bicep_command:
        return bicep_command

    az_command = resolve_command("az")
    if not az_command:
        raise SystemExit("PSRule validation requires `az` or `bicep` on PATH.")

    bicep_path = (
        Path.home()
        / ".azure"
        / "bin"
        / ("bicep.exe" if sys.platform.startswith("win") else "bicep")
    )
    if not bicep_path.exists():
        subprocess.run(  # nosec B603 - fixed local Azure CLI command
            [az_command, "bicep", "install"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if not bicep_path.exists():
        raise SystemExit(f"Unable to resolve Bicep CLI path at {bicep_path}")
    return str(bicep_path)


def main() -> None:
    if not PSRULE_CONFIG.exists():
        raise SystemExit(f"PSRule configuration not found: {PSRULE_CONFIG}")
    if not INFRA_ROOT.exists():
        raise SystemExit(f"Infrastructure directory not found: {INFRA_ROOT}")

    pwsh_command = resolve_command("pwsh") or resolve_command("powershell")
    if not pwsh_command:
        raise SystemExit(
            "PSRule validation requires PowerShell (`pwsh` or `powershell`)."
        )

    ensure_psrule_module(pwsh_command)
    bicep_path = resolve_bicep_path()

    validate_command = f"""
$ErrorActionPreference = 'Stop'
$Env:PSRULE_AZURE_BICEP_USE_AZURE_CLI = 'false'
$Env:PSRULE_AZURE_BICEP_PATH = '{Path(bicep_path).as_posix()}'
Import-Module PSRule.Rules.Azure `
    -MinimumVersion '{MINIMUM_MODULE_VERSION}' `
    -ErrorAction Stop
$inputFiles = @(
    {", ".join(f"'{path}'" for path in PARAM_FILES)}
)
Assert-PSRule `
    -InputPath $inputFiles `
    -Module 'PSRule.Rules.Azure' `
    -Baseline '{SECURITY_BASELINE}' `
    -Format File `
    -ErrorAction Stop
"""
    run_pwsh(pwsh_command, validate_command)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
