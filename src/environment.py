"""Execution-environment diagnostics and confirmed package installation.

Both tools in this module are harness-side, but neither guesses about the
environment: the report comes from code executed by the MCP server, and the
install runs through the server's ``install_package``. That is the environment the
agent's Python tools actually use, which is the one worth reporting.

Order of operations for an install (all enforced here, not in the model's prose):

1. validate the package argument (a name, optionally with ``==``-style pin);
2. refuse anything that would target a global interpreter;
3. ask the person at the terminal for approval (``src.approval``);
4. install through the MCP server into the environment that runs agent code;
5. verify the import with that same interpreter.
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata as importlib_metadata

from src import approval, config

# ``name`` or ``name==1.2.3`` / ``name>=1.2``. URLs, local paths, extras, wheel
# files and flags are rejected: the package name is passed to pip as a single
# argv element, and anything that starts with "-" would become an option.
_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)"
    r"((?:===|==|~=|!=|>=|<=|>|<)[A-Za-z0-9.*+_!-]+)?$"
)

# Distribution names and import names are different namespaces. Normalising
# "python-dotenv" to "python_dotenv" imports nothing, so the exceptions are
# listed explicitly and installed distributions are asked for their real
# top-level modules below. Keys are PEP 503-normalised distribution names.
_KNOWN_IMPORT_NAMES = {
    "python-dotenv": "dotenv",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "python-dateutil": "dateutil",
    "python-multipart": "multipart",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "pyserial": "serial",
    "pyzmq": "zmq",
}

# Modules worth naming in a diagnostic: the data-science stack the agent reaches
# for, plus the harness's own dependencies.
_INTERESTING_MODULES = (
    "pandas",
    "numpy",
    "matplotlib",
    "scipy",
    "openpyxl",
    "requests",
    "tiktoken",
    "mcp",
)

# Placeholders are substituted with str.replace, not str.format: this is Python
# source containing dict literals, and format() would read their braces as fields.
_INFO_SCRIPT = """
import json, os, sys
from importlib.metadata import version

found = {}
for name in __MODULES__:
    try:
        found[name] = version(name)
    except Exception:
        pass
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "version": sys.version.split()[0],
    "cwd": os.getcwd(),
    "workspace": __WORKSPACE__,
    "packages": found,
}))
"""

# Runs in the *code* interpreter after an install. It resolves the distribution
# first and asks installed metadata for its top-level modules, so verification
# does not depend on a guessed import name. Candidate names are passed only as a
# fallback for distributions that do not declare one; a failed fallback import is
# reported as unverified, never as an installation failure.
_VERIFY_SCRIPT = """
import importlib, json, re
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions

def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()

package = __PACKAGE__
candidates = __CANDIDATES__
report = {
    "package": package,
    "installed": False,
    "version": None,
    "modules": [],
    "source": "none",
    "imports": {},
}
try:
    dist = distribution(package)
except PackageNotFoundError:
    pass
else:
    report["installed"] = True
    report["version"] = dist.version
    discovered = sorted({
        module
        for module, dists in packages_distributions().items()
        if any(canonical(item) == canonical(package) for item in dists)
    })
    checked = discovered or list(candidates)
    report["modules"] = checked
    report["source"] = "metadata" if discovered else "fallback"
    for module in checked:
        try:
            importlib.import_module(module)
        except Exception as exc:
            report["imports"][module] = f"{type(exc).__name__}: {exc}"
        else:
            report["imports"][module] = None
print(json.dumps(report))
"""


def parse_requirement(raw: object) -> tuple[str, str]:
    """Validates a package request and returns ``(name, requirement)``.

    Raises ``ValueError`` with a model-readable explanation.
    """
    text = str(raw or "").strip()
    if not text:
        raise ValueError("no package name was given")
    if text.startswith("-"):
        raise ValueError(f"'{text}' looks like a pip option, not a package name")
    match = _REQUIREMENT.match(text)
    if not match:
        raise ValueError(
            f"'{text}' is not a valid requirement: use 'name' or 'name==version' "
            "(no URLs, paths, extras or shell syntax)"
        )
    return match.group(1), text


def _canonical_distribution(name: str) -> str:
    """PEP 503-normalises a distribution name for comparisons."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_modules(package: str) -> list[str]:
    """Top-level import names that installed metadata attributes to ``package``."""
    try:
        mapping = importlib_metadata.packages_distributions()
    except Exception:  # noqa: BLE001 - metadata must never block an install check
        return []
    target = _canonical_distribution(package)
    return sorted(
        module
        for module, distributions in mapping.items()
        if any(_canonical_distribution(item) == target for item in distributions)
    )


def module_names(package: str) -> list[str]:
    """Candidate import names for a distribution, most reliable first.

    Installed metadata and a small explicit table are trusted; the legacy
    ``-``/``.`` to ``_`` transformation is only the final fallback, because it is
    wrong for ``python-dotenv``, ``scikit-learn`` and similar packages.
    """
    name = package.strip()
    explicit = _KNOWN_IMPORT_NAMES.get(_canonical_distribution(name))
    discovered = _installed_modules(name)

    candidates: list[str] = []
    for candidate in (explicit, *discovered):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        candidates.append(re.sub(r"[-.]+", "_", name).lower())
    return candidates


def module_name(package: str) -> str:
    """The import name for a distribution name (``scikit-learn`` -> ``sklearn``)."""
    return module_names(package)[0]


def _parse_json_line(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def python_environment_info() -> str:
    """Reports the environment that actually executes agent code.

    The report is produced by running a short script through ``run_python_code``
    on the MCP server, so it describes that process — interpreter path, version,
    active virtualenv, working directory and installed versions — rather than the
    harness process guessing about it.
    """
    script = _INFO_SCRIPT.replace(
        "__MODULES__", repr(list(_INTERESTING_MODULES))
    ).replace("__WORKSPACE__", json.dumps(str(config.WORKSPACE_DIR)))
    try:
        from src.mcp_client import get_bridge

        output = get_bridge().call(
            "run_python_code",
            {
                "code": script,
                "execution_mode": "inline",
                "session_id": "default",
                "environment": "system",
            },
        )
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the model
        return f"Error: could not inspect the execution environment: {type(exc).__name__}: {exc}"

    report = _parse_json_line(str(output))
    if not report:
        return (
            "Error: the execution environment did not return a readable report. "
            f"Raw output:\n{output}"
        )

    in_venv = report["prefix"] != report["base_prefix"]
    packages = report.get("packages") or {}
    lines = [
        f"Python:     {report['executable']} ({report['version']})",
        (
            f"Virtualenv: {'yes — ' + report['prefix'] if in_venv else 'NO — global interpreter ' + report['prefix']}"
        ),
        f"Workspace:  {report['workspace']}  (relative tool paths resolve here)",
        f"Cwd:        {report['cwd']}",
        "Packages:   "
        + (", ".join(f"{name} {ver}" for name, ver in packages.items()) or "none of the checked packages"),
    ]
    if report["executable"] != config.MCP_PYTHON:
        # Should not happen once one environment serves the whole agent; saying so
        # is what turns a mysterious ImportError into a diagnosable mismatch.
        lines.append(
            "WARNING:    the tool interpreter differs from the harness interpreter "
            f"({config.MCP_PYTHON}). Install packages into both or restart the harness "
            "with the intended interpreter."
        )
    return "\n".join(lines)


def install_package(arguments: dict) -> str:
    """Installs one package after validating the request and asking the user.

    ``upgrade`` is refused on purpose: the agent must name the version it wants
    instead of silently replacing unrelated packages.
    """
    try:
        name, requirement = parse_requirement(
            arguments.get("package_name") or arguments.get("package")
        )
    except ValueError as exc:
        return f"Error: {exc}. Nothing was installed."

    if arguments.get("upgrade"):
        return (
            "Error: installing with upgrade is not allowed, because it can replace "
            f"unrelated packages. Ask for an explicit version instead, e.g. '{name}==<version>'."
        )

    if sys.prefix == sys.base_prefix:
        return (
            "Error: refusing to install — the harness is running on a global "
            f"interpreter ({sys.executable}), and packages must not be installed "
            "there. Run the agent from its project environment instead: "
            "python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
        )

    prompt = (
        f"[Approval] The agent wants to install '{requirement}' into the environment "
        f"that runs its code:\n    {config.MCP_PYTHON}\n"
        "This changes the environment. Approve this install?"
    )
    if not approval.request(prompt):
        return (
            f"Installation of '{requirement}' was not approved, so nothing was "
            "installed. Explain why the package is needed or find a way to finish "
            "the task without it."
        )

    try:
        from src.mcp_client import get_bridge

        bridge = get_bridge()
        result = str(
            bridge.call(
                "install_package",
                {"package_name": requirement, "environment": "system"},
            )
        )
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the model
        return f"Error: installing '{requirement}' failed: {type(exc).__name__}: {exc}"

    if "success" not in result.lower():
        return f"Error: installing '{requirement}' failed:\n{result}"

    candidates = module_names(name)
    script = _VERIFY_SCRIPT.replace("__PACKAGE__", json.dumps(name)).replace(
        "__CANDIDATES__", json.dumps(candidates)
    )
    verification = str(
        bridge.call(
            "run_python_code",
            {
                "code": script,
                "execution_mode": "inline",
                "session_id": "default",
                "environment": "system",
            },
        )
    )

    report = _parse_json_line(verification)
    if not isinstance(report, dict):
        return (
            f"Installed {requirement}, but the import check returned no readable "
            f"report:\n{verification}"
        )

    if not report.get("installed"):
        return (
            f"Installed {requirement}, but the distribution '{name}' is not visible "
            f"to the code interpreter ({config.MCP_PYTHON}). Nothing was reported as "
            "verified."
        )

    imports = report.get("imports") or {}
    working = [module for module, error in imports.items() if error is None]
    version = report.get("version") or "(unknown version)"
    if working:
        modules = ", ".join(working)
        return (
            f"Installed {requirement} {version} (approved by the user). Verified in "
            f"the same interpreter: import {modules} succeeded."
        )

    if imports and report.get("source") == "metadata":
        failures = "\n".join(
            f"  {module}: {error}" for module, error in imports.items()
        )
        return (
            f"Installed {requirement} {version}, but importing its declared module "
            f"still fails in the code interpreter:\n{failures}"
        )

    # Only a guessed fallback name was available. The distribution itself is
    # installed, so a wrong guess is not an installation failure.
    checked = ", ".join(imports) or "(no candidate)"
    return (
        f"Installed {requirement} {version} (approved by the user). The "
        f"distribution is present; its import name could not be confirmed from "
        f"installed metadata (tried: {checked}), so the import itself was not "
        "verified."
    )
