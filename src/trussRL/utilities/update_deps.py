"""Bump pyproject lower-bound pins to the currently installed versions.

Dev-only maintenance; writes new_pyproject.toml for review instead of
editing in place.

Usage:
    uv run --extra training python -m trussRL.utilities.update_deps

The --extra training flag is needed so the training extra's packages are
installed and visible to `uv pip list`; plain `uv run` bumps only the base
and dev dependencies.
"""

import re
import subprocess
from typing import cast

import tomlkit
from packaging import version
from tomlkit.items import Array, Table


def normalize_package_name(name: str) -> str:
    """Normalize a package name for comparison across spelling variants.

    Args:
        name: package name as it appears in pyproject.toml or `uv pip list`
            output

    Returns:
        str: lowercase name with underscores replaced by hyphens, so
            `huggingface_hub` and `huggingface-hub` compare equal
    """
    return name.lower().replace("_", "-")


def bump_lower_bounds(dependencies: Array, installed: dict[str, str]) -> None:
    """Bump each dependency's >= lower bound to the installed version in place.

    Assumptions:
        1. Packages absent from the installed mapping are silently left
           unchanged, so bumping an optional extra requires that extra to be
           synced into the current venv first.
        2. Skips a bump (with a printed notice) when the installed version
           violates an existing <X upper bound rather than crossing it.
        3. Only the >=X.Y part of each requirement string is rewritten;
           environment markers, upper bounds, and tomlkit array comments are
           preserved.

    Args:
        dependencies: tomlkit array of PEP 508 requirement strings, mutated in place
        installed: mapping of normalized package name to installed version,
            as reported by `uv pip list`

    Returns:
        None
    """
    for i, dep in enumerate(dependencies):
        # Extract package name (handle various formats)
        match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
        if match:
            pkg_name = normalize_package_name(match.group(1))
            if pkg_name in installed:
                installed_version = installed[pkg_name]

                # Check if there's an upper bound constraint
                upper_bound_match = re.search(r"<([\d.]+)", dep)

                # If upper bound exists, verify installed version satisfies it
                if upper_bound_match:
                    upper_bound = upper_bound_match.group(1)
                    if version.parse(installed_version) >= version.parse(upper_bound):
                        print(
                            f"Skipping {pkg_name}: installed {installed_version} violates upper bound <{upper_bound}"
                        )
                        continue

                # Replace lower bound, preserve everything else
                new_dep = re.sub(r">=[\d.]+", f">={installed_version}", dep)
                if new_dep != dep:
                    dependencies[i] = new_dep
                    print(f"Updated {pkg_name}: {dep} -> {new_dep}")


def update_dependencies() -> None:
    """Bump all pyproject.toml lower bounds and write new_pyproject.toml.

    Processes project.dependencies, every project.optional-dependencies
    extra, and dependency-groups.dev against the versions installed in the
    current venv, then writes the result to new_pyproject.toml and formats
    it with taplo.

    Assumptions:
        1. Never edits pyproject.toml in place; the output file is a review
           artifact, not meant to be committed.
        2. Expects to run from the repo root with `uv` and `taplo` available
           on PATH.

    Args:
        None

    Returns:
        None
    """
    # Get current installed versions
    result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True)
    installed = {}
    for line in result.stdout.strip().split("\n")[2:]:  # Skip header lines
        parts = line.split()
        if len(parts) >= 2:
            installed[normalize_package_name(parts[0])] = parts[1]

    # Read pyproject.toml (preserves formatting)
    with open("pyproject.toml", "r") as f:
        data = tomlkit.load(f)

    # Update dependencies
    project = cast(Table, data["project"])
    bump_lower_bounds(cast(Array, project["dependencies"]), installed)

    # Update optional-dependency extras
    if "optional-dependencies" in project:
        extras = cast(Table, project["optional-dependencies"])
        for extra_name in extras:
            bump_lower_bounds(cast(Array, extras[extra_name]), installed)

    # Update development dependencies
    dependency_groups = cast(Table, data["dependency-groups"])
    bump_lower_bounds(cast(Array, dependency_groups["dev"]), installed)

    # Write to new file (preserves all formatting!)
    with open("new_pyproject.toml", "w") as f:
        tomlkit.dump(data, f)

    # Format the output file with taplo
    subprocess.run(["taplo", "format", "new_pyproject.toml"])

    print("\nNew file written to: new_pyproject.toml")
    print("Review it, then replace the original if it looks good.")


if __name__ == "__main__":
    update_dependencies()
