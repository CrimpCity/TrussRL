"""Bump pyproject lower-bound pins to the currently installed versions.

Dev-only maintenance; writes new_pyproject.toml for review instead of
editing in place.

Usage:
    python -m src.trussRL.utilities.update_deps
"""

import re
import subprocess
from typing import cast

import tomlkit
from packaging import version
from tomlkit.items import Array, Table


def update_dependencies() -> None:  # Get current installed versions
    result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True)
    installed = {}
    for line in result.stdout.strip().split("\n")[2:]:  # Skip header lines
        parts = line.split()
        if len(parts) >= 2:
            installed[parts[0].lower()] = parts[1]

    # Read pyproject.toml (preserves formatting)
    with open("pyproject.toml", "r") as f:
        data = tomlkit.load(f)

    # Update dependencies
    project = cast(Table, data["project"])
    dependencies = cast(Array, project["dependencies"])

    for i, dep in enumerate(dependencies):
        # Extract package name (handle various formats)
        match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
        if match:
            pkg_name = match.group(1).lower()
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

    # Update development dependencies
    dependency_groups = cast(Table, data["dependency-groups"])
    dependencies = cast(Array, dependency_groups["dev"])

    for i, dep in enumerate(dependencies):
        # Extract package name (handle various formats)
        match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
        if match:
            pkg_name = match.group(1).lower()
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

    # Write to new file (preserves all formatting!)
    with open("new_pyproject.toml", "w") as f:
        tomlkit.dump(data, f)

    # Format the output file with taplo
    subprocess.run(["taplo", "format", "new_pyproject.toml"])

    print("\nNew file written to: new_pyproject.toml")
    print("Review it, then replace the original if it looks good.")


if __name__ == "__main__":
    update_dependencies()
