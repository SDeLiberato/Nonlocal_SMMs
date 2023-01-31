"""Nox sessions."""

import nox

from nox_poetry import Session
from nox_poetry import session

package = "nonlocal_smms"

nox.options.sessions = "lint", "tests", "black", "ruff"
locations = "src", "tests", "noxfile.py", "docs/source"


@session(python=["3.10"])
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@session(venv_backend="conda")
def tests(session: Session) -> None:
    """Run the test suite."""
    session.conda_install("--file", "conda-osx-64.lock")
    session.install("pytest", "pytest-cov")
    session.install(".")
    session.run("pytest", "--cov", "-s", "-vv")


@session(python=["3.10"])
def black(session: Session) -> None:
    """Re-format using Black."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args, "--target-version", "py310")


@session(python=["3.10"])
def ruff(session: Session) -> None:
    """Lint using ruff: experimental for now..."""
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", *args)
