"""
setup_multiarch.py
==================
Build script for multiple ISA-tuned variants of the C++ pybind11 extension.

Produces one shared library per ISA tier:
    stuart_landau_simulator_avx512.<ext>   (Skylake-X, Ice Lake, …)
    stuart_landau_simulator_avx2.<ext>     (Haswell, Zen 2/3/4, …)
    stuart_landau_simulator_avx.<ext>      (Sandy Bridge, Bulldozer, …)
    stuart_landau_simulator_sse42.<ext>    (Nehalem and later – safe fallback)
    stuart_landau_simulator_baseline.<ext> (generic x86-64, ultra-safe)

At import time ``source/core/simulation_engine.py`` calls
``select_best_extension()`` which returns the fastest variant the CPU can run.

Build
-----
    pip install pybind11
    python setup_multiarch.py build_ext --inplace

Or drive all tiers via the helper script:
    bash scripts/build_multiarch.sh
"""

from __future__ import annotations

import os
#import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# ---------------------------------------------------------------------------
# ISA tier definitions  (name_suffix → (march_flag, required_cpu_flags))
# required_cpu_flags are checked at runtime in simulation_engine.py
# ---------------------------------------------------------------------------
ISA_TIERS: list[tuple[str, str, list[str]]] = [
    # suffix        march flag              CPU flags that must be present at runtime
    ("avx512",  "-march=skylake-avx512",   ["avx512f"]),
    ("avx2",    "-march=haswell",           ["avx2", "fma"]),
    ("avx",     "-march=sandybridge",       ["avx"]),
    ("sse42",   "-march=nehalem",           ["sse4_2"]),
    ("baseline","-march=x86-64",            []),          # always safe
]


# ---------------------------------------------------------------------------
# pybind11 include path
# ---------------------------------------------------------------------------
class _Pybind11Include:
    def __str__(self) -> str:
        import pybind11  # type: ignore
        return pybind11.get_include()


# ---------------------------------------------------------------------------
# Compiler-flag helpers
# ---------------------------------------------------------------------------
def _has_flag(compiler, flag: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main(int, char**) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flag])
        return True
    except Exception:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass


def _common_compile_args(march: str) -> tuple[list[str], list[str]]:
    """Return (compile_args, link_args) for a given -march flag."""
    if sys.platform == "win32":
        # Windows: no -march equivalent; use /O2 + /openmp only
        return ["/O2", "/openmp", "/std:c++17", "/fp:fast", "/EHsc"], []

    if sys.platform == "darwin":
        compile_args = ["-O3", "-std=c++17", march,
                        "-Xpreprocessor", "-fopenmp", "-ffast-math"]
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True
            ).strip()
            compile_args += [f"-I{prefix}/include"]
            link_args = ["-lomp", f"-L{prefix}/lib"]
        except (FileNotFoundError, subprocess.CalledProcessError):
            link_args = ["-lomp"]
        return compile_args, link_args

    # Linux / POSIX
    compile_args = ["-O3", "-std=c++17", march, "-fopenmp", "-ffast-math"]
    link_args    = ["-fopenmp"]
    return compile_args, link_args


# ---------------------------------------------------------------------------
# Custom build_ext
# ---------------------------------------------------------------------------
class _BuildExt(build_ext):
    _UNIX_FLAGS = {
        "darwin": ["-stdlib=libc++", "-mmacosx-version-min=10.14"],
        "other":  [],
    }

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        opts:  list[str] = []
        lopts: list[str] = []

        if ct == "unix":
            plat_flags = self._UNIX_FLAGS.get(sys.platform,
                                               self._UNIX_FLAGS["other"])
            opts  += plat_flags
            lopts += plat_flags
            ver = f'-DVERSION_INFO="{self.distribution.get_version()}"'
            opts.append(ver)
            if _has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
            ext.extra_link_args    = lopts + ext.extra_link_args

        build_ext.build_extensions(self)


# ---------------------------------------------------------------------------
# Build one Extension per ISA tier
# ---------------------------------------------------------------------------
ext_modules = []
for suffix, march, _cpu_flags in ISA_TIERS:
    compile_args, link_args = _common_compile_args(march)
    module_name = f"stuart_landau_simulator_{suffix}"

    # Inject the module name as a preprocessor definition so that
    # PYBIND11_MODULE(MODULE_NAME, m) expands to the correct PyInit_<name>
    # symbol that Python expects when importing under this filename.
    if sys.platform == "win32":
        compile_args.append(f"/DMODULE_NAME={module_name}")
    else:
        compile_args.append(f"-DMODULE_NAME={module_name}")

    ext_modules.append(
        Extension(
            module_name,
            sources=["source/core/stuart_landau_simulator.cpp"],
            include_dirs=[_Pybind11Include()],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    )

# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------
long_description = (
    Path("docs/README.md").read_text() if Path("docs/README.md").exists() else ""
)

setup(
    name="neuro-sl-simulator",
    version="2.0.0",
    author="Alejandro Aguado",
    description=(
        "C++ accelerated Stuart-Landau neural network simulator "
        "(multi-arch build)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": _BuildExt},
    packages=["source", "source.core", "source.analysis", "source.utils"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "pybind11>=2.10",
    ],
    extras_require={"dev": ["pytest>=7", "pytest-cov", "black", "ruff"]},
    zip_safe=False,
)