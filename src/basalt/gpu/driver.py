# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Talking to the GPU through the CUDA Driver API, with ctypes and nothing else.

Anything basalt measures on real silicon has to get there somehow, and the
obvious routes all cost more than they are worth here. The runtime API needs a
host compiler and a toolkit; a Python CUDA binding is a heavyweight dependency
for a project that otherwise has none.

The driver API needs neither. `nvcuda` ships with the display driver, so it is
present on any machine with a working GPU, and it loads a cubin directly, which
is exactly the artefact basalt already produces. The whole surface used here is
about a dozen entry points.

Handles are freed in reverse order of acquisition by the context manager. Leaking
a CUDA context on a desktop machine wedges the display driver, which is a poor
way to find out your cleanup was wrong.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import Any

__all__ = ["CudaError", "Device", "Module", "cuda_available", "driver"]

# CUdevice is an int; CUcontext, CUmodule, CUfunction and CUdeviceptr are all
# pointer-sized. Naming them makes the signatures below readable.
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_size_t


class CudaError(RuntimeError):
    """A driver call returned something other than CUDA_SUCCESS."""

    def __init__(self, code: int, call: str, name: str = "") -> None:
        self.code = code
        self.call = call
        detail = f" ({name})" if name else ""
        super().__init__(f"{call} failed with CUDA error {code}{detail}")


_LIB: ctypes.CDLL | None = None


def driver() -> ctypes.CDLL:
    """Load the driver library once, lazily.

    Lazily because importing basalt must not require a GPU: the ISA database,
    the prober and the verifier all run on machines with no NVIDIA hardware, and
    that property is worth protecting.
    """
    global _LIB
    if _LIB is not None:
        return _LIB

    candidates = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
    errors = []
    for name in candidates:
        try:
            # an `if` rather than a ternary so the branch for the other platform
            # is dead code a type checker can skip; `ctypes.WinDLL` does not
            # exist off Windows and a ternary makes it look like it should
            if sys.platform == "win32":
                _LIB = ctypes.WinDLL(name)
            else:
                _LIB = ctypes.CDLL(name)
            return _LIB
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise CudaError(-1, "loading the CUDA driver", "; ".join(errors))


def cuda_available() -> bool:
    """True when a driver and at least one device are present."""
    try:
        lib = driver()
        if lib.cuInit(0) != 0:
            return False
        count = ctypes.c_int()
        return lib.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except CudaError:
        return False


def _check(lib: ctypes.CDLL, code: int, call: str) -> None:
    if code == 0:
        return
    name = ctypes.c_char_p()
    try:
        lib.cuGetErrorName(code, ctypes.byref(name))
        text = name.value.decode() if name.value else ""
    except Exception:
        text = ""
    raise CudaError(code, call, text)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Enough about the part to label a measurement honestly."""

    name: str
    compute_capability: tuple[int, int]
    multiprocessors: int
    clock_khz: int
    memory_bytes: int

    @property
    def arch(self) -> str:
        major, minor = self.compute_capability
        return f"sm_{major}{minor}"

    def describe(self) -> str:
        return (
            f"{self.name} ({self.arch}, {self.multiprocessors} SMs, "
            f"{self.clock_khz / 1000:.0f} MHz nominal)"
        )


# CUdevice_attribute values used below
_ATTR_MULTIPROCESSOR_COUNT = 16
_ATTR_CLOCK_RATE = 13


class Module:
    """A loaded cubin and the kernels inside it."""

    def __init__(self, device: Device, handle: CUmodule) -> None:
        self._device = device
        # cleared on unload, so it outlives the handle it was given
        self._handle: CUmodule | None = handle
        self._lib = device._lib

    def function(self, name: str) -> CUfunction:
        fn = CUfunction()
        _check(
            self._lib,
            self._lib.cuModuleGetFunction(ctypes.byref(fn), self._handle, name.encode()),
            f"cuModuleGetFunction({name})",
        )
        return fn

    def unload(self) -> None:
        if self._handle:
            self._lib.cuModuleUnload(self._handle)
            self._handle = None


class Device:
    """One GPU, with a context, for the lifetime of a `with` block."""

    def __init__(self, ordinal: int = 0) -> None:
        self._ordinal = ordinal
        self._lib = driver()
        self._ctx: CUcontext | None = None
        self._modules: list[Module] = []
        self._allocations: list[CUdeviceptr] = []

    # ---- lifecycle -----------------------------------------------------

    def __enter__(self) -> Device:
        lib = self._lib
        _check(lib, lib.cuInit(0), "cuInit")

        dev = CUdevice()
        _check(lib, lib.cuDeviceGet(ctypes.byref(dev), self._ordinal), "cuDeviceGet")
        self._dev = dev

        ctx = CUcontext()
        # cuCtxCreate is versioned; the unsuffixed symbol is the old ABI
        create = getattr(lib, "cuCtxCreate_v2", lib.cuCtxCreate)
        _check(lib, create(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        self._ctx = ctx
        return self

    def __exit__(self, *exc: object) -> None:
        lib = self._lib
        for ptr in reversed(self._allocations):
            getattr(lib, "cuMemFree_v2", lib.cuMemFree)(ptr)
        self._allocations.clear()
        for mod in reversed(self._modules):
            mod.unload()
        self._modules.clear()
        if self._ctx:
            getattr(lib, "cuCtxDestroy_v2", lib.cuCtxDestroy)(self._ctx)
            self._ctx = None

    # ---- properties ----------------------------------------------------

    @property
    def info(self) -> DeviceInfo:
        lib = self._lib
        name = ctypes.create_string_buffer(256)
        _check(lib, lib.cuDeviceGetName(name, 256, self._dev), "cuDeviceGetName")

        major, minor = ctypes.c_int(), ctypes.c_int()
        _check(
            lib,
            lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), self._dev),
            "cuDeviceComputeCapability",
        )

        def attr(which: int) -> int:
            value = ctypes.c_int()
            _check(
                lib,
                lib.cuDeviceGetAttribute(ctypes.byref(value), which, self._dev),
                f"cuDeviceGetAttribute({which})",
            )
            return value.value

        total = ctypes.c_size_t()
        getattr(lib, "cuDeviceTotalMem_v2", lib.cuDeviceTotalMem)(ctypes.byref(total), self._dev)

        return DeviceInfo(
            name=name.value.decode(),
            compute_capability=(major.value, minor.value),
            multiprocessors=attr(_ATTR_MULTIPROCESSOR_COUNT),
            clock_khz=attr(_ATTR_CLOCK_RATE),
            memory_bytes=total.value,
        )

    # ---- memory --------------------------------------------------------

    def alloc(self, size: int) -> int:
        """Allocate device memory, returning a plain integer address.

        Handing back the ctypes object instead makes every call site wrap it
        again, and `c_size_t(c_size_t(...))` is a TypeError rather than a
        no-op, so the plain int is the friendlier contract.
        """
        lib = self._lib
        ptr = CUdeviceptr()
        alloc = getattr(lib, "cuMemAlloc_v2", lib.cuMemAlloc)
        _check(lib, alloc(ctypes.byref(ptr), ctypes.c_size_t(size)), "cuMemAlloc")
        self._allocations.append(ptr)
        return int(ptr.value)

    def upload(self, ptr: int, data: bytes) -> None:
        lib = self._lib
        copy = getattr(lib, "cuMemcpyHtoD_v2", lib.cuMemcpyHtoD)
        _check(
            lib,
            copy(CUdeviceptr(ptr), data, ctypes.c_size_t(len(data))),
            "cuMemcpyHtoD",
        )

    def download(self, ptr: int, size: int) -> bytes:
        lib = self._lib
        buf = ctypes.create_string_buffer(size)
        copy = getattr(lib, "cuMemcpyDtoH_v2", lib.cuMemcpyDtoH)
        _check(
            lib,
            copy(buf, CUdeviceptr(ptr), ctypes.c_size_t(size)),
            "cuMemcpyDtoH",
        )
        return buf.raw

    # ---- execution -----------------------------------------------------

    def load_cubin(self, cubin: bytes) -> Module:
        lib = self._lib
        mod = CUmodule()
        _check(lib, lib.cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
        module = Module(self, mod)
        self._modules.append(module)
        return module

    def launch(
        self,
        fn: CUfunction,
        params: list[Any],
        *,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        shared_bytes: int = 0,
    ) -> None:
        """Launch a kernel and wait for it.

        `params` holds ctypes objects; the driver wants an array of pointers to
        the arguments rather than the arguments themselves.
        """
        lib = self._lib
        arg_ptrs = (ctypes.c_void_p * len(params))(
            *[ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in params]
        )
        _check(
            lib,
            lib.cuLaunchKernel(
                fn,
                ctypes.c_uint(grid[0]),
                ctypes.c_uint(grid[1]),
                ctypes.c_uint(grid[2]),
                ctypes.c_uint(block[0]),
                ctypes.c_uint(block[1]),
                ctypes.c_uint(block[2]),
                ctypes.c_uint(shared_bytes),
                None,
                arg_ptrs,
                None,
            ),
            "cuLaunchKernel",
        )
        _check(lib, lib.cuCtxSynchronize(), "cuCtxSynchronize")
