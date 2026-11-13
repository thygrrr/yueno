"""Windows GPU process memory counters, read without launching a shell per sample."""

from __future__ import annotations

import ctypes as ct
import os


class _CounterValue(ct.Structure):
    _fields_ = [("status", ct.c_uint32), ("value", ct.c_int64)]


class _CounterItem(ct.Structure):
    _fields_ = [("name", ct.c_wchar_p), ("value", _CounterValue)]


def _check(status: int) -> None:
    if status:
        raise OSError(f"Windows GPU performance counter error: 0x{status:08x}")


class ProcessGpuMemory:
    """Dedicated/shared bytes for one PID, summed across its GPU adapters."""

    def __init__(self, pid: int) -> None:
        if os.name != "nt":
            raise OSError("Windows GPU counters are unavailable on this platform")
        self._pdh = ct.WinDLL("pdh.dll")
        signatures = {
            "PdhOpenQueryW": [ct.c_wchar_p, ct.c_size_t, ct.POINTER(ct.c_void_p)],
            "PdhAddEnglishCounterW": [ct.c_void_p, ct.c_wchar_p, ct.c_size_t, ct.POINTER(ct.c_void_p)],
            "PdhCollectQueryData": [ct.c_void_p],
            "PdhGetFormattedCounterArrayW": [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32),
                                            ct.POINTER(ct.c_uint32), ct.c_void_p],
            "PdhCloseQuery": [ct.c_void_p],
        }
        for name, args in signatures.items():
            function = getattr(self._pdh, name)
            function.argtypes = args
            function.restype = ct.c_uint32
        self._prefix = f"pid_{pid}_"
        self._query = ct.c_void_p()
        _check(self._pdh.PdhOpenQueryW(None, 0, ct.byref(self._query)))
        self._counters = []
        try:
            for kind in ("Dedicated Usage", "Shared Usage"):
                counter = ct.c_void_p()
                _check(self._pdh.PdhAddEnglishCounterW(
                    self._query, rf"\GPU Process Memory(*)\{kind}", 0, ct.byref(counter)))
                self._counters.append(counter)
        except OSError:
            self.close()
            raise

    def _read(self, counter) -> int | None:
        size, count = ct.c_uint32(), ct.c_uint32()
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, 0x400, ct.byref(size), ct.byref(count), None)  # PDH_FMT_LARGE
        if status != 0x800007D2:  # PDH_MORE_DATA
            _check(status)
            return None
        buffer = ct.create_string_buffer(size.value)
        _check(self._pdh.PdhGetFormattedCounterArrayW(
            counter, 0x400, ct.byref(size), ct.byref(count), buffer))
        items = ct.cast(buffer, ct.POINTER(_CounterItem))
        values = []
        for index in range(count.value):
            item = items[index]
            if item.name and item.name.startswith(self._prefix):
                if item.value.status not in (0, 1) or item.value.value < 0:
                    return None
                values.append(item.value.value)
        return sum(values) if values else None

    def sample(self) -> tuple[int, int] | None:
        _check(self._pdh.PdhCollectQueryData(self._query))
        dedicated, shared = (self._read(counter) for counter in self._counters)
        if dedicated is None or shared is None:
            return None
        return dedicated, shared

    def close(self) -> None:
        if self._query.value:
            self._pdh.PdhCloseQuery(self._query)
            self._query.value = None
