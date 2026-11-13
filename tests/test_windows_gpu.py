import ctypes as ct
from types import SimpleNamespace

import pytest

from yueno.windows_gpu import ProcessGpuMemory, _CounterItem, _CounterValue


@pytest.mark.parametrize("bad_status,expected", [(False, 300), (True, None)])
def test_counter_reader_filters_pid_and_sums_adapters(bad_status, expected):
    items = (_CounterItem * 3)(
        _CounterItem("pid_123_luid_a_phys_0", _CounterValue(0, 100)),
        _CounterItem("pid_123_luid_b_phys_0", _CounterValue(0xC0000BC6 if bad_status else 1, 200)),
        _CounterItem("pid_1234_luid_a_phys_0", _CounterValue(0, 10000)),
    )

    def read(counter, fmt, size, count, buffer):
        size._obj.value = ct.sizeof(items)
        count._obj.value = len(items)
        if buffer is None:
            return 0x800007D2
        ct.memmove(buffer, ct.addressof(items), ct.sizeof(items))
        return 0

    reader = ProcessGpuMemory.__new__(ProcessGpuMemory)
    reader._prefix = "pid_123_"
    reader._pdh = SimpleNamespace(PdhGetFormattedCounterArrayW=read)
    assert reader._read(None) == expected
