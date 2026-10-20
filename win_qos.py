"""
Keep this process at High QoS on Windows 11 (opt out of power throttling / EcoQoS).

On hybrid Intel CPUs (P-cores + E-cores, e.g. i9-13900K), Windows assigns a QoS
level per process. A process without a foreground window - one started from a
background shell, an IDE task or a tool - can be given a lower QoS, and its
threads are then scheduled on the E-cores. The texture stage's CPU work
(pymeshlab remesh, xatlas UV unwrap, cv2.inpaint) is mostly single-threaded and
runs ~1.6x slower there: seam inpainting measured 9.6-12.3 s wall by default
vs 7.4-7.8 s with this opt-out (RTX 4090 / i9-13900K, 2026-10-20).

Setting PROCESS_POWER_THROTTLING_EXECUTION_SPEED in ControlMask with a zero
StateMask tells Windows never to throttle this process. It affects only the
calling process (not system settings or the power plan) and is a no-op on
other platforms. Each entry point calls it; child processes call it themselves.
"""

import sys

_PROCESS_POWER_THROTTLING = 4               # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1

_applied = None


def request_high_qos():
    """Opt the current process out of execution-speed throttling. Returns True on success."""
    global _applied
    if _applied is not None:
        return _applied
    _applied = False
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL

        state = _PowerThrottlingState(
            _PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            _PROCESS_POWER_THROTTLING_EXECUTION_SPEED,  # control execution speed throttling...
            0,                                          # ...and turn it off (High QoS)
        )
        _applied = bool(k32.SetProcessInformation(
            k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))
    except (OSError, AttributeError):
        _applied = False
    return _applied
