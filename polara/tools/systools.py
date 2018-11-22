import ctypes
import os
import sys


# Getting memory info, based on
# http://stackoverflow.com/a/2017659/6621667 and
# https://doeidoei.wordpress.com/2009/03/22/python-tip-3-checking-available-ram-with-python/
class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

    def __init__(self):
        self.dwLength = ctypes.sizeof(self)
        super().__init__()


def platform_free_memory():
    if sys.platform == 'win32':
        memory_status = MemoryStatus()
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        mem = memory_status.ullAvailPhys / (1024**3)  # return in gigabytes
    elif sys.platform == 'darwin':
        try:
            import psutil
        except ImportError:
            print('Please, install psutil.')
        memory_status = psutil.virtual_memory
        mem = memory_status.free / (1024**3)  # return in gigabytes
    else:
        memory_status = os.popen("free -m").readlines()
        if memory_status[0].split()[2].lower() == 'free':
            mem = int(memory_status[1].split()[3]) / 1024  # return in gigabytes
        else:
            raise ValueError('Unrecognized memory info')
    return mem


def get_available_memory():
    try:
        import psutil  # preferred way
        res = psutil.virtual_memory().available
    except ImportError:
        res = platform_free_memory()
    return res
