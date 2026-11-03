"""CUPTI GPU activities with exclusive step/stage correlation and host spans."""
import ctypes as C
import ctypes.util
import glob
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class ActivityTrace:
    def __init__(self):
        if os.name == 'nt':
            candidates = list((Path(torch.__file__).parent / 'lib').glob('cupti64_*.dll'))
            self.lib = C.WinDLL(str(candidates[0]))
            callback = C.WINFUNCTYPE
        else:
            candidates = glob.glob(str(Path(sys.prefix) / 'lib/python*/site-packages/nvidia/cuda_cupti/lib/libcupti.so*'))
            self.lib = C.CDLL(candidates[0] if candidates else ctypes.util.find_library('cupti'))
            callback = C.CFUNCTYPE
        pointer, size, uint = C.c_void_p, C.c_size_t, C.c_uint32
        self.request_type = callback(None, C.POINTER(pointer), C.POINTER(size), C.POINTER(size))
        self.complete_type = callback(None, pointer, uint, pointer, size, size)
        self.register = self.bind('cuptiActivityRegisterCallbacks', self.request_type, self.complete_type)
        self.enable = self.bind('cuptiActivityEnable', C.c_int)
        self.disable = self.bind('cuptiActivityDisable', C.c_int)
        self.flush = self.bind('cuptiActivityFlushAll', uint)
        self.next_record = self.bind('cuptiActivityGetNextRecord', pointer, size, C.POINTER(pointer))
        self.get_dropped = self.bind('cuptiActivityGetNumDroppedRecords', pointer, uint, C.POINTER(size))
        self.push = self.bind('cuptiActivityPushExternalCorrelationId', C.c_int, C.c_uint64)
        self.pop = self.bind('cuptiActivityPopExternalCorrelationId', C.c_int, C.POINTER(C.c_uint64))
        self.buffers = {}
        self.events, self.host = [], []
        self.correlation = {}
        self.errors, self.dropped = [], []
        self.step = 0
        self.request = self.request_type(self.buffer_requested)
        self.complete = self.complete_type(self.buffer_completed)

    def bind(self, name, *types):
        function = getattr(self.lib, name)
        function.restype = C.c_int
        function.argtypes = list(types)
        return function

    def check(self, code):
        if code:
            error = C.c_char_p()
            lookup = self.bind('cuptiGetResultString', C.c_int, C.POINTER(C.c_char_p))
            lookup(code, C.byref(error))
            raise RuntimeError(f'CUPTI {code}: {error.value}')

    def buffer_requested(self, output, size, limit):
        buffer = C.create_string_buffer((8 << 20) + 8)
        address = (C.addressof(buffer) + 7) & ~7
        self.buffers[address] = buffer
        output[0], size[0], limit[0] = address, 8 << 20, 0

    def buffer_completed(self, context, stream, buffer, size, valid):
        try:
            record = C.c_void_p()
            while valid:
                code = self.next_record(buffer, valid, C.byref(record))
                if code:
                    if code != 12:
                        self.errors.append(code)
                    break
                address = record.value
                kind = C.c_uint32.from_address(address).value
                if kind == 39:
                    external_kind = C.c_uint32.from_address(address + 4).value
                    if external_kind == 3:
                        external = C.c_uint64.from_address(address + 8).value
                        correlation = C.c_uint32.from_address(address + 16).value
                        old = self.correlation.setdefault(correlation, external)
                        if old != external:
                            self.errors.append('Conflicting external correlation')
                elif kind in (1, 2, 3, 10):
                    start, end = (C.c_uint64 * 2).from_address(address + 16)
                    offset = 92 if kind in (3, 10) else 44
                    correlation = C.c_uint32.from_address(address + offset).value
                    name = (C.c_char_p.from_address(address + 104).value.decode()
                            if kind in (3, 10) else 'memcpy' if kind == 1 else 'memset')
                    self.events.append(dict(start=start, end=end, name=name, correlation=correlation))
            dropped = C.c_size_t()
            code = self.get_dropped(context, stream, C.byref(dropped))
            if code:
                self.errors.append(code)
            self.dropped.append(dropped.value)
        except Exception as exc:
            self.errors.append(str(exc))
        self.buffers.pop(buffer, None)

    def start(self):
        self.check(self.register(self.request, self.complete))
        for kind in (39, 4, 5, 1, 2, 10):
            self.check(self.enable(kind))

    def stop(self):
        torch.cuda.synchronize()
        for kind in (10, 2, 1, 5, 4, 39):
            self.check(self.disable(kind))
        self.check(self.flush(0))
        if self.errors or any(self.dropped):
            raise RuntimeError(f'Incomplete activity trace: {self.errors}, drops {self.dropped}')

    @contextmanager
    def scope(self, stage):
        external = self.step * 10 + stage
        self.check(self.push(3, external))
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - started
            popped = C.c_uint64()
            self.check(self.pop(3, C.byref(popped)))
            if popped.value != external:
                raise RuntimeError('Unbalanced trace scopes')
            self.host.append(dict(step=self.step, stage=stage, ns=elapsed))

    def finish(self):
        for event in self.events:
            external = self.correlation.get(event['correlation'])
            if external is None:
                raise RuntimeError(f'Unattributed GPU event: {event["name"]}')
            event['step'], event['stage'] = divmod(external, 10)
        return dict(events=self.events, host=self.host, dropped=sum(self.dropped))
