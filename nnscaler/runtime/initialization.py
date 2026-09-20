"""Capture initializer specifications and initialize local parameter shards on GPU.

Random weights follow native CUDA distributions. Values intentionally differ from
legacy full-model initialization. Identical replicas use the same logical region
and private seed; overlapping layouts are refined into common rectangular tiles.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path
import hashlib
import math

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

PLAN_FILE = "init_plan.pt"
PLAN_VERSION = 4
SCHEME = "local_cuda_native_chunked_v3"
# Keep parameter seeds stable when only the sampling/storage implementation changes.
_SEED_DOMAIN = "local_cuda_direct_v2"
_active_capture = ContextVar("nnscaler_initialization_capture", default=None)
_checkpoint_readers = ContextVar("nnscaler_initialization_readers", default=None)
_checkpoint_sources = ContextVar("nnscaler_initialization_sources", default=None)


def is_metadata_capture():
    recorder = _active_capture.get()
    return recorder is not None and recorder.metadata_only

@contextmanager
def checkpoint_sources(paths, *, aliases=None):
    """Bind portable source names to this process's checkpoint paths.

    Bindings are runtime-only. Plans contain source names and content digests;
    compiler cache metadata uses the same names instead of node-local paths.
    ``aliases`` identifies equivalent paths in model arguments (e.g. a directory
    option that resolves to a particular safetensors file).
    """
    files, substitutions = {}, {}
    for name, path in paths.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint source names must be nonempty strings")
        resolved = Path(path).expanduser().resolve()
        if any(entry["path"] == resolved for entry in files.values()):
            raise ValueError(f"Duplicate checkpoint source path: {resolved}")
        files[name] = dict(path=resolved, signature=None, sha256=None)
        for alias in (path, resolved, *(aliases or {}).get(name, ())):
            for value in (str(alias), str(Path(alias).expanduser().resolve())):
                if value in substitutions and substitutions[value] != name:
                    raise ValueError(f"Ambiguous checkpoint path alias: {value}")
                substitutions[value] = name
    token = _checkpoint_sources.set((files, substitutions))
    try:
        yield
    finally:
        _checkpoint_sources.reset(token)


def portable_checkpoint_arguments(value):
    """Normalize registered paths in cache metadata, leaving actual args intact."""
    bindings = _checkpoint_sources.get()
    if bindings is None:
        return value
    if isinstance(value, (str, Path)) and str(value) in bindings[1]:
        return {"__nnscaler_checkpoint_source__": bindings[1][str(value)]}
    if isinstance(value, dict):
        return {key: portable_checkpoint_arguments(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(portable_checkpoint_arguments(item) for item in value)
    return value


def _checkpoint_binding(name):
    bindings = _checkpoint_sources.get()
    if bindings is None or name not in bindings[0]:
        raise ValueError(f"Bind initialization checkpoint source {name!r} using checkpoint_sources()")
    return bindings[0][name]


def _checkpoint_fingerprint(binding):
    """Hash once per unchanged local file in this scope; never serialize mtime."""
    path = binding["path"]

    def signature():
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    before = signature()
    if binding["signature"] != before:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if signature() != before:
            raise RuntimeError(f"Initialization checkpoint changed while reading: {path}")
        binding.update(signature=before, sha256=digest.hexdigest())
    return before[2], binding["sha256"]


def dtype_name(dtype):
    return str(dtype).split(".")[-1]


def compact_rules(records, calls):
    """Merge equal per-head/expert distributions while preserving exact aliases."""
    fingerprints = {}
    recipe_users = defaultdict(set)
    for name, record in records.items():
        segments = record.get("segments", [])
        fingerprint = (
            record["shape"],
            record["dtype"],
            tuple(
                (id(s["recipe"]), s["begin"], s["end"], s["source"], tuple(s["casts"]))
                for s in segments
            ),
        )
        key = fingerprints.setdefault(fingerprint, name) if segments else name
        record["rng_key"] = key
        for segment in segments:
            if segment["recipe"]["kind"] in ("uniform", "normal"):
                recipe_users[id(segment["recipe"])].add(key)
    if any(len(users) > 1 for users in recipe_users.values()):
        raise NotImplementedError(
            "Partial or reshaped copies of random-initialized parameters need an explicit initialization specification"
        )
    for record in records.values():
        segments = record.get("segments", [])
        if not segments:
            continue
        first = segments[0]
        signature = lambda s: (
            s["recipe"]["kind"],
            s["recipe"].get("a"),
            s["recipe"].get("b"),
            s["recipe"].get("seed"),
            s["recipe"].get("dtype"),
            tuple(s["casts"]),
        )
        if first["recipe"]["kind"] in ("uniform", "normal") and all(
            signature(s) == signature(first) for s in segments
        ):
            record["segments"] = [
                dict(first, begin=0, end=math.prod(record["shape"]), source=0)
            ]
        for segment in record["segments"]:
            segment["recipe"].pop("id", None)
    return dict(
        version=PLAN_VERSION, scheme=SCHEME, records=records, initialization_calls=calls
    )


class Capture(TorchDispatchMode):
    def __init__(self, *, metadata_only=False):
        self.storages = {}
        self.calls = 0
        self.metadata_only = metadata_only
        self.devices = {}

    def logical_device(self, tensor):
        if not tensor.is_meta:
            return tensor.device
        entry = self.devices.get(tensor.untyped_storage()._cdata)
        return entry[1] if entry is not None and not entry[0].expired() else torch.device('cpu')

    def __enter__(self):
        self._token = _active_capture.set(self)
        return super().__enter__()

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            _active_capture.reset(self._token)

    def segments(self, tensor):
        key = tensor.untyped_storage()._cdata
        entry = self.storages.get(key)
        if entry is None or entry[0].expired():
            return []
        return entry[1]

    def write(self, tensor, segments):
        if not tensor.is_contiguous():
            raise NotImplementedError("Non-contiguous initializer write")
        begin = tensor.storage_offset()
        end = begin + tensor.numel()
        kept = []
        for seg in self.segments(tensor):
            if seg["end"] <= begin or seg["begin"] >= end:
                kept.append(seg)
                continue
            if seg["begin"] < begin:
                kept.append(dict(seg, end=begin))
            if seg["end"] > end:
                kept.append(
                    dict(seg, begin=end, source=seg["source"] + end - seg["begin"])
                )
        kept.extend(segments)
        self.storages[tensor.untyped_storage()._cdata] = (
            StorageWeakRef(tensor.untyped_storage()),
            sorted(kept, key=lambda s: s["begin"]),
        )

    def copy_segments(self, source, destination):
        if not source.is_contiguous() or source.numel() != destination.numel():
            return []
        start = source.storage_offset()
        stop = start + source.numel()
        shift = destination.storage_offset() - start
        result = []
        for seg in self.segments(source):
            begin = max(start, seg["begin"])
            end = min(stop, seg["end"])
            if begin < end:
                casts = seg["casts"]
                if destination.dtype != source.dtype:
                    casts = casts + [dtype_name(destination.dtype)]
                result.append(
                    dict(
                        seg,
                        begin=begin + shift,
                        end=end + shift,
                        source=seg["source"] + begin - seg["begin"],
                        casts=casts,
                    )
                )
        return result

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        original_device = kwargs.get('device')
        if original_device is None or torch.device(original_device).type == 'meta':
            original_device = next((self.logical_device(t) for t in args if isinstance(t, torch.Tensor)), torch.device('cpu'))
        original_device = torch.device(original_device)
        if self.metadata_only and name.startswith((
            "aten.rand.", "aten.randn.", "aten.randint.", "aten.randperm.",
            "aten.bernoulli", "aten.multinomial",
        )):
            raise NotImplementedError(f"Data-dependent random constructor operation: {name}")
        # Empty parameter storage has no values to preserve. Other factories
        # (arange, sin/cos, etc.) remain real so data-dependent buffers and
        # constructor metadata retain their original values.
        if self.metadata_only and name in (
            "aten.empty.memory_format", "aten.empty_strided.default",
        ):
            kwargs = dict(kwargs, device=torch.device("meta"), pin_memory=False)
        elif (self.metadata_only and name == 'aten._to_copy.default' and args[0].is_meta):
            kwargs = dict(kwargs, device=torch.device('meta'))
        recipe = None
        if name in ("aten.uniform_.default", "aten.normal_.default"):
            t = args[0]
            if not t.is_contiguous() or not t.is_floating_point():
                raise NotImplementedError(
                    f"Unsupported initializer source: {name} {t.shape} {t.dtype}"
                )
            kind = "uniform" if "uniform" in name else "normal"
            generator = kwargs.get("generator")
            if generator is None:
                device = self.logical_device(t)
                generator = (
                    torch.default_generator
                    if device.type == 'cpu'
                    else torch.cuda.default_generators[
                        device.index if device.index is not None else torch.cuda.current_device()]
                )
            a = (
                args[1]
                if len(args) > 1
                else kwargs.get("from" if kind == "uniform" else "mean", 0.0)
            )
            b = (
                args[2]
                if len(args) > 2
                else kwargs.get("to" if kind == "uniform" else "std", 1.0)
            )
            recipe = {
                "kind": kind,
                "a": a,
                "b": b,
                "seed": generator.initial_seed(),
                "id": self.calls,
                "dtype": dtype_name(t.dtype),
            }
            self.calls += 1
        if (self.metadata_only and name == "aten.copy_.default"
                and args[1].is_meta and not args[0].is_meta):
            segments = self.copy_segments(args[1], args[0])
            if len(segments) != 1 or segments[0]["recipe"]["kind"] != "checkpoint":
                raise NotImplementedError("A real constructor tensor depends on deferred random values")
            # Keep small real checkpoint-backed buffers/constant parameters
            # valid for constructor control flow, rather than copying no data.
            from torch.utils._python_dispatch import _disable_current_modes
            with _disable_current_modes():
                segment = segments[0]
                source = read_checkpoint_slice(segment["recipe"], list(args[1].shape),
                                               list(args[1].stride()), segment["source"])
                out = func(args[0], source, **kwargs)
        else:
            out = func(*args, **kwargs)
        if self.metadata_only and isinstance(out, torch.Tensor) and out.is_meta:
            self.devices[out.untyped_storage()._cdata] = (StorageWeakRef(out.untyped_storage()), original_device)
        if recipe is not None:
            begin = out.storage_offset()
            self.write(
                out,
                [
                    {
                        "begin": begin,
                        "end": begin + out.numel(),
                        "source": 0,
                        "recipe": recipe,
                        "casts": [],
                    }
                ],
            )
        elif name in (
            "aten.copy_.default",
            "aten._to_copy.default",
            "aten.clone.default",
        ):
            source = args[1] if name == "aten.copy_.default" else args[0]
            self.write(out, self.copy_segments(source, out))
        elif name in (
            "aten.ones.default",
            "aten.zeros.default",
            "aten.fill_.Scalar",
            "aten.zero_.default",
        ):
            value = (
                args[1]
                if name == "aten.fill_.Scalar"
                else (1 if "ones." in name else 0)
            )
            begin = out.storage_offset()
            self.write(
                out,
                [
                    {
                        "begin": begin,
                        "end": begin + out.numel(),
                        "source": 0,
                        "recipe": {
                            "kind": "constant",
                            "value": value,
                            "dtype": dtype_name(out.dtype),
                        },
                        "casts": [],
                    }
                ],
            )
        elif func._schema.is_mutable:
            # Also cover foreach lists and keyword out= tensors: an unknown
            # mutation must not leave an earlier initialization rule valid.
            from torch.utils._pytree import tree_leaves

            for index, argument in enumerate(func._schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                for tensor in tree_leaves(value):
                    if isinstance(tensor, torch.Tensor) and self.segments(tensor):
                        self.write(tensor, [])
        return out

    def export(self, model, buffer_limit=64 * 1024**2):
        params = dict(model.named_parameters(remove_duplicate=False))
        tensors = {**params, **dict(model.named_buffers(remove_duplicate=False))}
        records = {}
        unknown = []
        buffer_bytes = 0
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise NotImplementedError(f"Noncontiguous attribute {name}")
            segs = self.copy_segments(tensor, tensor)
            covered = sum(s["end"] - s["begin"] for s in segs)
            if covered != tensor.numel():
                if name in params:
                    unknown.append((name, tuple(tensor.shape), covered, tensor.numel()))
                    continue
                buffer_bytes += tensor.numel() * tensor.element_size()
                if buffer_bytes > buffer_limit:
                    raise ValueError(
                        "Non-random buffer payload exceeds configured budget"
                    )
                records[name] = {
                    "shape": tuple(tensor.shape),
                    "dtype": dtype_name(tensor.dtype),
                    "buffer": tensor.detach().cpu().clone(),
                }
                continue
            segs = [
                dict(
                    s,
                    begin=s["begin"] - tensor.storage_offset(),
                    end=s["end"] - tensor.storage_offset(),
                )
                for s in segs
            ]
            records[name] = {
                "shape": tuple(tensor.shape),
                "dtype": dtype_name(tensor.dtype),
                "segments": segs,
            }
        if unknown:
            raise ValueError(f"Untracked parameters: {unknown}")
        return compact_rules(records, self.calls)


def rectangles(shape, start, stop):
    if not shape:
        if start == 0 and stop == 1:
            yield [], []
        return
    strides = []
    p = 1
    for n in reversed(shape):
        strides.insert(0, p)
        p *= n
    while start < stop:
        coord = [(start // st) % n for n, st in zip(shape, strides)]
        for dim, st in enumerate(strides):
            if start % st == 0 and stop - start >= st:
                count = min((stop - start) // st, shape[dim] - coord[dim])
                lower = coord[:dim] + [coord[dim]] + [0] * (len(shape) - dim - 1)
                sizes = [1] * dim + [count] + list(shape[dim + 1 :])
                yield lower, sizes
                start += count * st
                break


def geometry_runs(sizes, strides, base):
    import itertools

    if not all(sizes):
        return
    dense = 1
    first = len(sizes)
    while first and strides[first - 1] == dense:
        first -= 1
        dense *= sizes[first]
    dst = 0
    for coord in itertools.product(*(range(n) for n in sizes[:first])):
        start = base + sum(c * s for c, s in zip(coord, strides))
        yield start, dense, dst
        dst += dense


def record_checkpoint(path, tensors, prefix=""):
    """Attach existing safetensors provenance during the compiler's construction only."""
    recorder = _active_capture.get()
    if recorder is None:
        return
    path = Path(path).resolve()
    bindings = _checkpoint_sources.get()
    names = [] if bindings is None else [
        name for name, entry in bindings[0].items() if entry["path"] == path
    ]
    if not names:
        raise ValueError(f"Register checkpoint {path} with checkpoint_sources() before capturing initialization")
    source = names[0]
    file_size, sha256 = _checkpoint_fingerprint(_checkpoint_binding(source))
    for name, tensor in tensors.items():
        begin = tensor.storage_offset()
        recipe = {
            "kind": "checkpoint",
            "checkpoint_source": source,
            "key": prefix + name,
            "shape": tuple(tensor.shape),
            "total": tensor.numel(),
            "file_size": file_size,
            "sha256": sha256,
        }
        recorder.write(
            tensor,
            [
                {
                    "begin": begin,
                    "end": begin + tensor.numel(),
                    "source": 0,
                    "recipe": recipe,
                    "casts": [],
                }
            ],
        )


def load_initialization_checkpoint(path, prefix=""):
    """Read checkpoint metadata only during deferred compiler construction."""
    from safetensors import safe_open
    from safetensors.torch import _getdtype

    recorder = _active_capture.get()
    metadata_only = recorder is not None and recorder.metadata_only
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {}
        for key in handle.keys():
            if not key.startswith(prefix):
                continue
            if metadata_only:
                view = handle.get_slice(key)
                tensor = torch.empty(view.get_shape(), dtype=_getdtype(view.get_dtype()), device="meta")
            else:
                tensor = handle.get_tensor(key)
            tensors[key[len(prefix):]] = tensor
    record_checkpoint(path, tensors, prefix)
    return tensors


@contextmanager
def checkpoint_reader(recipe):
    from safetensors import safe_open

    source = recipe["checkpoint_source"]
    binding = _checkpoint_binding(source)
    path = binding["path"]
    cache = _checkpoint_readers.get()
    cache_key = (source, path, recipe["file_size"], recipe["sha256"])
    if cache is not None and cache_key in cache[1]:
        yield cache[1][cache_key]
        return
    if _checkpoint_fingerprint(binding) != (recipe["file_size"], recipe["sha256"]):
        raise RuntimeError(f"Initialization checkpoint content changed for source {source!r}: {path}")
    if cache is not None:
        handle = cache[0].enter_context(
            safe_open(path, framework="pt", device="cpu")
        )
        cache[1][cache_key] = handle
        yield handle
    else:
        with safe_open(path, framework="pt", device="cpu") as handle:
            yield handle


def read_checkpoint_slice(recipe, sizes, strides, base):
    source_shape = recipe["shape"]
    source_strides = []
    p = 1
    for n in reversed(source_shape):
        source_strides.insert(0, p)
        p *= n
    # safetensors creates tensors through PyTorch; override an enclosing GPU
    # allocation context so checkpoint slices stay on CPU until the staged copy.
    with torch.device("cpu"), checkpoint_reader(recipe) as handle:
        source = handle.get_slice(recipe["key"])
        if len(sizes) == len(source_shape) and list(strides) == source_strides:
            starts = [(base // st) % n for st, n in zip(source_strides, source_shape)]
            if all(
                i + n <= extent for i, n, extent in zip(starts, sizes, source_shape)
            ):
                return source[tuple(slice(i, i + n) for i, n in zip(starts, sizes))]
        parts = []
        for start, length, _ in geometry_runs(sizes, strides, base):
            for lower, extent in rectangles(source_shape, start, start + length):
                parts.append(
                    source[
                        tuple(slice(i, i + n) for i, n in zip(lower, extent))
                    ].reshape(-1)
                )
        return torch.cat(parts).reshape(sizes)


def normalized_slices(shape, slices):
    if len(shape) != len(slices):
        raise ValueError("Initialization slice rank does not match parameter shape")
    bounds = [s.indices(n) for s, n in zip(slices, shape)]
    if any(step != 1 for _, _, step in bounds):
        raise NotImplementedError(
            "Local GPU initialization requires rectangular, unit-stride shards"
        )
    return tuple((start, stop) for start, stop, _ in bounds)


def partition_cuts(plan, fullmaps):
    """Common tiles make replicated and overlapping layouts agree on values."""
    cuts = {}
    for mapping in fullmaps:
        for meta in mapping.values():
            record = plan["records"][meta.orig_name]
            key = record.get("rng_key", meta.orig_name)
            shape = tuple(record["shape"])
            if tuple(meta.shape) != shape:
                raise ValueError(f"Initialization shape mismatch for {meta.orig_name}")
            axes = cuts.setdefault(key, [{0, n} for n in shape])
            for points, bounds in zip(axes, normalized_slices(shape, meta.slicers)):
                points.update(bounds)
    return {key: [sorted(axis) for axis in axes] for key, axes in cuts.items()}


def initialization_pieces(record, slices, cuts=None):
    shape = record["shape"]
    local = normalized_slices(shape, slices)
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    for segment in record["segments"]:
        for lower, extent in rectangles(shape, segment["begin"], segment["end"]):
            axes = []
            for dim, (start, stop) in enumerate(local):
                start = max(start, lower[dim])
                stop = min(stop, lower[dim] + extent[dim])
                if start >= stop:
                    break
                points = [start]
                if cuts is not None and segment["recipe"]["kind"] in (
                    "normal",
                    "uniform",
                ):
                    points.extend(x for x in cuts[dim] if start < x < stop)
                points.append(stop)
                axes.append(list(zip(points, points[1:])))
            else:
                for bounds in product(*axes):
                    destination = tuple(
                        slice(a - local[i][0], b - local[i][0])
                        for i, (a, b) in enumerate(bounds)
                    )
                    sizes = [b - a for a, b in bounds]
                    base = sum(a * stride for (a, _), stride in zip(bounds, strides))
                    base += segment["source"] - segment["begin"]
                    yield segment, destination, sizes, strides, base, bounds


def region_seed(record, segment, bounds):
    recipe = segment["recipe"]
    identity = (
        _SEED_DOMAIN,
        recipe["seed"],
        record["rng_key"],
        segment["begin"],
        segment["end"],
        tuple(bounds),
    )
    return int.from_bytes(
        hashlib.sha256(repr(identity).encode()).digest()[:8], "little"
    )


@lru_cache(maxsize=1)
def _strided_chunk_copy_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def copy(source, target, begin, count, shape: tl.constexpr,
             strides: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        remaining = begin + lane
        offset = tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(shape) - 1, -1, -1):
            offset += (remaining % shape[axis]) * strides[axis]
            remaining //= shape[axis]
        value = tl.load(source + lane, mask=lane < count, other=0)
        tl.store(target + offset, value, mask=lane < count)

    return copy


@lru_cache(maxsize=None)
def _native_random_wave(device_index):
    # ATen CUDA DistributionTemplates: 256 threads/block, float4 draws, and
    # maxThreadsPerMultiProcessor / 256 resident blocks per SM. One complete
    # wave preserves both curand thread subsequences and Philox offsets.
    props = torch.cuda.get_device_properties(device_index)
    threads = props.multi_processor_count * (props.max_threads_per_multi_processor // 256) * 256
    return threads * 4


def _native_random_chunks(target, recipe, casts, generator):
    """Match native contiguous FP32 draws, with one wave of staging storage.

    Pad the final draw to a full wave: drawing just its tail would change
    ATen's launch grid and hence its random sequence. Large FP32 tensors also
    split at TensorIterator's 32-bit byte-offset limit; preserve the parent
    Philox reservation before recursively generating their two halves.
    """
    wave = _native_random_wave(target.device.index)
    buffer = torch.empty(min(target.numel(), wave), device=target.device, dtype=torch.float32)
    flat = target.view(-1) if target.is_contiguous() else None

    # An explicit stack avoids a recursive closure retaining the GPU staging
    # buffer and target until Python's cyclic garbage collector runs.
    pending = [(0, target.numel())]
    with torch.cuda.device(target.device):
        while pending:
            begin, count = pending.pop()
            if 4 * (count - 1) > 2**31 - 1 or count > 2**31 - 1:
                generator.set_offset(generator.get_offset() + ((count - 1) // wave + 1) * 4)
                middle = count // 2
                pending.append((begin + middle, count - middle))
                pending.append((begin, middle))
                continue
            chunk = buffer[:min(buffer.numel(), count)]
            for start in range(0, count, chunk.numel()):
                if recipe['kind'] == 'normal':
                    chunk.normal_(recipe['a'], recipe['b'], generator=generator)
                else:
                    chunk.uniform_(recipe['a'], recipe['b'], generator=generator)
                value = chunk
                for cast in casts[:-1]:
                    value = value.to(getattr(torch, cast))
                size = min(chunk.numel(), count - start)
                if flat is not None:
                    flat[begin + start:begin + start + size].copy_(value[:size])
                else:
                    _strided_chunk_copy_kernel()[((size + 1023) // 1024,)](
                        value, target, begin + start, size, tuple(target.shape),
                        tuple(target.stride()), BLOCK=1024,
                    )


@torch.no_grad()
def initialize(plan, name, slices, *, out, cuts=None, generator=None, _trace=False):
    record = plan["records"][name]
    bounds = normalized_slices(record["shape"], slices)
    if tuple(out.shape) != tuple(b - a for a, b in bounds):
        raise ValueError(f"Initialization target shape mismatch for {name}")
    if not out.numel():
        return out
    if "buffer" in record:
        out.copy_(record["buffer"][slices])
        return out
    covered = 0
    for segment, dst, sizes, strides, base, region in initialization_pieces(
        record, slices, cuts
    ):
        recipe = segment["recipe"]
        target = out[dst]
        covered += target.numel()
        if recipe["kind"] in ("normal", "uniform"):
            if target.device.type != "cuda" and not (_trace and target.device.type == "cpu"):
                raise ValueError(
                    "shard_init requires generated parameter storage on CUDA"
                )
            if generator is None:
                generator = torch.Generator(device=target.device)
            generator.manual_seed(region_seed(record, segment, region))
            # Canonical contiguous draws avoid stride/alignment-dependent CUDA
            # iterator paths when the same region occurs in different layouts.
            draw_dtype = getattr(torch, recipe["dtype"])
            direct = (
                target.dtype == draw_dtype
                and all(getattr(torch, cast) == draw_dtype for cast in segment["casts"])
                and target.is_contiguous()
                and target.storage_offset() == 0
            )
            if target.is_cuda and draw_dtype == torch.float32 and not direct:
                _native_random_chunks(target, recipe, segment["casts"], generator)
                continue
            value = (
                target
                if direct
                else torch.empty(sizes, device=target.device, dtype=draw_dtype)
            )
            if recipe["kind"] == "normal":
                value.view(-1).normal_(recipe["a"], recipe["b"], generator=generator)
            else:
                value.view(-1).uniform_(recipe["a"], recipe["b"], generator=generator)
            if not direct:
                for cast in segment["casts"][:-1]:
                    value = value.to(getattr(torch, cast))
                target.copy_(value)
            # Release this region before allocating the next one. Shared full
            # weights may cover several canonical tiles; keeping the previous
            # draw alive would needlessly double the FP32 staging peak.
            del value
        elif recipe["kind"] == "constant":
            value = torch.tensor(
                recipe["value"], dtype=getattr(torch, recipe["dtype"]), device="cpu"
            )
            for cast in segment["casts"]:
                value = value.to(getattr(torch, cast))
            target.fill_(value.to(target.dtype).item())
        elif recipe["kind"] == "checkpoint":
            value = read_checkpoint_slice(recipe, sizes, strides, base)
            for cast in segment["casts"][:-1]:
                value = value.to(getattr(torch, cast))
            if target.device.type == "cuda":
                stage = torch.empty(
                    sizes, dtype=target.dtype, device="cpu", pin_memory=True
                )
                stage.copy_(value)
                target.copy_(stage, non_blocking=True)
            else:
                target.copy_(value)
        else:
            raise ValueError(f"Unsupported initializer {recipe['kind']}")
    if covered != out.numel():
        raise ValueError(
            f"Incomplete initialization for {name}: {covered}/{out.numel()}"
        )
    return out


def construct(factory, dtype, outdir):
    with Capture(metadata_only=True) as recorder:
        module = factory()
        if dtype is not None:
            module = module.to(dtype=dtype)
    plan = recorder.export(module)
    torch.save(plan, Path(outdir) / PLAN_FILE, pickle_protocol=4)
    module._nnscaler_init_plan = plan
    return module


def save_trace_attributes(outdir, attrs):
    path = Path(outdir) / PLAN_FILE
    plan = torch.load(path, weights_only=False, map_location="cpu")
    if plan.get("version") != PLAN_VERSION:
        raise RuntimeError(
            "Regenerate initialization plans for native local GPU initialization"
        )
    records = {}
    for tensor, (name, value) in attrs.items():
        record = plan["records"].get(name)
        if record is None:
            if tensor.is_param():
                raise ValueError(
                    f"Missing initialization specification for parameter {name}"
                )
            if value.is_meta:
                raise NotImplementedError(f"Traced buffer depends on deferred parameter values: {name}")
            if value.numel() * value.element_size() > 64 * 1024**2:
                raise ValueError(
                    f"Traced buffer {name} exceeds initialization payload budget"
                )
            record = dict(
                shape=tuple(value.shape),
                dtype=dtype_name(value.dtype),
                buffer=value.detach().cpu().clone(),
            )
        if tuple(record["shape"]) != tuple(value.shape) or record[
            "dtype"
        ] != dtype_name(value.dtype):
            raise ValueError(f"Tracing changed initialization shape/dtype for {name}")
        records[name] = record
    plan["records"] = records
    torch.save(plan, path, pickle_protocol=4)


@torch.no_grad()
def load_module_parameters(module, path, *, fullmaps=None):
    if not module._fullmap:
        return
    plan = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    if plan.get("version") != PLAN_VERSION or plan.get("scheme") != SCHEME:
        raise RuntimeError(
            "Regenerate initialization plans for native local GPU initialization"
        )
    if fullmaps is None:
        # Main loads each scale unit's metadata when importing the generated
        # class. Reuse its public accessor; replicated scale units have the
        # same cuts and need not be read again from disk.
        fullmaps = [
            module.get_attr_meta_map(rank)
            for rank in range(module.compute_config.plan_ngpus)
        ]
    cuts = partition_cuts(plan, fullmaps)
    generators = {}
    with ExitStack() as stack:
        token = _checkpoint_readers.set((stack, {}))
        try:
            for attr, meta in module._fullmap.items():
                target = getattr(module, attr)
                if target.device.type == "cuda" and target.device not in generators:
                    generators[target.device] = torch.Generator(device=target.device)
                record = plan["records"][meta.orig_name]
                initialize(
                    plan,
                    meta.orig_name,
                    meta.slicers,
                    out=target,
                    cuts=cuts[record.get("rng_key", meta.orig_name)],
                    generator=generators.get(target.device),
                )
                if meta.val_chunks != 1:
                    target.div_(meta.val_chunks)
        finally:
            _checkpoint_readers.reset(token)
