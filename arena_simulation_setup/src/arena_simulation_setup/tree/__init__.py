from __future__ import annotations

import abc
import asyncio
import itertools
import logging
import os
import subprocess
import threading
import typing
import warnings
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Optional

import attrs

try:
    from typing import Self
except ImportError:
    Self = typing.TypeVar('Self')

from arena_simulation_setup import (
    ARENA_ASSETS_DIR,
    DOMAIN_DEFAULT,
)
from arena_simulation_setup.utils.cattrs import Idempotent, Parseable, Serializable

NETWORK_PROVIDERS: Sequence[str] = os.environ.get('ASSET_BUCKETS', 'default').split(',')

# Utils


class PathContainer:
    def __init__(self, path: Path):
        self._path: Path = path

    @property
    def path(self) -> Path:
        return self._path


class PathView(PathContainer):
    @property
    def name(self) -> str:
        return self.path.name


class DynamicPath(PathContainer):
    @PathContainer.path.setter
    def path(self, value: Path):
        self._path = value

    def __init__(self, path: Path = Path('/dev/null')):
        super().__init__(path)


class DynamicPaths:
    WORLD = DynamicPath()
    ARENA = DynamicPath(Path(os.getenv('ARENA_ASSETS_DIR_LOCAL', ARENA_ASSETS_DIR / 'local')))

    @classmethod
    def as_resolvers(cls, _T: typing.Type[IdentifierT], /) -> Iterable[DynamicPathResolver[IdentifierT]]:
        yield DynamicPathResolver(_T, cls.WORLD, fn=lambda p: p / 'assets')
        yield DynamicPathResolver(_T, cls.ARENA)


IdentifierT = typing.TypeVar('IdentifierT', bound='IdentifierProtocol')


class ResolverBase(abc.ABC, typing.Generic[IdentifierT]):
    """
    Base class for resolvers.
    """
    @property
    def _asset_type(self) -> str:
        return self._IdentifierT.label()

    _cache: dict[IdentifierT, Path] = attrs.field(factory=dict, init=False)

    def __init__(self, _T: typing.Type[IdentifierT], /):
        self._IdentifierT: typing.Type[IdentifierT] = _T
        self._cache = {}

    @abc.abstractmethod
    async def resolve(self, identifier: IdentifierT) -> Optional[Path]:
        """
        Resolve the given identifier.
        """

    def invalidate(self):
        """
        Invalidate the cache.
        """
        self._cache.clear()

    def listall(self, **kwargs) -> Iterator[IdentifierT]:
        """
        List all local assets available. Builds the cache in the process.
        """
        return iter(())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._asset_type}>'


class PathResolverBase(ResolverBase[IdentifierT], abc.ABC, typing.Generic[IdentifierT]):
    """
    Resolve asset paths from disk.
    """
    @property
    @abc.abstractmethod
    def path(self) -> Path:
        ...

    async def resolve(self, identifier: IdentifierT) -> Optional[Path]:
        """
        Resolve the given identifier.
        """
        if identifier not in self._cache:
            candidate = self.path / identifier.relpath()
            if candidate.exists():
                self._cache[identifier] = candidate
                return candidate
        return self._cache.get(identifier, None)

    def listall(self, **kwargs) -> Iterator[IdentifierT]:
        """
        List all local assets available.
        """
        source = self.path
        if not source.is_dir():
            yield from ()
            return
        for root, dirs, files in os.walk(source):
            relpath = Path(root).relative_to(source)
            try:
                yield self._IdentifierT.from_relpath(relpath)
                # don't recurse
                dirs.clear()
                continue
            except Exception:
                pass

            for file in files:
                file_relpath = relpath / file
                try:
                    yield self._IdentifierT.from_relpath(file_relpath)
                except Exception:
                    pass

    def __repr__(self) -> str:
        return f"{super().__repr__()}(path={self.path})"


class SimplePathResolver(PathResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolve asset paths from a single disk location.
    """

    def __init__(self, _T: typing.Type[IdentifierT], /, path: Path, **kwargs):
        super().__init__(_T, **kwargs)
        self._path = path

    @property
    def path(self) -> Path:
        return self._path


class DynamicPathResolver(PathResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolve asset paths from a dynamic disk location.
    """

    def __init__(self, _T: typing.Type[IdentifierT], /, path: DynamicPath, *, fn: typing.Callable[[Path], Path] | None = None, **kwargs):
        super().__init__(_T, **kwargs)
        if fn is None:
            def _identity_fn(p): return p
            fn = _identity_fn
        self._dynamic_path = path
        self._fn = fn

    @property
    def path(self) -> Path:
        return self._fn(self._dynamic_path.path)


class NetResolver(typing.Generic[IdentifierT], SimplePathResolver[IdentifierT], ResolverBase[IdentifierT]):
    """
    Resolve asset paths from both disk and network.
    """

    def __init__(self, _T: typing.Type[IdentifierT], /, provider: str, **kwargs):
        path = ARENA_ASSETS_DIR / provider
        super().__init__(_T, path=path, **kwargs)
        self._provider: str = provider

        self._running_fetches: dict[IdentifierT, asyncio.Task[Optional[Path]]] = {}
        self._running_lock = asyncio.Lock()

    @classmethod
    async def check_output_async(cls, args: Iterable[str], **kwargs) -> bytes:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode or -1, list(args), output=stdout, stderr=stderr)
        return stdout

    async def _network_fetch(self, provider: str, identifier: IdentifierT) -> Optional[Path]:
        relpath = identifier.relpath()
        root_path = ARENA_ASSETS_DIR / provider
        target_path = root_path / relpath

        formats = os.environ.get('ARENA_MODELS_FORMATS', '').split(',')

        try:
            if (await self.check_output_async([
                'ros2',
                'run',
                'arena_models',
                'arena_models',
                '-s',
                'net',
                provider,
                'exists',
                str(relpath),
            ])).strip().decode() == '1':
                logging.info(f"Fetching asset {identifier} from network provider {provider}...")
                await self.check_output_async([
                    'ros2',
                    'run',
                    'arena_models',
                    'arena_models',
                    '-s',
                    'net',
                    provider,
                    'fetch',
                    str(relpath),
                    '-o',
                    str(root_path),
                    *itertools.chain.from_iterable(('--format', format) for format in formats if format),
                ])
                return target_path

        except subprocess.CalledProcessError:
            import traceback
            logging.warning(traceback.format_exc())
            return None

    async def resolve(self, identifier):
        """
        Resolve the given name.
        """
        if identifier in self._cache:
            return self._cache[identifier]

        local_result = await SimplePathResolver.resolve(self, identifier)
        if local_result is not None:
            return local_result

        async with self._running_lock:
            if identifier in self._running_fetches:
                fetch_task = self._running_fetches[identifier]
            else:
                fetch_task = asyncio.create_task(self._network_fetch(self._provider, identifier))
                self._running_fetches[identifier] = fetch_task

        net_result = await fetch_task
        if net_result is not None:
            self._cache[identifier] = net_result

        return self._cache.get(identifier, None)

    def listall(self, *, network: bool = False, **kwargs) -> Iterator[IdentifierT]:
        """
        List all local assets available. Builds the cache in the process.
        """
        yield from SimplePathResolver.listall(self, **kwargs)
        if network:
            warnings.warn(f'Network listing is not yet supported in {repr(self.__class__)}', UserWarning)
            yield from ()

    def __repr__(self) -> str:
        return f"{super().__repr__()}(net={self._provider})"

    @classmethod
    def all(cls, _T: typing.Type[IdentifierT], /) -> Iterable[Self]:
        for provider in NETWORK_PROVIDERS:
            yield cls(_T, provider=provider)


class FallbackResolver(ResolverBase[IdentifierT], typing.Generic[IdentifierT]):
    """
    Resolver that always yields a fallback path.

    ``resolve`` returns a path **without checking that it exists**, and that is
    load-bearing rather than an oversight: it is how a not-yet-existing asset
    gets a location to be written to (``World.save`` does
    ``os.makedirs(..., exist_ok=True)`` on it), so this resolver cannot be made
    unconditionally strict without breaking asset generation.

    The cost is that, once this resolver is in an identifier's chain, resolution
    always succeeds, so :meth:`Identifier.resolve_path`'s "not found among"
    diagnostic becomes unreachable and a missing asset surfaces much later as an
    unexplained failure somewhere downstream -- for worlds, as an ``open()`` on a
    ``map.yaml`` that names the map file and not the missing world.  Readers that
    want the diagnostic ask for it explicitly with ``require_exists=True``; see
    :meth:`Identifier.resolve_path`.
    """

    def __init__(self, _T: typing.Type[IdentifierT], /, path: Path, **kwargs):
        super().__init__(_T, **kwargs)
        self._path = path

    async def resolve(self, identifier: IdentifierT) -> Optional[Path]:
        """
        Resolve the given identifier.
        """
        return self._path / identifier.relpath()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._path)})"

# IDENTIFIERS


#: Default for ``require_exists`` on ``Identifier.resolve_path`` / ``resolve`` /
#: ``resolve_sync``.  ``False`` preserves the behaviour every existing caller
#: relies on -- notably the asset *generation* callers, which resolve a location
#: precisely because it does not exist yet.
#:
#: Declared once on purpose.  Repeating the literal in each of the three
#: signatures is how a rule acquires several expressions that can drift apart, and
#: it also silently costs a mutation test its power: flipping one of three copies
#: changes nothing observable.
REQUIRE_EXISTS_DEFAULT: bool = False


T = typing.TypeVar('T')
T_co = typing.TypeVar('T_co', covariant=True)


class IdentifierProtocol(typing.Protocol, typing.Generic[T_co]):
    async def resolve(self, **kwargs) -> T_co:
        """Resolve and load the asset referenced by this identifier.
        """
        ...

    @property
    def shortname(self) -> str:
        """Get the short name of the identifier.
        """
        ...

    def relpath(self) -> Path:
        """Get the path representation of the identifier.

        Returns:
            Path: The path of the identifier relative to a repository.
        """
        ...

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        """Create an Identifier from a relative path.

        Args:
            relpath(Path): The relative path.
        Returns:
            Identifier: The created Identifier.
        """
        ...

    @classmethod
    def label(cls) -> str:
        """Short label for the identifier type.
        """
        ...

    @classmethod
    def listall(cls, **kwargs) -> Iterator[Self]:
        """List all local assets available.
        """
        ...


@attrs.define(eq=False, hash=False)
class Identifier(IdentifierProtocol[T], Parseable, Serializable, Idempotent, typing.Generic[T]):
    """Represents an identifier referencing an path.
    """
    name: str

    @property
    def shortname(self) -> str:
        return self.name

    def relpath(self) -> Path:
        return Path(self.name)

    async def resolve_path(
        self, *, require_exists: bool = REQUIRE_EXISTS_DEFAULT,
    ) -> Path:
        """Locate the asset referenced by this identifier.

        Args:
            require_exists: Treat a resolver result that does not exist on disk as
                *not* a resolution, so the caller gets this method's
                "not found among" ``FileNotFoundError`` -- naming the identifier,
                every resolver searched, and each non-existent candidate path --
                instead of a path it will fail on later for a reason that does not
                mention the asset.  Defaults to ``False``, which preserves the
                behaviour every existing caller relies on, including the asset
                *generation* callers that resolve a location precisely because it
                does not exist yet (see :class:`FallbackResolver`).

        Raises:
            FileNotFoundError: No resolver produced a usable path.
        """

        candidates: list[tuple[ResolverBase, Optional[Path]]] = []
        for resolver in self._resolvers:
            resolved = await resolver.resolve(self)
            candidates.append((resolver, resolved))
            if resolved is None:
                continue
            if require_exists and not resolved.exists():
                continue
            return resolved

        msg = f'{self} not found among'
        for resolver, resolved in candidates:
            msg += f'\n\t{repr(resolver)}'
            if resolved is not None:
                # Reported from the loop above rather than re-resolved: calling a
                # resolver twice can trigger a second network fetch.
                msg += f'\n\t\t-> {resolved} (does not exist)'
        raise FileNotFoundError(msg)

    async def resolve(
        self, *, require_exists: bool = REQUIRE_EXISTS_DEFAULT, **kwargs,
    ) -> T:
        return self.load(
            await self.resolve_path(require_exists=require_exists),
            **kwargs,
        )

    def resolve_sync(
        self, *, require_exists: bool = REQUIRE_EXISTS_DEFAULT, **kwargs,
    ) -> T:
        """Synchronously load the asset referenced by this identifier.
        """
        result: T = None  # type: ignore
        exc: Optional[Exception] = None

        def _run_async_load():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                nonlocal result
                result = loop.run_until_complete(
                    self.resolve(require_exists=require_exists, **kwargs))
            except Exception as e:
                nonlocal exc
                exc = e
            finally:
                loop.close()

        # NOTE (unfixed, documented): this join has no timeout, so a resolver that
        # blocks forever hangs the caller with no output at all.  It is reachable
        # only through NetResolver.resolve -> _network_fetch, whose
        # `process.communicate()` is also untimed; NetResolver was measured NOT to
        # be in WorldIdentifier's resolver chain, so this is latent rather than
        # active and is deliberately left alone here.
        thread = threading.Thread(target=_run_async_load)
        thread.start()
        thread.join()

        if exc is not None:
            raise exc
        return result

    @abc.abstractmethod
    def load(self, path: Path, /, **kwargs) -> T:
        """Load the asset referenced by this identifier.
        """

    def serialize(self):
        return self.shortname

    # Class Methods

    @classmethod
    def label(cls) -> str:
        return cls.__name__

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        assert len(relpath.parts) >= 1, f'Expected at least 1 part in relpath, got {len(relpath.parts)}'
        return cls(name=str(relpath))

    @classmethod
    def parse(cls, value: str | Self) -> Self:
        """Parse path into an Identifier.

        Args:
            identifier(str): The identifier string to parse.

        Returns:
            Identifier: The parsed Identifier object.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TypeError(f'Expected str, got {repr(value)}')

        return cls(name=value)

    @classmethod
    def converter(cls, *args, **kwargs):
        return super().instance_or(cls.parse)(*args, **kwargs)

    @classmethod
    def listall(cls, **kwargs) -> Iterator[Self]:
        for resolver in cls._resolvers:
            yield from resolver.listall(**kwargs)

    # Resolvers
    _resolvers: typing.ClassVar[list[ResolverBase]] = []

    @classmethod
    def use(cls, *resolver: ResolverBase[Self]):
        if '_resolvers' not in cls.__dict__:
            cls._resolvers = []
        cls._resolvers.extend(resolver)

    @classmethod
    def inline(cls: typing.Type[Self], data: T, /, name: str = '', modifiers: dict | None = None) -> Self:
        """Create an InlineIdentifier containing the given data.

        Args:
            data(T_co): The asset data.
            name(str, optional): The name of the asset. Defaults to ''.
            modifiers(dict | None, optional): The modifiers dict for the asset. Defaults to None.

        Returns:
            InlineIdentifier[T_co]: The created InlineIdentifier.
        """
        del modifiers

        TypedInline = type(
            f"TypedInline_{cls.__name__}",
            (InlineIdentifier, cls),
            {}
        )
        return typing.cast(Self, TypedInline(data, name=name))


@attrs.define(eq=False, hash=False)
class AssetIdentifier(Identifier[T], typing.Generic[T]):
    """Represents an identifier referencing an asset.
    """
    name: str

    _asset_type: typing.ClassVar[str]

    def relpath(self) -> Path:
        return Path(self._asset_type) / self.name

    @classmethod
    def label(cls) -> str:
        """Short label for the identifier type.
        """
        return cls._asset_type

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        """Create an Identifier from a relative path.

        Args:
            relpath(Path): The relative path.
        Returns:
            Identifier: The created Identifier.
        """
        assert len(relpath.parts) >= 2, f'Expected at least 2 parts in relpath, got {len(relpath.parts)}'
        assert relpath.parts[0] == cls._asset_type, f'Expected asset type {cls._asset_type}, got {relpath.parts[0]}'
        return cls(name=str(Path(*relpath.parts[1:])))


@attrs.define(eq=False, hash=False)
class DomainAssetIdentifier(AssetIdentifier[T], typing.Generic[T]):
    """Represents an identifier referencing an asset.
    """
    name: str
    domain: str = attrs.field(default=DOMAIN_DEFAULT)

    def __hash__(self) -> int:
        return hash(
            (
                self.domain,
                self.name,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DomainAssetIdentifier):
            return False
        return self.domain == other.domain and self.name == other.name

    @classmethod
    def parse(cls, value: str | Self) -> Self:  # type: ignore
        """Parse path of the form[domain:]name into an Identifier.

        Args:
            identifier(str): The identifier string to parse.
            default_target(AssetType): The default target type if not specified.
            default_domain(str): The default domain if not specified.

        Returns:
            Identifier: The parsed Identifier object.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise TypeError(f'Expected str, got {repr(value)}')

        parts = list(value.split('/', 3))

        if len(parts) > 1 and parts[1] == cls._asset_type:
            # asset type included, shift
            parts.pop(1)
        if len(parts) > 1:
            # domain included
            domain = parts.pop(0)
        else:
            domain = DOMAIN_DEFAULT
        name = "/".join(parts)

        return cls(
            domain=domain,
            name=name,
        )

    @property
    def shortname(self) -> str:
        return f'{self.domain}/{self._asset_type}/{self.name}'

    def relpath(self) -> Path:
        """Get the path representation of the identifier.

        Returns:
            Path: The path of the identifier relative to a repository.
        """
        return Path(self.domain) / self._asset_type / self.name

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        """Create an Identifier from a relative path.

        Args:
            relpath(Path): The relative path.
        Returns:
            Identifier: The created Identifier.
        """
        assert len(relpath.parts) >= 3, f'Expected at least 3 parts in relpath, got {len(relpath.parts)}'
        assert relpath.parts[1] == cls._asset_type, f'Expected asset type {cls._asset_type}, got {relpath.parts[0]}'
        return cls(domain=relpath.parts[0], name=str(Path(*relpath.parts[2:])))


@attrs.define(eq=False, hash=False)
class ModifiersDomainAssetIdentifier(DomainAssetIdentifier[T], typing.Generic[T]):
    """DomainIdentifiers that supports modifiers.
    """
    _modifiers: dict | None = attrs.field(default=None, alias='modifiers', repr=False)

    @property
    def modifiers(self) -> dict:
        """Get the modifiers dictionary.
        """
        return {} if self._modifiers is None else self._modifiers

    def __hash__(self) -> int:
        return hash(
            (
                self.domain,
                self.name,
                frozenset(self.modifiers.items())
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModifiersDomainAssetIdentifier):
            return False
        return self.domain == other.domain and self.name == other.name and self.modifiers == other.modifiers

    @classmethod
    def parse(cls, value: str | tuple[str, dict] | Self) -> Self:  # type: ignore
        """Parse path of the form[domain:]name into an Identifier.

        Args:
            identifier(str): The identifier string to parse.
            default_target(AssetType): The default target type if not specified.
            default_domain(str): The default domain if not specified.

        Returns:
            Identifier: The parsed Identifier object.
        """
        if isinstance(value, DomainAssetIdentifier):
            if isinstance(value, cls):
                return value
            raise TypeError(f'Received Identifier of type {type(value)}, expected {cls}')

        modifiers = None
        if isinstance(value, Iterable) and not isinstance(value, str):
            if not len(value) == 2:
                raise ValueError(f'Expected Iterable of length 2, got {len(value)}')
            value, modifiers = value[0], value[1]

            if not isinstance(value, str):
                raise TypeError(f'Expected value to be str, got {repr(value)}')
            if not isinstance(modifiers, dict):
                raise TypeError(f'Expected modifiers to be dict, got {repr(modifiers)}')

        base = super().parse(value)

        return cls(
            domain=base.domain,
            name=base.name,
            modifiers=modifiers,
        )

    def serialize(self) -> typing.Any:
        base = super().serialize()
        if self.modifiers is None:
            return base
        return (base, self.modifiers)


class InlineIdentifier(Identifier[T], typing.Generic[T]):
    """An identifier that directly contains the asset data.
    """
    _data: T

    def __init__(self, data: T, /, name: str = '', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self._data = data

    def serialize(self) -> str:
        raise TypeError('InlineIdentifier cannot be serialized')

    async def resolve(self, **kwargs) -> T:
        del kwargs  # unused
        return self._data

    def load(self, path: Path, /, **kwargs) -> T:
        raise NotImplementedError('How did you even get here?')
