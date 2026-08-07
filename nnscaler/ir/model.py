#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

_COMPONENTS = frozenset({
    'embedding',
    'attention',
    'moe',
    'mlp',
    'normalization',
    'output',
    'other',
})


def _validate_stable_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string')
    if not value or value != value.strip() or not value.isprintable():
        raise ValueError(f'{name} must be a non-empty stable string')
    if any(character.isspace() or character == '=' for character in value):
        raise ValueError(f"{name} may not contain whitespace or '='")


def _is_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    component: str
    model_fqn: str
    model_site: str

    def __post_init__(self) -> None:
        _validate_stable_string('component', self.component)
        _validate_stable_string('model_fqn', self.model_fqn)
        _validate_stable_string('model_site', self.model_site)
        if self.component not in _COMPONENTS:
            raise ValueError(
                f'component must be one of {sorted(_COMPONENTS)}, got {self.component!r}'
            )
        if _is_absolute_path(self.model_fqn):
            raise ValueError('model_fqn must not be an absolute path')
        if _is_absolute_path(self.model_site):
            raise ValueError('model_site must not be an absolute path')
        normalized_site = PurePosixPath(self.model_site.replace('\\', '/'))
        if '..' in normalized_site.parts:
            raise ValueError('model_site must not contain parent traversal')


def consensus_model_spec(
    specs: Iterable[ModelSpec | None],
) -> ModelSpec | None:
    iterator = iter(specs)
    try:
        model_spec = next(iterator)
    except StopIteration:
        return None
    return model_spec if all(spec == model_spec for spec in iterator) else None
