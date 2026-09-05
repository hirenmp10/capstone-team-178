"""Catalogue of the USD assets the scene can be built from.

Pure Python. This module must never import Isaac Sim -- it is loaded by the
config layer and by tests that run without a GPU.

Why a registry rather than paths in the scene config
----------------------------------------------------
A scene config says *what to place and where*. It should not also have to know
that a soda can is 139 mm tall, weighs 414 g, lives at
``/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd``, and -- crucially --
ships **without** colliders so the framework has to author them.

Splitting those apart means the physical facts about an object are stated once.
A scene refers to ``master_chef_can`` and inherits the correct mass, size,
collider strategy and semantic label; get the mass wrong and every scene using
that object is wrong in the same way, which is a bug you can actually find.

The graspable/container flags are the part that earns its keep at runtime. The
Franka's fingers span 80 mm, so the 159 mm YCB bowl is not graspable by this
embodiment -- but it is a perfectly good *destination*. Encoding that here stops
the planner from spending a full grasp-synthesis cycle discovering it, and stops
"put the marker in the bowl" from being rejected because the bowl failed a
grasp check it should never have been given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mfw.config.schema import ConfigError

__all__ = ["AssetSpec", "FurnitureSpec", "AssetRegistry", "DEFAULT_ASSETS_PATH"]

DEFAULT_ASSETS_PATH = Path(__file__).resolve().parents[2] / "configs" / "assets.yaml"

_VALID_GRASP_AXES = ("any", "vertical", "horizontal")


@dataclass(frozen=True)
class AssetSpec:
    """One manipulable object: where its mesh is, and what it physically is.

    ``size_m`` is the object's real axis-aligned extent in metres. It is not used
    to scale the mesh -- YCB assets are already authored at true scale -- but to
    sanity-check perception. A detector that measures a 66 mm can as 130 mm has a
    bug, and without a ground-truth extent to compare against, that bug is
    invisible.
    """

    name: str
    usd: str
    category: str
    semantic: str
    mass_kg: float
    size_m: tuple[float, float, float]
    graspable: bool = True
    container: bool = False
    has_physics: bool = False
    """True only for the four YCB assets under ``Axis_Aligned_Physics``, which
    ship with colliders and mass already authored. For everything else the
    framework applies them, and skipping that step spawns a mesh that falls
    straight through the table."""
    upright_quat: tuple[float, float, float, float] = (0.70710678, -0.70710678, 0.0, 0.0)
    """Rotation (w, x, y, z) that stands this asset upright in a Z-up world.

    The default is -90 degrees about X, because **the YCB set is authored with
    its up axis along -Y** while Isaac's world is Z-up. Every asset in this
    catalogue is YCB, so the correction is the default rather than the
    exception; a non-YCB asset should set this to identity explicitly.

    Two separate things were measured here, because each hides the other.

    *That* a correction is needed: with identity orientation the tomato soup can
    measures 67.7 x 101.9 x 67.7 mm -- its 102 mm axis lying along world Y, the
    can on its side. After the rotation it measures 67.7 x 67.7 x 101.9,
    standing up. Without this, cans lie down and roll and bottles are prone,
    which reads as unstable physics rather than a wrong orientation.

    *Which sign* it takes: bounding boxes cannot answer this, because extents
    are magnitudes and +90 and -90 produce identical numbers. Settled by
    function instead -- a brick dropped over the bowl lands inside it at -90
    (33.8 mm above the table) and perches on an inverted dome at +90 (82.4 mm).
    The wrong sign leaves every open container upside down while every dimension
    still checks out, so "put it in the bowl" would quietly become "put it on
    the bowl".
    """
    grasp_axis: str = "any"
    collision_approximation: str = ""
    """Override the collider strategy that :mod:`mfw.physics.collision` would
    otherwise derive from ``category``. Empty means "derive it"; set this only
    when an individual asset's geometry defeats its category's default."""
    substitutes: str = ""
    """What real-world item this stands in for, when it is a substitute. Recorded
    so the mapping is auditable rather than assumed."""
    notes: str = ""

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("asset spec has an empty name")
        if not self.usd.startswith("/"):
            raise ConfigError(
                f"asset {self.name!r}: usd must be a path relative to the assets root "
                f"(starting with '/'), got {self.usd!r}"
            )
        if self.mass_kg <= 0.0:
            raise ConfigError(f"asset {self.name!r}: mass_kg must be > 0, got {self.mass_kg}")
        if len(self.size_m) != 3 or any(s <= 0.0 for s in self.size_m):
            raise ConfigError(f"asset {self.name!r}: size_m must be three positive values")
        if self.grasp_axis not in _VALID_GRASP_AXES:
            raise ConfigError(
                f"asset {self.name!r}: grasp_axis must be one of {_VALID_GRASP_AXES}, "
                f"got {self.grasp_axis!r}"
            )

    @property
    def max_extent(self) -> float:
        """Largest dimension -- the one that decides if it fits the gripper."""
        return max(self.size_m)

    @property
    def min_extent(self) -> float:
        """Smallest dimension: the width the fingers would close across on the
        easiest approach."""
        return min(self.size_m)

    def fits_gripper(self, max_width: float) -> bool:
        """Whether *some* approach direction fits within ``max_width``.

        Uses the smallest dimension because the gripper may approach along any
        axis: a 250 mm bottle that is 65 mm across is graspable around its
        middle, even though its height dwarfs the finger span.
        """
        return self.min_extent <= max_width


@dataclass(frozen=True)
class FurnitureSpec:
    """A static fixture: furniture, storage, or decor.

    Static bodies, so no mass. They shape the workspace and give the cameras a
    realistic scene to perceive, but they are never manipulation targets.
    """

    name: str
    usd: str
    semantic: str
    role: str = "decor"
    notes: str = ""

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("furniture spec has an empty name")
        if not self.usd.startswith("/"):
            raise ConfigError(f"furniture {self.name!r}: usd must start with '/', got {self.usd!r}")


@dataclass(frozen=True)
class AssetRegistry:
    """Everything the scene builder is allowed to place.

    Lookup is by name and raises on a miss rather than returning ``None``. A
    typo'd asset name should stop scene construction at once -- the alternative
    is a scene that silently comes up one object short, which then reads as a
    perception failure when the robot cannot find what it was asked for.
    """

    objects: dict[str, AssetSpec] = field(default_factory=dict)
    furniture: dict[str, FurnitureSpec] = field(default_factory=dict)
    environments: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AssetRegistry:
        """Read the catalogue from YAML."""
        import yaml  # noqa: PLC0415

        resolved = Path(path) if path is not None else DEFAULT_ASSETS_PATH
        if not resolved.is_file():
            raise ConfigError(f"asset catalogue not found: {resolved}")

        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{resolved}: expected a mapping at the top level")

        return cls.from_dict(raw, source=str(resolved))

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> AssetRegistry:
        objects: dict[str, AssetSpec] = {}
        for entry in raw.get("objects", ()) or ():
            spec = _build(AssetSpec, entry, source, "objects")
            spec.validate()
            if spec.name in objects:
                raise ConfigError(f"{source}: duplicate object asset {spec.name!r}")
            objects[spec.name] = spec

        furniture: dict[str, FurnitureSpec] = {}
        for entry in raw.get("furniture", ()) or ():
            spec = _build(FurnitureSpec, entry, source, "furniture")
            spec.validate()
            if spec.name in furniture:
                raise ConfigError(f"{source}: duplicate furniture asset {spec.name!r}")
            furniture[spec.name] = spec

        environments: dict[str, str] = {}
        for name, entry in (raw.get("environments", {}) or {}).items():
            if isinstance(entry, str):
                environments[name] = entry
            elif isinstance(entry, dict) and "usd" in entry:
                environments[name] = str(entry["usd"])
            else:
                raise ConfigError(f"{source}: environment {name!r} needs a 'usd' key")

        return cls(objects=objects, furniture=furniture, environments=environments)

    # -- lookup ---------------------------------------------------------

    def object(self, name: str) -> AssetSpec:
        try:
            return self.objects[name]
        except KeyError:
            raise ConfigError(
                f"unknown object asset {name!r}; available: {sorted(self.objects)}"
            ) from None

    def furniture_item(self, name: str) -> FurnitureSpec:
        try:
            return self.furniture[name]
        except KeyError:
            raise ConfigError(
                f"unknown furniture asset {name!r}; available: {sorted(self.furniture)}"
            ) from None

    def environment(self, name: str) -> str:
        try:
            return self.environments[name]
        except KeyError:
            raise ConfigError(
                f"unknown environment {name!r}; available: {sorted(self.environments)}"
            ) from None

    # -- queries --------------------------------------------------------

    def by_category(self, category: str) -> list[AssetSpec]:
        return sorted(
            (s for s in self.objects.values() if s.category == category),
            key=lambda s: s.name,
        )

    def by_semantic(self, semantic: str) -> list[AssetSpec]:
        return sorted(
            (s for s in self.objects.values() if s.semantic == semantic),
            key=lambda s: s.name,
        )

    def graspable(self, max_width: float | None = None) -> list[AssetSpec]:
        """Objects this embodiment can actually pick up.

        ``max_width`` additionally filters by the gripper's span, so a registry
        shared across robots still yields a per-robot answer.
        """
        found = [s for s in self.objects.values() if s.graspable]
        if max_width is not None:
            found = [s for s in found if s.fits_gripper(max_width)]
        return sorted(found, key=lambda s: s.name)

    def containers(self) -> list[AssetSpec]:
        """Objects that can receive a placement ("put it in the bowl")."""
        return sorted((s for s in self.objects.values() if s.container), key=lambda s: s.name)

    def categories(self) -> list[str]:
        return sorted({s.category for s in self.objects.values()})

    def substitutions(self) -> dict[str, str]:
        """Asset name -> what it stands in for.

        Surfaced in the scene report so a reader can see that the "water bottle"
        is a YCB mustard bottle, rather than discovering it from a screenshot.
        """
        return {n: s.substitutes for n, s in sorted(self.objects.items()) if s.substitutes}


def _build(cls: type, entry: Any, source: str, section: str) -> Any:
    """Construct a spec from a YAML mapping, rejecting unknown keys.

    Unknown keys are an error, not a warning. A misspelled ``mass_kgs`` would
    otherwise leave the object at its default mass while looking configured --
    exactly the kind of silent wrongness the config layer exists to prevent.
    """
    if not isinstance(entry, dict):
        raise ConfigError(f"{source}: each {section} entry must be a mapping, got {type(entry)}")

    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(entry) - valid
    if unknown:
        raise ConfigError(
            f"{source}: {section} entry {entry.get('name', '?')!r} has unknown keys "
            f"{sorted(unknown)}; valid keys are {sorted(valid)}"
        )

    kwargs = dict(entry)
    if "size_m" in kwargs and kwargs["size_m"] is not None:
        kwargs["size_m"] = tuple(float(v) for v in kwargs["size_m"])
    return cls(**kwargs)
