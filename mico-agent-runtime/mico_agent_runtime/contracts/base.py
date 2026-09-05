from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationInfo, field_validator


class ClosedModel(BaseModel):
    """Pydantic v2 base model used for every external or wire-facing object."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    # A subclass may explicitly allow structural whitespace for a bounded
    # source-text field.  All other fields keep the default closed-wire rule.
    allowed_control_character_fields: ClassVar[frozenset[str]] = frozenset()

    @field_validator("*", mode="after")
    @classmethod
    def reject_control_characters(cls, value: object, info: ValidationInfo) -> object:
        allowed = "\r\n\t" if info.field_name in cls.allowed_control_character_fields else ""
        if isinstance(value, str) and any(
            ord(char) < 32 and char not in allowed for char in value
        ):
            raise ValueError("control characters are not allowed")
        return value


NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
]
