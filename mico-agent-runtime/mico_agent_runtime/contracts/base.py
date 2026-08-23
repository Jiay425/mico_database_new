from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator


class ClosedModel(BaseModel):
    """Pydantic v2 base model used for every external or wire-facing object."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*", mode="after")
    @classmethod
    def reject_control_characters(cls, value: object) -> object:
        if isinstance(value, str) and any(ord(char) < 32 for char in value):
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
