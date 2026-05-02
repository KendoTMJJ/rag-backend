from pydantic import BaseModel
from src.models.helpdesk import intent_to_display_label


class HelpdeskCategoryOut(BaseModel):
    id:            int
    intent:        str
    display_label: str
    description:   str | None
    pdf_url:       str | None

    class Config:
        from_attributes = True

    @classmethod
    def from_row(cls, row) -> "HelpdeskCategoryOut":
        return cls(
            id=row.id,
            intent=row.intent,
            display_label=intent_to_display_label(row.intent),
            description=row.description,
            pdf_url=row.pdf_url,
        )


class HelpdeskCategoryCreate(BaseModel):
    intent:      str
    description: str | None = None
    pdf_url:     str | None = None


class HelpdeskCategoryUpdate(BaseModel):
    description: str | None = None
    pdf_url:     str | None = None
