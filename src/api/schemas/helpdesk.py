from pydantic import BaseModel
from src.models.helpdesk import intent_to_display_label


class HelpdeskCategoryOut(BaseModel):
    id:            int
    intent:        str
    display_label: str
    description:   str | None
    has_document:  bool
    document_url:  str | None

    class Config:
        from_attributes = True

    @classmethod
    def from_row(cls, row) -> "HelpdeskCategoryOut":
        has_doc = bool(row.document_data)
        return cls(
            id=row.id,
            intent=row.intent,
            display_label=intent_to_display_label(row.intent),
            description=row.description,
            has_document=has_doc,
            document_url=f"/helpdesk/document/{row.intent}" if has_doc else None,
        )


class HelpdeskCategoryCreate(BaseModel):
    intent:      str
    description: str | None = None


class HelpdeskCategoryUpdate(BaseModel):
    description: str | None = None
