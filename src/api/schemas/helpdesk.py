from pydantic import BaseModel


class HelpdeskCategoryOut(BaseModel):
    id:          int
    intent:      str
    label:       str
    description: str | None
    pdf_url:     str | None

    class Config:
        from_attributes = True


class HelpdeskCategoryCreate(BaseModel):
    intent:      str
    label:       str
    description: str | None = None
    pdf_url:     str | None = None


class HelpdeskCategoryUpdate(BaseModel):
    label:       str | None = None
    description: str | None = None
    pdf_url:     str | None = None
