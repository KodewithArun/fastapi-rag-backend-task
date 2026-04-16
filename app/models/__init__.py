from app.db.session import Base
from app.models.metadata import DocumentMetadata
from app.models.booking import InterviewBooking

# Export Base so alembic/metadata can easily pick it up
__all__ = ["Base", "DocumentMetadata", "InterviewBooking"]
