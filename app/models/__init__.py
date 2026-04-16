from app.db.session import Base
from app.models.booking import InterviewBooking
from app.models.metadata import DocumentMetadata

# Export Base so alembic/metadata can easily pick it up
__all__ = ["Base", "DocumentMetadata", "InterviewBooking"]
