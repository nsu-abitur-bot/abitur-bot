"""Add category and tagged_at to message table

Revision ID: 2026_04_02_0000-add_category_and_tagged_at_to_message
Revises: 2026_04_01_0000-add_message_logs_table
Create Date: 2026-04-02 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9d8e7f6a5b4c'
down_revision: Union[str, None] = '8c7d6e5a4b3c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add category column to message table
    op.add_column('message', sa.Column('category', sa.String(length=50), nullable=True))
    
    # Add tagged_at column to message table
    op.add_column('message', sa.Column('tagged_at', sa.DateTime(), nullable=True))
    
    # Create indexes for performance
    op.create_index('idx_message_category', 'message', ['category'])
    op.create_index('idx_message_tagged_at', 'message', ['tagged_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_message_tagged_at', table_name='message')
    op.drop_index('idx_message_category', table_name='message')
    
    # Drop columns
    op.drop_column('message', 'tagged_at')
    op.drop_column('message', 'category')
