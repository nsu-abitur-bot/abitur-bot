"""Add message logs table

Revision ID: 2026_04_01_0000-add_message_logs_table
Revises: 2026_03_08_0000-add_message_table
Create Date: 2026-04-01 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2026_04_01_0000-add_message_logs_table'
down_revision: Union[str, None] = '2026_03_08_0000-add_message_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create message_logs table
    op.create_table(
        'message_logs',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('message_type', sa.String(length=50), nullable=False),  # 'user_input', 'rag_context', 'llm_response', 'faq_match'
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),  # для источников, длины ответа и т.д.
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('ix_message_logs_user_id', 'message_logs', ['user_id'])
    op.create_index('ix_message_logs_session_id', 'message_logs', ['session_id'])
    op.create_index('ix_message_logs_created_at', 'message_logs', ['created_at'])
    op.create_index('ix_message_logs_message_type', 'message_logs', ['message_type'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_message_logs_message_type', table_name='message_logs')
    op.drop_index('ix_message_logs_created_at', table_name='message_logs')
    op.drop_index('ix_message_logs_session_id', table_name='message_logs')
    op.drop_index('ix_message_logs_user_id', table_name='message_logs')
    
    # Drop table
    op.drop_table('message_logs')
