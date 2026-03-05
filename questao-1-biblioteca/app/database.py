"""
database.py — Configuração assíncrona do SQLAlchemy e fábrica de sessões.

Decisão Arquitetural:
    Ao construir a base de dados com `aiosqlite` e SQLAlchemy Mapped, adotamos
    o ciclo de vida totalmente não-bloqueante no I/O. As sessões do banco 
    são controladas via Generator na função `get_db()`, delegando a 
    responsabilidade de fechamento seguro ao Dependency Injection do FastAPI.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL: str = "sqlite+aiosqlite:///./biblioteca.db"

engine = create_async_engine(DATABASE_URL, echo=False)

async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Classe base declarativa para todos os models SQLAlchemy."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Fornece uma sessão de banco de dados por requisição (Dependency Injection)."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Cria as tabelas no banco de dados (idempotente)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
