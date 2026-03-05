"""
database.py — Configuração do banco de dados assíncrono.

Decisão Arquitetural:
    Este módulo centraliza TODA a infraestrutura de acesso a dados,
    seguindo o padrão "Single Source of Truth" para configuração do banco.
    Isso garante que qualquer mudança de engine, driver ou estratégia de
    conexão seja feita em um único lugar, sem impactar models ou endpoints.

    Utilizamos SQLAlchemy 2.0+ com driver assíncrono (aiosqlite) para
    aproveitar o modelo async/await nativo do FastAPI, evitando bloqueio
    de I/O no event loop. A escolha de SQLite simplifica a execução local
    e nos testes, sem necessidade de infraestrutura externa.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Configuração do Engine Assíncrono
# ---------------------------------------------------------------------------
# "sqlite+aiosqlite" é a forma assíncrona do SQLAlchemy para SQLite.
# `echo=False` em produção; troque para `True` durante debugging para
# visualizar as queries SQL geradas pelo ORM no console.
# ---------------------------------------------------------------------------
DATABASE_URL: str = "sqlite+aiosqlite:///./biblioteca.db"

engine = create_async_engine(DATABASE_URL, echo=False)

# ---------------------------------------------------------------------------
# Session Factory
# ---------------------------------------------------------------------------
# `expire_on_commit=False` evita que os atributos dos objetos ORM sejam
# invalidados após o commit, o que causaria lazy-loads síncronos
# incompatíveis com o contexto assíncrono.
# ---------------------------------------------------------------------------
async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Base Declarativa
# ---------------------------------------------------------------------------
# Todas as entidades (models) herdam desta classe base. Centralizar a Base
# aqui — e não no models.py — evita importações circulares quando múltiplos
# módulos precisam referenciá-la (ex: database.py para criar tabelas,
# models.py para definir entidades).
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Classe base declarativa para todos os models SQLAlchemy."""


# ---------------------------------------------------------------------------
# Dependency Injection — Sessão do Banco
# ---------------------------------------------------------------------------
# Função geradora assíncrona usada como dependência do FastAPI via `Depends`.
# O padrão `try/finally` garante que a sessão seja SEMPRE fechada, mesmo
# em caso de exceção, prevenindo connection leaks.
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Fornece uma sessão de banco de dados por requisição (Dependency Injection)."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Inicialização do Banco
# ---------------------------------------------------------------------------
# Cria todas as tabelas definidas nos models que herdam de `Base`.
# Chamada no startup da aplicação (via lifespan). Em produção, seria
# substituída por um sistema de migrations (ex: Alembic).
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """Cria as tabelas no banco de dados (idempotente)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
