"""
main.py — Aplicação FastAPI e definição dos endpoints da API.

Decisão Arquitetural:
    Este módulo é o ponto de entrada da aplicação e contém APENAS a
    configuração do FastAPI e as rotas (endpoints). Toda lógica de
    persistência, validação e modelagem está delegada aos módulos
    especializados (database.py, models.py, schemas.py).

    Essa separação permite que:
    - Os endpoints sejam lidos como uma "tabela de rotas" limpa.
    - Mudanças no banco não afetem a camada de apresentação.
    - Os testes possam substituir dependências (ex: banco) facilmente.

    Utilizamos o padrão `lifespan` (ASGI lifecycle) em vez do
    deprecated `@app.on_event("startup")` para inicializar o banco.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, init_db
from app.models import Livro
from app.schemas import LivroCreate, LivroResponse


# ---------------------------------------------------------------------------
# Lifespan — Ciclo de vida da aplicação
# ---------------------------------------------------------------------------
# O `asynccontextmanager` nos permite executar código no startup (antes do
# `yield`) e no shutdown (após o `yield`). Aqui, criamos as tabelas do
# banco na inicialização. Em produção, isso seria gerenciado pelo Alembic.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gerencia o ciclo de vida da aplicação: cria tabelas no startup."""
    await init_db()
    yield


# ---------------------------------------------------------------------------
# Instância da Aplicação
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Biblioteca Virtual",
    description=(
        "API RESTful para gerenciamento de uma biblioteca virtual. "
        "Permite cadastrar e buscar livros por título e autor."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# POST /livros/ — Cadastro de livros
# ---------------------------------------------------------------------------
@app.post(
    "/livros/",
    response_model=LivroResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar um novo livro",
    description="Recebe os dados de um livro e o persiste no banco de dados.",
)
async def criar_livro(
    livro_data: LivroCreate,
    db: AsyncSession = Depends(get_db),
) -> Livro:
    """
    Cria um novo registro de livro no banco de dados.

    O fluxo é:
    1. O Pydantic valida os dados de entrada automaticamente (schema).
    2. Os dados validados são mapeados para o model SQLAlchemy.
    3. O objeto é persistido no banco via sessão assíncrona.
    4. O `db.refresh()` recarrega o objeto com o `id` gerado pelo banco.

    Args:
        livro_data: Dados do livro validados pelo schema LivroCreate.
        db: Sessão do banco injetada via Depends (dependency injection).

    Returns:
        O livro criado com todos os campos, incluindo o id gerado.
    """
    novo_livro = Livro(
        titulo=livro_data.titulo,
        autor=livro_data.autor,
        data_publicacao=livro_data.data_publicacao,
        resumo=livro_data.resumo,
    )

    db.add(novo_livro)
    await db.commit()
    await db.refresh(novo_livro)

    return novo_livro


# ---------------------------------------------------------------------------
# GET /livros/ — Listagem e busca de livros
# ---------------------------------------------------------------------------
@app.get(
    "/livros/",
    response_model=list[LivroResponse],
    summary="Listar e buscar livros",
    description=(
        "Retorna todos os livros cadastrados. Aceita parâmetros de query "
        "opcionais para filtrar por título e/ou autor (busca parcial, "
        "case-insensitive)."
    ),
)
async def listar_livros(
    titulo: Optional[str] = Query(
        default=None,
        description="Filtrar por título (busca parcial, case-insensitive).",
    ),
    autor: Optional[str] = Query(
        default=None,
        description="Filtrar por autor (busca parcial, case-insensitive).",
    ),
    db: AsyncSession = Depends(get_db),
) -> list[Livro]:
    """
    Lista livros com filtros opcionais por título e/ou autor.

    A busca utiliza `icontains` (LIKE case-insensitive) para permitir
    buscas parciais — por exemplo, buscar "python" retornará livros
    cujo título contenha "Python", "PYTHON" ou "python". Os filtros
    são cumulativos: se ambos forem fornecidos, aplica-se um AND lógico.

    Args:
        titulo: Termo parcial para busca no título (opcional).
        autor: Termo parcial para busca no autor (opcional).
        db: Sessão do banco injetada via Depends.

    Returns:
        Lista de livros que atendem aos critérios de busca.
    """
    query = select(Livro)

    if titulo:
        query = query.where(Livro.titulo.icontains(titulo))

    if autor:
        query = query.where(Livro.autor.icontains(autor))

    result = await db.execute(query)
    livros = result.scalars().all()

    return list(livros)
