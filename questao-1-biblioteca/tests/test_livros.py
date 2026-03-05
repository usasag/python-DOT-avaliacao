"""
test_livros.py — Testes unitários para a API de Biblioteca Virtual.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.database import Base, get_db
from app.main import app


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)

async_session_testing = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)



@pytest_asyncio.fixture
async def client():
    """Fixture que fornece um AsyncClient configurado com banco in-memory."""

    # Cria as tabelas
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Sobrescreve a dependência do banco
    async def override_get_db():
        async with async_session_testing() as session:
            try:
                yield session
            finally:
                await session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Fornece o client assíncrono
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    # Limpeza: remove overrides e destrói tabelas
    app.dependency_overrides.clear()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ===========================================================================
#                        TESTES DE SUCESSO
# ===========================================================================


@pytest.mark.asyncio
async def test_criar_livro_sucesso(client: AsyncClient) -> None:
    """Testa a criação de um livro com todos os campos válidos."""
    payload = {
        "titulo": "O Senhor dos Anéis",
        "autor": "J.R.R. Tolkien",
        "data_publicacao": "1954-07-29",
        "resumo": "Uma épica jornada pela Terra Média.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["titulo"] == payload["titulo"]
    assert data["autor"] == payload["autor"]
    assert data["data_publicacao"] == payload["data_publicacao"]
    assert data["resumo"] == payload["resumo"]
    assert "id" in data
    assert isinstance(data["id"], int)


@pytest.mark.asyncio
async def test_listar_livros_vazio(client: AsyncClient) -> None:
    """Testa que a listagem retorna lista vazia quando não há livros."""
    response = await client.get("/livros/")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_listar_livros_apos_cadastro(client: AsyncClient) -> None:
    """Testa que livros cadastrados aparecem na listagem."""
    livro_1 = {
        "titulo": "Dom Casmurro",
        "autor": "Machado de Assis",
        "data_publicacao": "1899-01-01",
        "resumo": "Romance que explora ciúme e dúvida.",
    }
    livro_2 = {
        "titulo": "Grande Sertão: Veredas",
        "autor": "Guimarães Rosa",
        "data_publicacao": "1956-05-01",
        "resumo": "Narrativa épica do sertão brasileiro.",
    }

    await client.post("/livros/", json=livro_1)
    await client.post("/livros/", json=livro_2)

    response = await client.get("/livros/")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


@pytest.mark.asyncio
async def test_buscar_por_titulo(client: AsyncClient) -> None:
    """Testa a busca parcial e case-insensitive por título."""
    livro_1 = {
        "titulo": "Clean Code",
        "autor": "Robert C. Martin",
        "data_publicacao": "2008-08-01",
        "resumo": "Guia de boas práticas de programação.",
    }
    livro_2 = {
        "titulo": "O Hobbit",
        "autor": "J.R.R. Tolkien",
        "data_publicacao": "1937-09-21",
        "resumo": "A aventura de Bilbo Bolseiro.",
    }

    await client.post("/livros/", json=livro_1)
    await client.post("/livros/", json=livro_2)

    # Busca parcial e case-insensitive
    response = await client.get("/livros/", params={"titulo": "clean"})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["titulo"] == "Clean Code"


@pytest.mark.asyncio
async def test_buscar_por_autor(client: AsyncClient) -> None:
    """Testa a busca parcial e case-insensitive por autor."""
    livro_1 = {
        "titulo": "1984",
        "autor": "George Orwell",
        "data_publicacao": "1949-06-08",
        "resumo": "Distopia sobre vigilância totalitária.",
    }
    livro_2 = {
        "titulo": "A Revolução dos Bichos",
        "autor": "George Orwell",
        "data_publicacao": "1945-08-17",
        "resumo": "Fábula política satírica.",
    }
    livro_3 = {
        "titulo": "Sapiens",
        "autor": "Yuval Noah Harari",
        "data_publicacao": "2011-01-01",
        "resumo": "História da humanidade.",
    }

    await client.post("/livros/", json=livro_1)
    await client.post("/livros/", json=livro_2)
    await client.post("/livros/", json=livro_3)

    response = await client.get("/livros/", params={"autor": "orwell"})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(
        "Orwell" in livro["autor"] for livro in data
    )


@pytest.mark.asyncio
async def test_buscar_por_titulo_e_autor(client: AsyncClient) -> None:
    """Testa a busca combinada por título E autor (AND lógico)."""
    livro_1 = {
        "titulo": "Python Fluente",
        "autor": "Luciano Ramalho",
        "data_publicacao": "2015-08-01",
        "resumo": "Guia avançado de Python.",
    }
    livro_2 = {
        "titulo": "Python Cookbook",
        "autor": "David Beazley",
        "data_publicacao": "2013-05-01",
        "resumo": "Receitas práticas de Python.",
    }

    await client.post("/livros/", json=livro_1)
    await client.post("/livros/", json=livro_2)

    # Ambos têm "Python" no título, mas filtrando por autor "Ramalho"
    response = await client.get(
        "/livros/", params={"titulo": "Python", "autor": "Ramalho"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["autor"] == "Luciano Ramalho"


@pytest.mark.asyncio
async def test_buscar_sem_resultados(client: AsyncClient) -> None:
    """Testa que uma busca sem correspondência retorna lista vazia."""
    livro = {
        "titulo": "O Alquimista",
        "autor": "Paulo Coelho",
        "data_publicacao": "1988-01-01",
        "resumo": "Jornada de autodescoberta.",
    }
    await client.post("/livros/", json=livro)

    response = await client.get(
        "/livros/", params={"titulo": "inexistente"}
    )

    assert response.status_code == 200
    assert response.json() == []


# ===========================================================================
#                        TESTES DE FALHA
# ===========================================================================


@pytest.mark.asyncio
async def test_criar_livro_sem_titulo(client: AsyncClient) -> None:
    """Testa que a criação falha quando o título está ausente."""
    payload = {
        "autor": "Autor Teste",
        "data_publicacao": "2023-01-01",
        "resumo": "Resumo do livro.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.asyncio
async def test_criar_livro_sem_autor(client: AsyncClient) -> None:
    """Testa que a criação falha quando o autor está ausente."""
    payload = {
        "titulo": "Livro Teste",
        "data_publicacao": "2023-01-01",
        "resumo": "Resumo do livro.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_sem_data_publicacao(client: AsyncClient) -> None:
    """Testa que a criação falha quando a data de publicação está ausente."""
    payload = {
        "titulo": "Livro Teste",
        "autor": "Autor Teste",
        "resumo": "Resumo do livro.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_sem_resumo(client: AsyncClient) -> None:
    """Testa que a criação falha quando o resumo está ausente."""
    payload = {
        "titulo": "Livro Teste",
        "autor": "Autor Teste",
        "data_publicacao": "2023-01-01",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_data_formato_invalido(client: AsyncClient) -> None:
    """Testa que a criação falha com data em formato inválido."""
    payload = {
        "titulo": "Livro Teste",
        "autor": "Autor Teste",
        "data_publicacao": "data-invalida",
        "resumo": "Resumo do livro.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_titulo_vazio(client: AsyncClient) -> None:
    """Testa que a criação falha quando o título é uma string vazia."""
    payload = {
        "titulo": "   ",
        "autor": "Autor Teste",
        "data_publicacao": "2023-01-01",
        "resumo": "Resumo do livro.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_autor_vazio(client: AsyncClient) -> None:
    """Testa que a criação falha quando o autor é uma string vazia."""
    payload = {
        "titulo": "Livro Teste",
        "autor": "",
        "data_publicacao": "2023-01-01",
        "resumo": "Resumo do livro.",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_resumo_vazio(client: AsyncClient) -> None:
    """Testa que a criação falha quando o resumo é uma string vazia."""
    payload = {
        "titulo": "Livro Teste",
        "autor": "Autor Teste",
        "data_publicacao": "2023-01-01",
        "resumo": "",
    }

    response = await client.post("/livros/", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_payload_vazio(client: AsyncClient) -> None:
    """Testa que a criação falha quando o payload está completamente vazio."""
    response = await client.post("/livros/", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_criar_livro_campo_extra_ignorado(client: AsyncClient) -> None:
    """Testa que campos extras no payload são ignorados (Pydantic strict)."""
    payload = {
        "titulo": "Livro Com Extra",
        "autor": "Autor Teste",
        "data_publicacao": "2023-01-01",
        "resumo": "Resumo do livro.",
        "campo_inexistente": "valor qualquer",
    }

    response = await client.post("/livros/", json=payload)

    # Deve criar normalmente, ignorando o campo extra
    assert response.status_code == 201
    data = response.json()
    assert "campo_inexistente" not in data
