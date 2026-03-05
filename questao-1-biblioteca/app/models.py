"""
models.py — Definição das entidades do banco de dados (ORM).

Decisão Arquitetural:
    Este módulo contém EXCLUSIVAMENTE as definições de tabelas do banco,
    separadas dos schemas Pydantic (schemas.py). Essa separação segue o
    princípio de que o model ORM representa a ESTRUTURA DE PERSISTÊNCIA,
    enquanto o schema Pydantic representa o CONTRATO DA API.

    Isso permite evoluir o banco (ex: adicionar índices, constraints) sem
    alterar a interface pública da API, e vice-versa — adicionando campos
    computados na resposta sem tocar no banco.
"""

import datetime

from sqlalchemy import Date, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Livro(Base):
    """
    Entidade que representa um livro na biblioteca virtual.

    Utiliza o estilo Mapped[] do SQLAlchemy 2.0+ para declaração de colunas,
    que oferece suporte nativo a Type Hints e melhor integração com IDEs
    e ferramentas de análise estática (mypy, pyright).
    """

    __tablename__ = "livros"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        doc="Identificador único do livro (chave primária).",
    )
    titulo: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Título do livro.",
    )
    autor: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Nome do autor do livro.",
    )
    data_publicacao: Mapped[datetime.date] = mapped_column(
        Date,
        nullable=False,
        doc="Data de publicação do livro.",
    )
    resumo: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Resumo ou sinopse do livro.",
    )

    def __repr__(self) -> str:
        return (
            f"Livro(id={self.id}, titulo='{self.titulo}', autor='{self.autor}')"
        )
