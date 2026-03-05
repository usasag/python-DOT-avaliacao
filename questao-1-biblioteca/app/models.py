"""
models.py — Entidades do ORM SQLAlchemy.

Decisão Arquitetural:
    Adotamos SQLAlchemy 2.0 com `Mapped` e `mapped_column` para aproveitar
    ao máximo as dicas de tipo estáticas (type hints) nativas do Python.
    Isso melhora a segurança e a integração com o Type Checker da IDE.
"""

import datetime

from sqlalchemy import Date, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Livro(Base):
    """Entidade que representa um livro na biblioteca virtual."""

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
