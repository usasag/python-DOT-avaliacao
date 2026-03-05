"""
schemas.py — Modelos Pydantic para validação e serialização de dados.

Decisão Arquitetural:
    A separação entre `models.py` (banco de dados) e `schemas.py` (API) 
    evita vazamento de regras de persistência para os handlers HTTP. 
    Pydantic garante que dados recebidos sejam validados rapidamente e os
    dados retornados sejam seguramente tipados e formatados em JSON.
"""

import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class LivroCreate(BaseModel):
    """Schema de entrada para criação de um livro (POST /livros/)."""

    titulo: str
    autor: str
    data_publicacao: datetime.date
    resumo: str

    @field_validator("titulo", mode="before")
    @classmethod
    def titulo_nao_vazio(cls, valor: object) -> object:
        """Garante que o título não seja uma string vazia ou somente espaços."""
        if isinstance(valor, str) and not valor.strip():
            raise ValueError("O campo 'titulo' não pode ser vazio.")
        return valor

    @field_validator("autor", mode="before")
    @classmethod
    def autor_nao_vazio(cls, valor: object) -> object:
        """Garante que o autor não seja uma string vazia ou somente espaços."""
        if isinstance(valor, str) and not valor.strip():
            raise ValueError("O campo 'autor' não pode ser vazio.")
        return valor

    @field_validator("resumo", mode="before")
    @classmethod
    def resumo_nao_vazio(cls, valor: object) -> object:
        """Garante que o resumo não seja uma string vazia ou somente espaços."""
        if isinstance(valor, str) and not valor.strip():
            raise ValueError("O campo 'resumo' não pode ser vazio.")
        return valor


class LivroResponse(BaseModel):
    """Schema de saída para representação de um livro nas respostas da API."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    titulo: str
    autor: str
    data_publicacao: datetime.date
    resumo: str
