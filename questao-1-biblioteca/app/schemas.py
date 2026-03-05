"""
schemas.py — Schemas Pydantic para validação e serialização de dados.

Decisão Arquitetural:
    Este módulo define os "contratos" da API — as estruturas de dados que
    entram (request) e saem (response) dos endpoints. A separação entre
    schemas de entrada e saída é uma best practice que oferece:

    1. Segurança: campos como `id` não podem ser enviados pelo cliente.
    2. Flexibilidade: a resposta pode incluir campos computados sem
       impactar a entrada.
    3. Validação centralizada: todas as regras de negócio sobre formato
       de dados ficam aqui, e não espalhadas pelos endpoints.

    Utilizamos Pydantic v2 (BaseModel + ConfigDict) para serialização
    automática dos objetos SQLAlchemy via `from_attributes=True`.
"""

import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class LivroCreate(BaseModel):
    """
    Schema de entrada para criação de um livro (POST /livros/).

    Todos os campos são obrigatórios. Validações adicionais garantem
    que strings não sejam vazias, prevenindo registros sem conteúdo
    semântico no banco.
    """

    titulo: str
    autor: str
    data_publicacao: datetime.date
    resumo: str

    # -------------------------------------------------------------------
    # Validadores de campo
    # -------------------------------------------------------------------
    # Utilizamos `field_validator` com `mode="before"` para interceptar
    # os valores ANTES da coerção de tipo do Pydantic, garantindo que
    # strings vazias ou apenas com espaços sejam rejeitadas.
    # -------------------------------------------------------------------
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
    """
    Schema de saída para representação de um livro nas respostas da API.

    Inclui o campo `id` gerado pelo banco, que não está presente no
    schema de criação. O `from_attributes=True` permite que o Pydantic
    leia diretamente os atributos do objeto SQLAlchemy ORM, sem
    necessidade de conversão manual para dicionário.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    titulo: str
    autor: str
    data_publicacao: datetime.date
    resumo: str
