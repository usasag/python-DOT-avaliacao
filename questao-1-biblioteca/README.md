# Questão 1 — API RESTful para Biblioteca Virtual

API RESTful desenvolvida com **FastAPI** para gerenciamento de uma biblioteca virtual, utilizando **SQLAlchemy assíncrono** com **SQLite** e validação de dados com **Pydantic**.

---

## 🛠️ Tecnologias

| Tecnologia | Finalidade |
|---|---|
| **Python 3.10+** | Linguagem principal |
| **FastAPI** | Framework web assíncrono |
| **SQLAlchemy 2.0+** | ORM com suporte assíncrono (`Mapped[]`) |
| **aiosqlite** | Driver assíncrono para SQLite |
| **Pydantic v2** | Validação e serialização de dados |
| **pytest + pytest-asyncio** | Testes unitários assíncronos |
| **httpx** | Client HTTP para testes (ASGI) |

---

## 📁 Estrutura do Projeto

```
questao-1-biblioteca/
├── app/
│   ├── __init__.py        # Pacote da aplicação
│   ├── database.py        # Configuração do banco (engine, sessão, Base)
│   ├── models.py          # Entidade Livro (SQLAlchemy ORM)
│   ├── schemas.py         # Schemas Pydantic (validação entrada/saída)
│   └── main.py            # Aplicação FastAPI + endpoints
├── tests/
│   ├── __init__.py
│   └── test_livros.py     # 17 testes unitários (sucesso + falha)
├── requirements.txt       # Dependências do projeto
├── .gitignore
└── README.md
```

### Separação de Responsabilidades

- **`database.py`** — Centraliza toda a infraestrutura de acesso a dados (engine, sessão, Base declarativa, dependency injection). Isso garante que mudanças de banco sejam feitas em um único lugar.
- **`models.py`** — Define exclusivamente a estrutura de persistência (tabelas ORM), separada da interface da API.
- **`schemas.py`** — Define os contratos da API (entrada/saída) com validações de negócio. Separar schemas de models permite evoluir banco e API independentemente.
- **`main.py`** — Contém apenas a configuração do FastAPI e as rotas. Toda lógica de validação e persistência é delegada aos módulos especializados.

---

## 🚀 Como Executar

### 1. Clonar e acessar o projeto

```bash
git clone <url-do-repositorio>
cd questao-1-biblioteca
```

### 2. Criar e ativar o ambiente virtual

```bash
# Criar
python -m venv venv

# Ativar (Windows)
.\venv\Scripts\activate

# Ativar (Linux/macOS)
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Iniciar o servidor

```bash
uvicorn app.main:app --reload
```

A API estará disponível em: **http://127.0.0.1:8000**

Documentação interativa (Swagger UI): **http://127.0.0.1:8000/docs**

---

## 📌 Endpoints

### `POST /livros/`

Cadastra um novo livro.

**Request Body:**
```json
{
  "titulo": "Clean Code",
  "autor": "Robert C. Martin",
  "data_publicacao": "2008-08-01",
  "resumo": "Guia de boas práticas de programação."
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "titulo": "Clean Code",
  "autor": "Robert C. Martin",
  "data_publicacao": "2008-08-01",
  "resumo": "Guia de boas práticas de programação."
}
```

### `GET /livros/`

Lista todos os livros. Aceita filtros opcionais por query string.

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `titulo` | `string` (opcional) | Busca parcial, case-insensitive |
| `autor` | `string` (opcional) | Busca parcial, case-insensitive |

**Exemplos:**
```
GET /livros/                          → Lista todos
GET /livros/?titulo=python            → Filtra por título
GET /livros/?autor=martin             → Filtra por autor
GET /livros/?titulo=clean&autor=martin → Filtro combinado (AND)
```

---

## 🧪 Testes

Executar todos os testes:

```bash
python -m pytest tests/ -v
```

### Cobertura dos Testes (17 testes)

| Categoria | Cenários |
|---|---|
| ✅ **Sucesso** | Criar livro, listar vazio, listar após cadastro, buscar por título, buscar por autor, busca combinada, busca sem resultados, campo extra ignorado |
| ❌ **Falha** | Título ausente, autor ausente, data ausente, resumo ausente, data inválida, título vazio, autor vazio, resumo vazio, payload vazio |

---

## 📐 Decisões Arquiteturais

1. **SQLAlchemy Assíncrono**: Utilizado `AsyncSession` + `aiosqlite` para manter compatibilidade total com o modelo async/await do FastAPI, evitando bloqueio de I/O.

2. **Pydantic v2 com `from_attributes`**: Permite serialização direta dos objetos ORM sem conversão manual para dicionário.

3. **Dependency Injection (`Depends`)**: A sessão do banco é injetada via `get_db()`, facilitando a substituição nos testes por um banco in-memory.

4. **`Mapped[]` (SQLAlchemy 2.0+)**: Estilo de declaração com type hints nativos, melhorando integração com IDEs e ferramentas de análise estática.

5. **`lifespan` ao invés de `on_event`**: Padrão recomendado pelo FastAPI/Starlette para gerenciar ciclo de vida da aplicação.

6. **Validação com `field_validator`**: Strings vazias ou com apenas espaços são rejeitadas nos campos `titulo`, `autor` e `resumo`.

7. **Testes com banco in-memory**: Cada teste cria e destrói tabelas, garantindo isolamento total sem efeitos colaterais.
