# Avaliação Técnica — Python Backend com IA - DOT Group
## Vitor Bispo Braúna - Engenheiro de IA

Repositório contendo as 3 questões da avaliação técnica de Python, cada uma resolvida como um projeto independente com estrutura profissional, testes automatizados e documentação.

---

## 📁 Estrutura do Repositório

```
python-DOT-avaliacao/
├── questao-1-biblioteca/   # API RESTful com FastAPI de CRUD de livros
├── questao-2-chatbot/      # Chatbot CLI com LangChain + GPT-4.1
├── questao-3-busca/        # Busca Semântica com FAISS
└── README.md               # Este arquivo
```

---

## ⚙️ Requisitos

- **Python 3.10+**
- Cada questão possui seu próprio `requirements.txt` e ambiente virtual isolado

### Setup rápido (por questão)

```bash
cd questao-X-nome/
python -m venv .

# Windows
.\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📋 Visão Geral das Questões

### [Questão 1 — API RESTful para Biblioteca Virtual](questao-1-biblioteca/)

API assíncrona para gerenciamento de uma biblioteca virtual com CRUD de livros.

| Stack | Destaques |
|---|---|
| FastAPI, SQLAlchemy 2.0, aiosqlite, Pydantic v2 | Async do início ao fim, busca parcial case-insensitive, 17 testes unitários |

```bash
cd questao-1-biblioteca
python -m venv .
source ./Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload        # Swagger: http://127.0.0.1:8000/docs
python -m pytest tests/ -v           
```

📖 [Documentação completa (endpoints, testes, decisões arquiteturais)](questao-1-biblioteca/README.md)

---

### [Questão 2 — Chatbot Tutor de Python](questao-2-chatbot/)

Chatbot CLI que atua como tutor de Python, com histórico de conversação e suporte a múltiplos modelos.

| Stack | Destaques |
|---|---|
| LangChain (LCEL), GPT-4.1, HuggingFace | Pipeline declarativo (`Prompt \| LLM \| Parser`), troca transparente de modelo via flag `--oss` |

```bash
cd questao-2-chatbot
python -m venv .
source ./Scripts/activate
pip install -r requirements.txt
cp .env.example .env                 # Configurar chave de API (GitHub Models ou OpenAI e HuggingFace)
python chatbot.py                    # Modo GPT-4.1 (GitHub Models ou OpenAI)
# Ou
python chatbot.py --oss              # Modo GPT-OSS-20B (HuggingFace)
```
## Para o correto funcionamento desse chatbot é necessário:
- Chave de API do GitHub Models ou OpenAI
- Chave de API do HuggingFace (opcional, para o modo OSS)

📖 [Documentação completa (arquitetura LCEL, configuração de API, uso)](questao-2-chatbot/README.md)

---

### [Questão 3 — Busca Semântica Vetorial](questao-3-busca/)

Sistema de busca semântica em duas fases: ingestão em lote e consulta interativa com destaque de termos.

| Stack | Destaques |
|---|---|
| FAISS, SentenceTransformers, HuggingFace Datasets | 1.500 documentos reais, embeddings R³⁸⁴, busca por distância L2, 12 testes unitários |

```bash
cd questao-3-busca
pip install -r requirements.txt
python gerador_indice.py             # Gera índice (uma vez)
python buscar.py                     # Busca interativa
python -m pytest test_busca.py -v    # 12 testes
```

📖 [Documentação completa (lógica matemática, arquitetura, pipeline)](questao-3-busca/README.md)

---

## 🧪 Testes

Cada projeto possui sua suíte de testes independente:

| Questão | Testes | Comando |
|---|---|---|
| Q1 — Biblioteca | 17 (sucesso + falha + edge cases) | `python -m pytest tests/ -v` |
| Q3 — Busca | 12 (busca semântica + formatação) | `python -m pytest test_busca.py -v` |

---

## 📐 Padrões Adotados

- **Código em português** — nomes de variáveis, funções, classes e docstrings
- **Type hints** em todas as assinaturas de função
- **Docstrings** com blocos de Decisão Arquitetural explicando escolhas técnicas
- **PEP 8** — formatação, imports organizados, nomes descritivos
- **Separação de responsabilidades** — módulos especializados por domínio
- **Testes automatizados** — cobertura de cenários de sucesso e falha
- **Segurança** — chaves de API via `.env` (não versionado)
