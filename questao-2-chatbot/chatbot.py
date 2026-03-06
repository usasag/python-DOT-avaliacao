"""
chatbot.py — Chatbot Tutor de Python com LangChain e GPT-4.1/OSS.

A tarefa pede o modelo GPT-4, mas esse modelo não está mais disponível, sendo necessário o uso do GPT-4.1.
O modelo OSS foi implementado também para maior variação de modelo, pois é boa prática ter fallbacks. Nesse caso só existe um, mas geralmente uso em torno de 4-5.

Decisão Arquitetural:
    Este script foi projetado para atuar como um chatbot CLI de tutoria Python.
    A arquitetura utiliza o padrão LangChain Expression Language (LCEL) com
    o operador pipe (`|`) para encadear os componentes (Prompt | LLM | Parser).

    A troca entre modelos pagos (GPT-4.1 via GitHub/OpenAI) e open-source
    (via HuggingFace) ocorre de forma transparente na função `build_llm`,
    sem modificações na cadeia central, provando o baixo acoplamento do LCEL.

Flags:
    --oss : Usa modelo open-source via HuggingFace (requer HUGGINGFACEHUB_API_TOKEN)
    (default): Usa GPT-4.1 via GitHub Models ou OpenAI (requer LLM_API_KEY)

Uso:
    python chatbot.py         # Modo LLM Padrão (GH Models / OpenAI)
    python chatbot.py --oss   # Modo HuggingFace (modelo OSS)
"""

import argparse
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()


# ---------------------------------------------------------------------------
# System Prompt — restringe o chatbot ao escopo de tutoria Python
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """Você é um tutor especialista em Python altamente \
qualificado. Seu papel é ajudar programadores de todos os níveis a entender \
e dominar a linguagem Python.

Regras de comportamento:
1. Responda EXCLUSIVAMENTE perguntas relacionadas a Python (sintaxe, \
bibliotecas, boas práticas, padrões de projeto, ferramentas do ecossistema).
2. Se o usuário fizer uma pergunta fora do escopo de Python, recuse \
educadamente e redirecione a conversa para Python.
3. Use exemplos de código quando apropriado para ilustrar conceitos.
4. Adapte o nível da explicação ao contexto da pergunta.
5. Quando relevante, mencione boas práticas (PEP 8, type hints, etc.).
6. Responda sempre em português brasileiro.
"""


# ---------------------------------------------------------------------------
# Histórico de Conversação — armazenamento in-memory por sessão
# ---------------------------------------------------------------------------
class InMemoryChatHistory(BaseChatMessageHistory):
    """Armazena mensagens em lista Python, indexadas por session_id."""

    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        """Adiciona uma mensagem ao histórico."""
        self.messages.append(message)

    def clear(self) -> None:
        """Limpa todo o histórico de mensagens."""
        self.messages = []


_session_store: dict[str, InMemoryChatHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retorna (ou cria) o histórico da sessão indicada."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatHistory()
    return _session_store[session_id]


# ---------------------------------------------------------------------------
# Construção do Modelo LLM
# ---------------------------------------------------------------------------
def build_llm(use_oss: bool = False) -> Any:
    """
    Instancia o modelo de linguagem com base na flag --oss.

    A abstração do LangChain permite trocar o provedor (OpenAI, HuggingFace,
    Github Models, etc.) sem alterar o restante da cadeia LCEL.
    """
    if use_oss:
        # Modo OSS: HuggingFace Inference API (modelo extremamente barato, com limite de uso para testes, sem GPU local)
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            temperature=0.7,
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=llm)
    else:
        # Detecta automaticamente o provedor pela chave de API
        import os
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("LLM_API_KEY", "")

        if api_key.lower().startswith(("ghp_", "gho_", "github_pat_")):
            return ChatOpenAI(
                model="openai/gpt-4.1",
                temperature=0.7,
                base_url="https://models.github.ai/inference",
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                api_key=api_key,
            )


# ---------------------------------------------------------------------------
# Construção da Chain LCEL (Prompt | LLM | Parser) + Histórico
# ---------------------------------------------------------------------------
def build_chain(use_oss: bool = False) -> RunnableWithMessageHistory:
    """Monta a cadeia LCEL completa com injeção automática de histórico."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    llm = build_llm(use_oss)

    # Composição LCEL: cada `|` conecta a saída de um Runnable à entrada do próximo
    chain = prompt | llm | StrOutputParser()

    # RunnableWithMessageHistory injeta o histórico antes e salva após cada invocação
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Chatbot Tutor de Python — LangChain + GPT-4",
    )
    parser.add_argument(
        "--oss",
        action="store_true",
        help=(
            "Usa modelo open-source via HuggingFace ao invés do GPT-4. "
            "Requer HUGGINGFACEHUB_API_TOKEN no .env."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Loop principal do chatbot CLI.

    Fluxo:
    1. Constrói a chain LCEL com ou sem flag --oss.
    2. Entra em loop infinito aguardando input do usuário.
    3. A cada pergunta, invoca a chain passando o session_id para
       recuperar/atualizar o histórico automaticamente.
    4. Comandos "sair" e "exit" encerram o programa.
    """
    args = parse_args()

    # Identifica qual modelo será usado para exibição ao usuário
    if args.oss:
        modelo_nome = "HuggingFace (openai/gpt-oss-20b)"
    else:
        import os
        api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("GITHUB_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
        if api_key.lower().startswith(("ghp_", "gho_", "github_pat_")):
            modelo_nome = "GitHub Models (GPT-4.1)"
        else:
            modelo_nome = "OpenAI (GPT-4)"

    print("=" * 60) # Para melhor visibilidade
    print("🐍  Chatbot Tutor de Python")
    print(f"    Modelo: {modelo_nome}")
    print("=" * 60)
    print("Faça suas perguntas sobre Python!")
    print("Digite 'sair' ou 'exit' para encerrar.\n")

    chain_with_history = build_chain(use_oss=args.oss)

    # Em CLI single-user usamos um session_id fixo
    session_config = {"configurable": {"session_id": "cli-session"}}

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C ou Ctrl+D encerra graciosamente
            print("\n\n👋 Até mais! Bons estudos em Python!")
            sys.exit(0)

        # Ignora entradas vazias
        if not pergunta:
            continue

        # Comandos de saída
        if pergunta.lower() in ("sair", "exit"):
            print("\n👋 Até mais! Bons estudos em Python!")
            break

        try:
            print("⏳ Pensando...", end="\r", flush=True)

            resposta = chain_with_history.invoke(
                {"input": pergunta},
                config=session_config,
            )

            print(" " * 20, end="\r", flush=True)
            print(f"\n🤖 Tutor: {resposta}\n")

        except Exception as e:
            print(f"\n❌ Erro ao processar sua pergunta: {e}")
            print("   Verifique sua chave de API e conexão com a internet.\n")


if __name__ == "__main__":
    main()
