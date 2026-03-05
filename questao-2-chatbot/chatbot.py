"""
chatbot.py — Chatbot Tutor de Python com LangChain e GPT-4.1/OSS.

A tarefa pede o modelo GPT-4, mas esse modelo não está mais disponível, sendo necessário o uso do GPT-4.1.
O modelo OSS foi implementado também para maior variação de modelo, pois é boa prática ter fallbacks. Nesse caso só existe um, mas geralmente uso em torno de 4-5.

Decisão Arquitetural:
    Este script foi projetado para atuar como um chatbot CLI de tutoria Python.
    A arquitetura utiliza o padrão LangChain Expression Language (LCEL) com 
    o operador pipe (`|`) para encadear os componentes (`Prompt | LLM | Parser`).
    
    Abstração de Modelo: A troca entre modelos pagos (GPT-4.1 via GitHub) e 
    open-source (via HuggingFace) ocorre de forma transparente (na função `build_llm`),
    sem modificações na cadeia central, provando o baixo acoplamento da solução LCEL.

Flags:
    --oss : Usa modelo open-source via HuggingFace (requer HUGGINGFACEHUB_API_TOKEN)
    (default): Usa GPT-4.1 via GitHub Models ou GPT-4.1 via OpenAI (requer LLM_API_KEY)

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
# Prompt de sistema para configurar o comportamento do tutor
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
# Histórico de Conversação — Armazenamento em Memória
# ---------------------------------------------------------------------------
# Implementamos um store simples baseado em dicionário para armazenar
# os históricos por session_id. O `RunnableWithMessageHistory` do LCEL
# utiliza essa função factory para obter/criar o histórico de cada sessão.
#
# Em produção, isso seria substituído por um backend persistente (Redis,
# banco de dados, etc.), mas para este chatbot CLI, o armazenamento
# in-memory é suficiente.
# ---------------------------------------------------------------------------
class InMemoryChatHistory(BaseChatMessageHistory):
    """Implementação simples de histórico de chat in-memory."""

    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        """Adiciona uma mensagem ao histórico."""
        self.messages.append(message)

    def clear(self) -> None:
        """Limpa todo o histórico de mensagens."""
        self.messages = []


# Store global de sessões de chat
_session_store: dict[str, InMemoryChatHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Factory function que retorna o histórico de uma sessão.

    O `RunnableWithMessageHistory` chama esta função a cada invocação
    da chain, passando o `session_id` para recuperar (ou criar) o
    histórico correspondente. Isso permite múltiplas sessões simultâneas.

    Args:
        session_id: Identificador único da sessão de conversa.

    Returns:
        Instância de BaseChatMessageHistory para a sessão solicitada.
    """
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatHistory()
    return _session_store[session_id]


# ---------------------------------------------------------------------------
# Construção do Modelo LLM
# ---------------------------------------------------------------------------
def build_llm(use_oss: bool = False) -> Any:
    """
    Constrói o modelo de linguagem com base na flag --oss.

    Args:
        use_oss: Se True, usa modelo open-source via HuggingFace.
                 Se False, usa GPT-4 da OpenAI.

    Returns:
        Instância do modelo LLM configurado.

    Decisão Arquitetural:
        A abstração do LangChain permite trocar o modelo (OpenAI, HuggingFace,
        Anthropic, etc.) sem alterar nenhuma outra parte do código — o prompt,
        o parser e o histórico permanecem idênticos. Isso demonstra o poder
        do LCEL: cada componente é um Runnable intercambiável.
    """
    if use_oss:
        # -----------------------------------------------------------------
        # Modo OSS: HuggingFace Inference API
        # -----------------------------------------------------------------
        # Usa a Inference API do HuggingFace, que permite acessar modelos
        # open-source hospedados sem necessidade de GPU local. A chave
        # HUGGINGFACEHUB_API_TOKEN é lida automaticamente pelo wrapper.
        # -----------------------------------------------------------------
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-120b",
            temperature=0.7,
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=llm)
    else:
        # -----------------------------------------------------------------
        # Modo padrão: Dinâmico (GitHub Models GPT-4.1 ou OpenAI GPT-4)
        # -----------------------------------------------------------------
        # Lê a chave genérica e identifica o provedor pelo prefixo
        # -----------------------------------------------------------------
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
# Construção da Chain LCEL
# ---------------------------------------------------------------------------
def build_chain(use_oss: bool = False) -> RunnableWithMessageHistory:
    """
    Constrói a cadeia LCEL completa com histórico de conversação.

    Arquitetura LCEL (LangChain Expression Language):
        A chain é composta por 3 componentes conectados via operador pipe:

        ┌───────────────────┐    ┌──────────┐   ┌─────────────────┐
        │ ChatPromptTemplate│ →  │ChatOpenAI│ → │ StrOutputParser │
        │  (System + History│    │ (GPT-4)  │   │ (extrai string) │
        │   + User Input)   │    │          │   │                 │
        └───────────────────┘    └──────────┘   └─────────────────┘

        1. ChatPromptTemplate: Monta o prompt com 3 seções:
           - SystemMessage: Define o comportamento do tutor (SYSTEM_PROMPT)
           - MessagesPlaceholder: Slot onde o histórico é injetado
           - HumanMessage: A pergunta atual do usuário

        2. ChatOpenAI / ChatHuggingFace: Envia o prompt ao LLM e recebe
           a resposta como um objeto AIMessage.

        3. StrOutputParser: Extrai apenas o conteúdo textual (string)
           do AIMessage, simplificando o uso no CLI.

        O `RunnableWithMessageHistory` envolve essa chain base e:
        - ANTES de cada invocação: injeta o histórico no MessagesPlaceholder
        - APÓS cada invocação: salva a pergunta e resposta no histórico

    Returns:
        RunnableWithMessageHistory pronto para ser invocado com `.invoke()`.
    """
    # -----------------------------------------------------------------
    # 1. Template do Prompt
    # -----------------------------------------------------------------
    # `MessagesPlaceholder("history")` é o slot dinâmico onde o
    # RunnableWithMessageHistory injetará as mensagens anteriores.
    # A variável `input` recebe a pergunta atual do usuário.
    # -----------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    llm = build_llm(use_oss)
    chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


# ---------------------------------------------------------------------------
# Interface CLI (Command Line Interface)
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

    # Indica qual modelo está em uso
    if args.oss:
        modelo_nome = "HuggingFace (openai/gpt-oss-120b)"
    else:
        import os
        api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("GITHUB_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
        if api_key.lower().startswith(("ghp_", "gho_", "github_pat_")):
            modelo_nome = "GitHub Models (GPT-4.1)"
        else:
            modelo_nome = "OpenAI (GPT-4)"

    print("=" * 60)
    print("🐍  Chatbot Tutor de Python")
    print(f"    Modelo: {modelo_nome}")
    print("=" * 60)
    print("Faça suas perguntas sobre Python!")
    print("Digite 'sair' ou 'exit' para encerrar.\n")

    # Constrói a chain LCEL com histórico
    chain_with_history = build_chain(use_oss=args.oss)

    # Configuração de sessão — em um CLI single-user, usamos um ID fixo.
    # Em uma aplicação multi-usuário, cada usuário teria seu próprio ID.
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
            # ---------------------------------------------------------
            # Invocação da Chain
            # ---------------------------------------------------------
            # O `.invoke()` executa toda a cadeia LCEL:
            # 1. get_session_history("cli-session") → recupera histórico
            # 2. Prompt monta: System + histórico + pergunta atual
            # 3. LLM processa e retorna AIMessage
            # 4. StrOutputParser extrai o texto
            # 5. Histórico é atualizado com pergunta + resposta
            # ---------------------------------------------------------
            # Indicador visual de carregamento
            print("⏳ Pensando...", end="\r", flush=True)

            resposta = chain_with_history.invoke(
                {"input": pergunta},
                config=session_config,
            )
            
            # Limpa o indicador (escreve espaços por cima) e volta ao início
            print(" " * 20, end="\r", flush=True)
            print(f"\n🤖 Tutor: {resposta}\n")

        except Exception as e:
            # Tratamento genérico para erros de API (rate limit, auth, etc.)
            print(f"\n❌ Erro ao processar sua pergunta: {e}")
            print("   Verifique sua chave de API e conexão com a internet.\n")



if __name__ == "__main__":
    main()
