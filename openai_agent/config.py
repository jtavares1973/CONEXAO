# ============================================================
# config.py
# Carregamento inteligente de variáveis de ambiente.
#
# Ordem de busca pelo .env:
#   1. Diretório atual (cwd) e seus pais (até 4 níveis)
#   2. ~/.naccarona_agent/.env  ← configuração global do usuário
#   3. Diretório do próprio pacote (fallback)
#
# Isso permite que o pacote seja instalado via pip e as chaves
# fiquem em UM lugar só na máquina, disponíveis para qualquer projeto.
#
# Configuração inicial (faça UMA vez):
#   python -m openai_agent setup
# ============================================================

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Pasta global do pacote no perfil do usuário
PASTA_GLOBAL = Path.home() / ".naccarona_agent"
ENV_GLOBAL   = PASTA_GLOBAL / ".env"

# Caminho do diretório deste arquivo (o pacote instalado)
ENV_PACOTE = Path(__file__).parent / ".env"

_ja_carregou = False


def carregar_env(forcar: bool = False) -> Path | None:
    """
    Carrega variáveis de ambiente na ordem de prioridade.
    Retorna o Path do arquivo .env que foi carregado, ou None.
    Executa apenas uma vez por processo (memoizado), a menos que
    forcar=True.
    """
    global _ja_carregou
    if _ja_carregou and not forcar:
        return None
    _ja_carregou = True

    # 1. Busca .env no cwd e até 4 diretórios acima
    cwd = Path.cwd()
    for pasta in [cwd, *cwd.parents[:4]]:
        candidato = pasta / ".env"
        if candidato.exists():
            load_dotenv(candidato, override=False)
            logger.debug("openai_agent: .env carregado de %s", candidato)
            return candidato

    # 2. Configuração global do usuário (~/.naccarona_agent/.env)
    if ENV_GLOBAL.exists():
        load_dotenv(ENV_GLOBAL, override=False)
        logger.debug("openai_agent: .env global carregado de %s", ENV_GLOBAL)
        return ENV_GLOBAL

    # 3. Fallback: .env junto ao pacote
    if ENV_PACOTE.exists():
        load_dotenv(ENV_PACOTE, override=False)
        logger.debug("openai_agent: .env do pacote carregado de %s", ENV_PACOTE)
        return ENV_PACOTE

    logger.warning(
        "openai_agent: nenhum .env encontrado. "
        "Execute 'python -m openai_agent setup' para configurar."
    )
    return None


def caminho_env_global() -> Path:
    """Retorna o caminho do .env global (cria a pasta se necessário)."""
    PASTA_GLOBAL.mkdir(parents=True, exist_ok=True)
    return ENV_GLOBAL


def status_keys() -> dict:
    """
    Retorna um dicionário mostrando quais chaves estão configuradas.
    Útil para diagnóstico.
    """
    carregar_env()
    chaves = [
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "CEREBRAS_API_KEY",
        "SAMBANOVA_API_KEY",
    ]
    return {
        k: ("✅ configurada" if os.getenv(k, "").strip() else "❌ ausente")
        for k in chaves
    }
