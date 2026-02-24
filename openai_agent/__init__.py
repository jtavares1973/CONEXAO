# ============================================================
# Pacote openai_agent
# Exporta os módulos principais para uso externo simplificado
# ============================================================

from .config import carregar_env as _carregar_env
_carregar_env()  # carrega .env automaticamente ao importar o pacote

from .model_selector import SeletorDeModelos
from .skills import SkillBase, SkillChat, SkillResumo, SkillAnalise, SkillTradução  # noqa: F401
from .agent import AgenteOpenAI
from .groq_provider import PoolGroq
from .groq_agent import AgenteGroq
from .gemini_provider import PoolGemini
from .gemini_agent import AgenteGemini
from .cerebras_provider import PoolCerebras
from .cerebras_agent import AgenteCerebras
from .sambanova_provider import PoolSambaNova
from .sambanova_agent import AgenteSambaNova

__all__ = [
    # Agente OpenRouter (padrão, modelos gratuitos via Round-Robin)
    "AgenteOpenAI",
    "SeletorDeModelos",
    # Agente Groq (30 RPM grátis — velocidade e volume)
    "AgenteGroq",
    "PoolGroq",
    # Agente Gemini (Google AI Studio — Gemini 2.5 Pro GRÁTIS!)
    "AgenteGemini",
    "PoolGemini",
    # Agente Cerebras (mais rápido do mundo — ~2.200 tok/s GRÁTIS)
    "AgenteCerebras",
    "PoolCerebras",
    # Agente SambaNova (DeepSeek V3.2, GPT-OSS 120B — GRÁTIS)
    "AgenteSambaNova",
    "PoolSambaNova",
    # Skills compartilhadas entre todos os agentes
    "SkillBase",
    "SkillChat",
    "SkillResumo",
    "SkillAnalise",
    "SkillTradução",
]
