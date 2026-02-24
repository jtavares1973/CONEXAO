# ============================================================
# groq_agent.py
# Agente que usa o Groq como provedor — drop-in replacement
# do AgenteOpenAI, com a mesma interface de skills.
#
# Use este quando precisar processar grandes volumes:
#   - 30 req/min (vs 8 do OpenRouter)
#   - 14.400 req/dia com llama-3.1-8b-instant
#   - Suporta múltiplas keys para ainda mais capacidade
# ============================================================

import logging
from typing import Optional, Type

from .groq_provider import PoolGroq, GROQ_MODELO_PADRAO
from .skills import (
    SkillBase, SkillChat, SkillResumo, SkillAnalise, SkillTradução,
)
# Importa o cliente OpenAI para criar um stub compatível com SkillBase
from openai import OpenAI

logger = logging.getLogger(__name__)


class _ClienteGroqStub:
    """
    Adaptador que faz o PoolGroq parecer um cliente OpenAI para as Skills.
    As Skills chamam client.chat.completions.create() — este stub
    redireciona para o PoolGroq com Round-Robin de keys.
    """

    def __init__(self, pool: PoolGroq):
        self._pool = pool
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list, max_tokens: int = 2048,
               temperature: float = 0.7, **kwargs):
        # Redireciona para o pool com Round-Robin
        conteudo, modelo_usado = self._pool.completar(
            mensagens=messages,
            max_tokens=max_tokens,
            temperatura=temperature,
        )
        # Retorna objeto com a mesma estrutura que o SDK openai retornaria
        return _RespostaFake(conteudo, modelo_usado)


class _RespostaFake:
    """Imita a estrutura de resposta do SDK openai."""
    def __init__(self, conteudo: str, modelo: str):
        self.choices = [_ChoiceFake(conteudo)]
        self.usage = None
        self.model = modelo


class _ChoiceFake:
    def __init__(self, conteudo: str):
        self.message = _MensagemFake(conteudo)


class _MensagemFake:
    def __init__(self, conteudo: str):
        self.content = conteudo


class AgenteGroq:
    """
    Agente Groq — mesma interface do AgenteOpenAI, porém usando
    o Groq como provedor (30 req/min grátis, muito mais rápido).

    Uso idêntico ao AgenteOpenAI:
        agente = AgenteGroq()
        resposta = agente.executar("chat", "Olá!")
        resumo   = agente.executar("resumo", texto_longo)
        analise  = agente.executar("analise", dados, foco="riscos")
        traduzido = agente.executar("traducao", texto_en)

    Com múltiplas keys no .env (GROQ_API_KEY, GROQ_API_KEY_2, ...),
    o Round-Robin é feito entre as keys automaticamente.
    """

    def __init__(
        self,
        modelo: str = GROQ_MODELO_PADRAO,
        keys_extras: Optional[list[str]] = None,
    ):
        # Monta o pool lendo as keys do .env
        self._pool = PoolGroq.do_env(modelo=modelo)
        self.model_id = modelo

        # Cria o adaptador que as Skills vão usar
        self._client_stub = _ClienteGroqStub(self._pool)

        # Registra as skills padrão
        self._skills: dict[str, SkillBase] = {}
        self._registrar_skills()

        logger.info("AgenteGroq pronto | modelo=%s | keys=%d",
                    modelo, len(self._pool._pool))

    # ------------------------------------------------------------------
    # Interface pública — idêntica ao AgenteOpenAI
    # ------------------------------------------------------------------
    def executar(self, skill_nome: str, entrada: str, **kwargs) -> str:
        """Executa uma skill usando o Groq como provedor."""
        if skill_nome not in self._skills:
            raise ValueError(
                f"Skill '{skill_nome}' não encontrada. "
                f"Disponíveis: {list(self._skills.keys())}"
            )
        skill = self._skills[skill_nome]
        skill.model_id = self.model_id
        resultado = skill.executar(entrada, **kwargs)
        # Atualiza model_id com o que foi realmente usado
        self.model_id = getattr(self._client_stub._pool._pool[0], 'modelo', self.model_id)
        return resultado

    def status_pool(self) -> list[dict]:
        """Mostra status de cada key no pool."""
        return self._pool.status()

    def skills_disponiveis(self) -> list[str]:
        return list(self._skills.keys())

    def registrar_skill(self, nome: str, skill_classe: Type[SkillBase], **kwargs) -> None:
        self._skills[nome] = skill_classe(self._client_stub, self.model_id, **kwargs)
        logger.info("Skill registrada: %s", nome)

    # ------------------------------------------------------------------
    def _registrar_skills(self) -> None:
        self._skills = {
            "chat":     SkillChat(self._client_stub, self.model_id),
            "resumo":   SkillResumo(self._client_stub, self.model_id),
            "analise":  SkillAnalise(self._client_stub, self.model_id),
            "traducao": SkillTradução(self._client_stub, self.model_id),
        }
