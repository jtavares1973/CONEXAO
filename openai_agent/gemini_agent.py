# ============================================================
# gemini_agent.py
# Agente que usa o Google Gemini como provedor — drop-in
# replacement do AgenteOpenAI, com a mesma interface de skills.
#
# Por que usar este agente?
#   - Gemini 2.5 Pro é GRATUITO e state-of-the-art
#   - Contexto de 1 MILHÃO de tokens (documentos enormes!)
#   - Qualidade superior para tarefas de análise e raciocínio
#   - Suporta múltiplas keys para ainda mais capacidade
#
# Limitações do free tier:
#   - 2.5 Pro: apenas 5 RPM / 25 RPD (use Flash para alto volume)
#   - 2.5 Flash: 15 RPM / 500 RPD (bom equilíbrio)
#   - 2.0 Flash: 15 RPM / 1500 RPD (alto volume)
# ============================================================

import logging
from typing import Optional

from .gemini_provider import PoolGemini, GEMINI_MODELO_PADRAO
from .skills import (
    SkillBase, SkillChat, SkillResumo, SkillAnalise, SkillTradução,
)

logger = logging.getLogger(__name__)


class _ClienteGeminiStub:
    """
    Adaptador que faz o PoolGemini parecer um cliente OpenAI para as Skills.
    As Skills chamam client.chat.completions.create() — este stub
    redireciona para o PoolGemini com Round-Robin de keys.
    """

    def __init__(self, pool: PoolGemini):
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


class AgenteGemini:
    """
    Agente Gemini — mesma interface do AgenteOpenAI, usando o
    Google AI Studio como provedor (Gemini 2.5 Pro GRÁTIS!).

    Uso idêntico ao AgenteOpenAI:
        agente = AgenteGemini()                        # 2.5 Pro (padrão)
        agente = AgenteGemini(modelo="gemini-2.5-flash")  # mais volume

        resposta  = agente.executar("chat", "Olá!")
        resumo    = agente.executar("resumo", texto_longo)
        analise   = agente.executar("analise", dados, foco="riscos")
        traduzido = agente.executar("traducao", texto_en)

    Modelos recomendados:
        "gemini-2.5-pro"        → qualidade máxima (5 RPM / 25 RPD)
        "gemini-2.5-flash"      → equilíbrio (15 RPM / 500 RPD)
        "gemini-2.0-flash"      → alto volume (15 RPM / 1500 RPD)
        "gemini-2.5-flash-lite" → máximo volume (30 RPM / 1500 RPD)

    Com múltiplas keys no .env (GEMINI_API_KEY, GEMINI_API_KEY_2, ...),
    o Round-Robin é feito entre as keys automaticamente.
    """

    def __init__(self, modelo: str = GEMINI_MODELO_PADRAO):
        # Monta o pool lendo as keys do .env
        self._pool = PoolGemini.do_env(modelo=modelo)
        self.model_id = modelo

        # Adaptador que as Skills vão usar
        self._client_stub = _ClienteGeminiStub(self._pool)

        # Registra as skills padrão
        self._skills: dict[str, SkillBase] = {}
        self._registrar_skills()

        logger.info("AgenteGemini pronto | modelo=%s | keys=%d",
                    modelo, len(self._pool._pool))

    # ------------------------------------------------------------------
    # Interface pública — idêntica ao AgenteOpenAI
    # ------------------------------------------------------------------
    def executar(self, skill_nome: str, entrada: str, **kwargs) -> str:
        """Executa uma skill usando o Gemini como provedor."""
        if skill_nome not in self._skills:
            raise ValueError(
                f"Skill '{skill_nome}' não encontrada. "
                f"Disponíveis: {list(self._skills.keys())}"
            )
        skill = self._skills[skill_nome]
        skill.model_id = self.model_id
        return skill.executar(entrada, **kwargs)

    def status_pool(self) -> list[dict]:
        """Mostra status de cada key no pool."""
        return self._pool.status()

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _registrar_skills(self) -> None:
        """Registra todas as skills disponíveis."""
        stub = self._client_stub
        for nome, cls in [
            ("chat",     SkillChat),
            ("resumo",   SkillResumo),
            ("analise",  SkillAnalise),
            ("traducao", SkillTradução),
        ]:
            skill = cls(client=stub, model_id=self.model_id)  # type: ignore[arg-type]
            self._skills[nome] = skill
