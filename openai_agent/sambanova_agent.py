# ============================================================
# sambanova_agent.py
# Agente SambaNova — modelos top open-source gratuitos.
# Drop-in replacement do AgenteOpenAI com a mesma interface.
# ============================================================

import logging
from .sambanova_provider import PoolSambaNova, SAMBANOVA_MODELO_PADRAO
from .skills import SkillBase, SkillChat, SkillResumo, SkillAnalise, SkillTradução

logger = logging.getLogger(__name__)


class _ClienteSambaNovaStub:
    def __init__(self, pool: PoolSambaNova):
        self._pool = pool
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: list, max_tokens: int = 2048,
               temperature: float = 0.7, **kwargs):
        conteudo, modelo_usado = self._pool.completar(
            mensagens=messages, max_tokens=max_tokens, temperatura=temperature,
        )
        return _RespostaFake(conteudo, modelo_usado)


class _RespostaFake:
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


class AgenteSambaNova:
    """
    Agente SambaNova — acesso gratuito a modelos de ponta.
    Mesma interface do AgenteOpenAI.

    Modelos disponíveis (passar como argumento 'modelo'):
        "DeepSeek-V3.2"                       ← padrão, qualidade top
        "Meta-Llama-3.3-70B-Instruct"         ← sólido e confiável
        "Qwen3-235B"                          ← raciocínio avançado
        "Llama-4-Maverick-17B-128E-Instruct"  ← multimodal
        "gpt-oss-120b"                        ← modelo open-source OpenAI
        "Meta-Llama-3.1-8B-Instruct"          ← mais rápido/leve

    Uso:
        agente = AgenteSambaNova()                              # DeepSeek V3.2
        agente = AgenteSambaNova(modelo="gpt-oss-120b")         # GPT open-source

        resp = agente.executar("chat", "sua pergunta")
        resp = agente.executar("resumo", texto_longo)
        resp = agente.executar("analise", dados, foco="riscos")
        resp = agente.executar("traducao", texto_en)

    Key gratuita em: https://cloud.sambanova.ai/apis
    """

    def __init__(self, modelo: str = SAMBANOVA_MODELO_PADRAO):
        self._pool = PoolSambaNova.do_env(modelo=modelo)
        self.model_id = modelo
        self._client_stub = _ClienteSambaNovaStub(self._pool)
        self._skills: dict[str, SkillBase] = {}
        self._registrar_skills()
        logger.info("AgenteSambaNova pronto | modelo=%s | keys=%d",
                    modelo, len(self._pool._pool))

    def executar(self, skill_nome: str, entrada: str, **kwargs) -> str:
        if skill_nome not in self._skills:
            raise ValueError(
                f"Skill '{skill_nome}' não encontrada. "
                f"Disponíveis: {list(self._skills.keys())}"
            )
        skill = self._skills[skill_nome]
        skill.model_id = self.model_id
        return skill.executar(entrada, **kwargs)

    def status_pool(self) -> list[dict]:
        return self._pool.status()

    def _registrar_skills(self) -> None:
        stub = self._client_stub
        for nome, cls in [
            ("chat",     SkillChat),
            ("resumo",   SkillResumo),
            ("analise",  SkillAnalise),
            ("traducao", SkillTradução),
        ]:
            self._skills[nome] = cls(client=stub, model_id=self.model_id)  # type: ignore[arg-type]
