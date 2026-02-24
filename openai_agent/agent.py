# ============================================================
# agent.py
# Orquestrador central — o "cérebro" do sistema.
#
# Estratégia de distribuição de requisições: ROUND-ROBIN
#
# Em vez de usar sempre o mesmo modelo e só trocar quando der erro,
# cada requisição já vai automaticamente para o próximo modelo da fila.
#
# Exemplo com 4 modelos (A, B, C, D):
#   req 1 → A
#   req 2 → B
#   req 3 → C
#   req 4 → D
#   req 5 → A  (cicla de volta)
#   ...
#
# Isso distribui a carga e evita esgotar o limite de um único modelo.
# Se um modelo retornar rate limit, ele entra em "cooldown" temporário
# e as requisições pulam para o próximo disponível.
# ============================================================

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Type
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

from .model_selector import SeletorDeModelos
from .skills import (
    SkillBase, SkillChat, SkillResumo, SkillAnalise, SkillTradução,
)

logger = logging.getLogger(__name__)

# ── Tempo de cooldown quando um modelo recebe rate limit (segundos) ──────────
COOLDOWN_RATE_LIMIT = 60   # 1 minuto de "descanso" para o modelo penalizado

# ── Modelos do pool round-robin (confirmados disponíveis nesta conta) ────────────
# Os modelos deepseek/deepseek-r1:free e qwen/qwen3-235b-a22b:free retornam 404
# nesta conta OpenRouter e são removidos automaticamente do pool em runtime.
# Para adicionar novos modelos, basta incluir o ID abaixo.
POOL_MODELOS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]


@dataclass
class _StatusModelo:
    """Controla o estado de um modelo no pool round-robin."""
    model_id: str
    cooldown_ate: float = 0.0   # timestamp unix até quando está em cooldown

    def em_cooldown(self) -> bool:
        return time.time() < self.cooldown_ate

    def aplicar_cooldown(self, segundos: int = COOLDOWN_RATE_LIMIT) -> None:
        self.cooldown_ate = time.time() + segundos
        logger.warning(
            "Modelo '%s' em cooldown por %ds (rate limit).", self.model_id, segundos
        )

    def resetar_cooldown(self) -> None:
        self.cooldown_ate = 0.0


class _PoolRoundRobin:
    """
    Gerencia o rodízio de modelos.

    A cada chamada de .proximo() retorna o próximo modelo disponível
    (que não esteja em cooldown), ciclando de forma circular pelo pool.
    """

    def __init__(self, model_ids: list[str]):
        self._pool: list[_StatusModelo] = [_StatusModelo(m) for m in model_ids]
        self._indice: int = 0   # aponta para o próximo modelo a ser usado

    def proximo(self) -> Optional[str]:
        """
        Retorna o ID do próximo modelo disponível (sem cooldown).
        Percorre todo o pool uma vez; se todos estiverem em cooldown, retorna None.
        """
        total = len(self._pool)
        for _ in range(total):
            status = self._pool[self._indice % total]
            self._indice = (self._indice + 1) % total   # avança o ponteiro

            if not status.em_cooldown():
                return status.model_id

        # Todos em cooldown — returna o que vai desbloquear mais cedo
        mais_cedo = min(self._pool, key=lambda s: s.cooldown_ate)
        espera = max(0.0, mais_cedo.cooldown_ate - time.time())
        logger.warning(
            "Todos os modelos em cooldown. Aguardando %.0fs pelo mais rápido (%s)...",
            espera, mais_cedo.model_id,
        )
        time.sleep(espera + 1)
        mais_cedo.resetar_cooldown()
        return mais_cedo.model_id

    def penalizar(self, model_id: str) -> None:
        """Coloca um modelo em cooldown por rate limit."""
        for s in self._pool:
            if s.model_id == model_id:
                s.aplicar_cooldown()
                return

    def status_pool(self) -> list[dict]:
        """Retorna situação atual de todos os modelos para diagnóstico."""
        agora = time.time()
        return [
            {
                "modelo": s.model_id,
                "disponivel": not s.em_cooldown(),
                "cooldown_restante_s": max(0, round(s.cooldown_ate - agora)),
            }
            for s in self._pool
        ]


class AgenteOpenAI:
    """
    Agente principal que orquestra modelos e skills.

    Exemplo de uso:
        agente = AgenteOpenAI()
        resposta = agente.executar("chat", "Olá, como você está?")
        resumo   = agente.executar("resumo", texto_longo)
        analise  = agente.executar("analise", dados, foco="riscos")
        traduzido = agente.executar("traducao", texto_en, idioma_alvo="pt-BR")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        apenas_gratuitos: bool = True,
        model_id: Optional[str] = None,  # força um modelo específico se informado
        app_name: Optional[str] = None,
    ):
        # ── 1. Configurar credenciais ────────────────────────────────────────
        self._api_key  = api_key  or os.getenv("OPENROUTER_API_KEY", "")
        self._base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self._app_name = app_name or os.getenv("OPENROUTER_APP_NAME", "Agente-Python")

        if not self._api_key:
            raise ValueError(
                "API key não encontrada! Defina OPENROUTER_API_KEY no .env "
                "ou passe api_key= ao instanciar AgenteOpenAI."
            )

        # ── 2. Criar cliente OpenAI apontando para OpenRouter ─────────────────
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/naccarona",
                "X-Title": self._app_name,
            },
        )

        # ── 3. Montar o pool de modelos para Round-Robin ──────────────────────
        if model_id:
            # Usuário forçou um modelo específico — pool com apenas ele
            pool_ids = [model_id]
            logger.info("Pool fixo (modelo forçado): %s", model_id)
        else:
            # Pool sempre usa os 4 modelos curados em POOL_MODELOS (todos gratuitos
            # e testados). O seletor apenas os reordena do melhor para o pior score,
            # assim o Round-Robin começa sempre pelo mais capaz.
            seletor = SeletorDeModelos(self.client, apenas_gratuitos=apenas_gratuitos)
            ids_ranqueados = [m.id for m in seletor.listar_candidatos()]

            pool_ids = sorted(
                POOL_MODELOS,
                key=lambda m: ids_ranqueados.index(m) if m in ids_ranqueados else 999,
            )

            logger.info("Pool Round-Robin montado com %d modelos:", len(pool_ids))
            for i, mid in enumerate(pool_ids, 1):
                logger.info("  %d. %s", i, mid)

        # Cria o gerenciador round-robin
        self._pool = _PoolRoundRobin(pool_ids)

        # model_id expõe o modelo "atual" (o próximo a ser chamado)
        self.model_id: str = pool_ids[0]

        # ── 4. Registrar skills disponíveis ──────────────────────────────────
        self._skills: dict[str, SkillBase] = {}
        self._registrar_skills_padrao()

        logger.info("Agente pronto | pool=%s", pool_ids)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------
    def executar(self, skill_nome: str, entrada: str, **kwargs) -> str:
        """
        Executa uma skill usando Round-Robin entre os modelos do pool.

        A cada chamada, pega o PRÓXIMO modelo disponível (sem cooldown).
        Se receber rate limit, penaliza aquele modelo por 60s e tenta outro.

          req 1 → llama
          req 2 → deepseek      ← já vai pro próximo automaticamente
          req 3 → qwen
          req 4 → mistral
          req 5 → llama          ← cicla de volta
          ...
        """
        if skill_nome not in self._skills:
            raise ValueError(
                f"Skill '{skill_nome}' não encontrada. "
                f"Disponíveis: {list(self._skills.keys())}"
            )

        skill = self._skills[skill_nome]
        ultimo_erro: Optional[Exception] = None
        total_modelos = len(self._pool._pool)

        # Tenta no máximo (total de modelos * 2) vezes.
        # O fator 2 garante que se todos entrarem em cooldown de uma vez,
        # o pool aguarda o primeiro desbloquear e tenta novamente.
        for tentativa in range(1, total_modelos * 2 + 1):
            modelo = self._pool.proximo()

            if modelo is None:
                break

            skill.model_id = modelo
            logger.info(
                "req #%d | skill=%-8s | modelo=%s",
                tentativa, skill_nome, modelo,
            )

            try:
                resultado = skill.executar(entrada, **kwargs)
                self.model_id = modelo  # atualiza o "modelo atual" visível
                return resultado

            except RateLimitError as exc:
                logger.warning("Rate limit em '%s' — colocando em cooldown.", modelo)
                self._pool.penalizar(modelo)
                ultimo_erro = exc

            except (APIStatusError, APIConnectionError) as exc:
                # Se for 404, o modelo não existe nesta conta — remove do pool
                if hasattr(exc, 'status_code') and exc.status_code == 404:
                    self.remover_modelo(modelo)
                else:
                    logger.error("Erro de API em '%s': %s — pulando.", modelo, exc)
                ultimo_erro = exc

            except Exception as exc:
                logger.error("Erro inesperado em '%s': %s", modelo, exc)
                ultimo_erro = exc

        raise RuntimeError(
            f"Todos os modelos do pool falharam para '{skill_nome}'. "
            f"Último erro: {ultimo_erro}"
        )

    def status_pool(self) -> list[dict]:
        """Mostra situação atual de cada modelo no pool (disponível / cooldown)."""
        return self._pool.status_pool()

    def remover_modelo(self, model_id: str) -> None:
        """
        Remove permanentemente um modelo do pool.
        Chamado automaticamente quando o modelo retorna 404 (endpoint inexistente).
        """
        self._pool._pool = [s for s in self._pool._pool if s.model_id != model_id]
        logger.warning("Modelo '%s' removido do pool (endpoint indisponível 404).", model_id)

    def trocar_modelo(self, novo_model_id: str) -> None:
        """Força um modelo específico como próximo na fila (não remove os outros)."""
        logger.info("Modelo prioritário definido: %s", novo_model_id)
        self.model_id = novo_model_id

    def registrar_skill(self, nome: str, skill_classe: Type[SkillBase], **kwargs) -> None:
        """
        Registra uma nova skill customizada no agente.

        Exemplo:
            agente.registrar_skill("minha_skill", MinhaSkill, temperatura=0.5)
        """
        self._skills[nome] = skill_classe(self.client, self.model_id, **kwargs)
        logger.info("Skill registrada: %s", nome)

    def skills_disponiveis(self) -> list[str]:
        """Retorna lista de nomes das skills registradas."""
        return list(self._skills.keys())

    def obter_skill(self, nome: str) -> SkillBase:
        """Retorna a instância da skill pelo nome."""
        if nome not in self._skills:
            raise ValueError(f"Skill '{nome}' não encontrada.")
        return self._skills[nome]

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------
    def _registrar_skills_padrao(self) -> None:
        """Registra o conjunto padrão de skills no agente."""
        self._skills = {
            "chat":     SkillChat(self.client, self.model_id),
            "resumo":   SkillResumo(self.client, self.model_id),
            "analise":  SkillAnalise(self.client, self.model_id),
            "traducao": SkillTradução(self.client, self.model_id),
        }
        logger.debug("Skills registradas: %s", list(self._skills.keys()))

