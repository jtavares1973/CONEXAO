# ============================================================
# sambanova_provider.py
# Provedor SambaNova — Melhores modelos open-source GRÁTIS
#
# Por que SambaNova?
#   - DeepSeek V3.2, Qwen3-235B, Llama 4 Maverick — todos grátis
#   - GPT-OSS 120B (OpenAI open-source) — grátis!
#   - 30-60 RPM, sem limite diário documentado no free tier
#   - ~500 tok/s de velocidade
#   - 100% compatível com SDK openai
#
# Como conseguir sua key gratuita:
#   1. Acesse https://cloud.sambanova.ai/apis
#   2. Crie conta (sem cartão de crédito)
#   3. Clique em "Get API Key"
#   4. Copie e coloque no .env como SAMBANOVA_API_KEY
#
# Modelos disponíveis (free, fev/2026):
#   DeepSeek-V3.2          → qualidade estado-da-arte
#   Meta-Llama-3.3-70B     → sólido e confiável
#   Qwen3-235B             → raciocínio avançado
#   Llama-4-Maverick-17B   → multimodal
#   gpt-oss-120b           → modelo OpenAI open-source
# ============================================================

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

logger = logging.getLogger(__name__)

# URL base do SambaNova (compatível com SDK openai)
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"

# Modelo padrão: DeepSeek V3.2 — qualidade top, grátis
SAMBANOVA_MODELO_PADRAO = "DeepSeek-V3.2"

# Modelos disponíveis no SambaNova (free tier)
SAMBANOVA_MODELOS = {
    # nome                               RPM   ctx
    "DeepSeek-V3.2":                   {"rpm": 30, "ctx": 163840},
    "Meta-Llama-3.3-70B-Instruct":     {"rpm": 30, "ctx": 131072},
    "Qwen3-235B":                       {"rpm": 30, "ctx": 131072},
    "Llama-4-Maverick-17B-128E-Instruct": {"rpm": 30, "ctx": 131072},
    "gpt-oss-120b":                     {"rpm": 30, "ctx": 131072},
    "Meta-Llama-3.1-8B-Instruct":      {"rpm": 60, "ctx": 131072},
}

COOLDOWN_RPM = 62


@dataclass
class _StatusKey:
    key: str
    modelo: str
    cooldown_ate: float = 0.0
    reqs_total: int = 0

    def em_cooldown(self) -> bool:
        return time.time() < self.cooldown_ate

    def aplicar_cooldown(self, segundos: int = COOLDOWN_RPM) -> None:
        self.cooldown_ate = time.time() + segundos
        logger.warning("SambaNova key ...%s em cooldown por %ds.", self.key[-6:], segundos)

    def resetar(self) -> None:
        self.cooldown_ate = 0.0


class PoolSambaNova:
    """
    Pool de API keys do SambaNova com Round-Robin.

    Acesso gratuito a modelos de ponta como DeepSeek V3.2, Qwen3-235B
    e o modelo open-source da OpenAI (gpt-oss-120b).
    """

    def __init__(self, keys_e_modelos: list[tuple[str, str]]):
        self._pool = [_StatusKey(key=k, modelo=m) for k, m in keys_e_modelos]
        self._indice = 0
        self._clientes: dict[str, OpenAI] = {}

        for sk in self._pool:
            self._clientes[sk.key] = OpenAI(
                api_key=sk.key,
                base_url=SAMBANOVA_BASE_URL,
            )

        logger.info(
            "PoolSambaNova iniciado com %d key(s). Modelos: %s",
            len(self._pool), [s.modelo for s in self._pool],
        )

    @classmethod
    def do_env(cls, modelo: str = SAMBANOVA_MODELO_PADRAO) -> "PoolSambaNova":
        """Cria pool lendo SAMBANOVA_API_KEY, SAMBANOVA_API_KEY_2, ... do .env."""
        from .config import carregar_env
        carregar_env()
        keys = []
        k1 = os.getenv("SAMBANOVA_API_KEY", "")
        if k1:
            keys.append((k1, modelo))
        for i in range(2, 10):
            k = os.getenv(f"SAMBANOVA_API_KEY_{i}", "")
            if k:
                keys.append((k, modelo))

        if not keys:
            raise ValueError(
                "Nenhuma SAMBANOVA_API_KEY encontrada no .env!\n"
                "Crie uma key gratuita em https://cloud.sambanova.ai/apis\n"
                "e adicione SAMBANOVA_API_KEY=sn-... no .env"
            )
        logger.info("PoolSambaNova: %d key(s) carregadas do .env.", len(keys))
        return cls(keys)

    def completar(
        self,
        mensagens: list[dict],
        max_tokens: int = 2048,
        temperatura: float = 0.7,
    ) -> tuple[str, str]:
        """Retorna (resposta, modelo_usado) com Round-Robin entre keys."""
        total = len(self._pool)
        ultimo_erro: Optional[Exception] = None

        for _ in range(total * 2):
            sk = self._proxima_disponivel()
            if sk is None:
                raise RuntimeError("Todas as keys SambaNova em cooldown.")

            try:
                resp = self._clientes[sk.key].chat.completions.create(
                    model=sk.modelo,
                    messages=mensagens,     # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperatura,
                )
                sk.reqs_total += 1
                conteudo = resp.choices[0].message.content or ""
                logger.debug("SambaNova OK | key=...%s | modelo=%s", sk.key[-6:], sk.modelo)
                return conteudo.strip(), sk.modelo

            except RateLimitError as exc:
                logger.warning("SambaNova rate limit | key=...%s: %s", sk.key[-6:], exc)
                sk.aplicar_cooldown()
                ultimo_erro = exc

            except (APIStatusError, APIConnectionError) as exc:
                logger.error("SambaNova erro API | key=...%s: %s", sk.key[-6:], exc)
                ultimo_erro = exc

        raise RuntimeError(f"Todas as keys SambaNova falharam. Último erro: {ultimo_erro}")

    def status(self) -> list[dict]:
        agora = time.time()
        return [
            {
                "key_sufixo": f"...{s.key[-6:]}",
                "modelo":     s.modelo,
                "disponivel": not s.em_cooldown(),
                "cooldown_s": max(0, round(s.cooldown_ate - agora)),
                "reqs_total": s.reqs_total,
            }
            for s in self._pool
        ]

    def _proxima_disponivel(self) -> Optional["_StatusKey"]:
        total = len(self._pool)
        for _ in range(total):
            sk = self._pool[self._indice % total]
            self._indice = (self._indice + 1) % total
            if not sk.em_cooldown():
                return sk
        mais_cedo = min(self._pool, key=lambda s: s.cooldown_ate)
        espera = max(0.0, mais_cedo.cooldown_ate - time.time())
        if espera > 0:
            logger.warning("Todas as keys SambaNova em cooldown. Aguardando %.0fs...", espera)
            time.sleep(espera + 1)
        mais_cedo.resetar()
        return mais_cedo
