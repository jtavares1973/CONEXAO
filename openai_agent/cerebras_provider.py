# ============================================================
# cerebras_provider.py
# Provedor Cerebras — O MAIS RÁPIDO do mundo (chip WSE, não GPU)
#
# Por que Cerebras?
#   - ~2.200 tokens/segundo (20x mais rápido que OpenAI/Anthropic)
#   - Free tier: 30 RPM, sem limite diário documentado
#   - Llama 3.1 8B: $0,10/M tokens (plano pago) — free tier generoso
#   - 100% compatível com SDK openai
#   - Ideal para processar 1000+ registros rapidamente
#
# Como conseguir sua key gratuita:
#   1. Acesse https://cloud.cerebras.ai
#   2. Crie conta (sem cartão de crédito)
#   3. Perfil → API Keys → Create
#   4. Copie e coloque no .env como CEREBRAS_API_KEY
#
# Limites free tier (fev/2026):
#   llama-3.1-8b-instant   → 30 RPM | ~2.200 tok/s
#   llama-3.3-70b          → 30 RPM | ~500 tok/s
#   qwen3-32b              → 60 RPM | ~1.400 tok/s
# ============================================================

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

logger = logging.getLogger(__name__)

# URL base do Cerebras (compatível com SDK openai)
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Modelo padrão: mais rápido disponível
CEREBRAS_MODELO_PADRAO = "llama3.1-8b"

# Modelos disponíveis no Cerebras (IDs reais da API)
CEREBRAS_MODELOS = {
    # nome                              RPM   velocidade (tok/s)  contexto
    "llama3.1-8b":                   {"rpm": 30, "tps": 2200, "ctx": 131072},
    "gpt-oss-120b":                  {"rpm": 30, "tps":  800, "ctx": 131072},
    "qwen-3-235b-a22b-instruct-2507":{"rpm": 60, "tps": 1400, "ctx": 131072},
    "zai-glm-4.7":                   {"rpm": 30, "tps":  600, "ctx": 131072},
}

# Cooldown quando bate rate limit
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
        logger.warning("Cerebras key ...%s em cooldown por %ds.", self.key[-6:], segundos)

    def resetar(self) -> None:
        self.cooldown_ate = 0.0


class PoolCerebras:
    """
    Pool de API keys do Cerebras com Round-Robin.
    
    O Cerebras usa chip WSE (Wafer Scale Engine) — não GPU —
    o que resulta na inferência mais rápida disponível publicamente.
    Ideal para processar grandes volumes rapidamente.
    """

    def __init__(self, keys_e_modelos: list[tuple[str, str]]):
        self._pool = [_StatusKey(key=k, modelo=m) for k, m in keys_e_modelos]
        self._indice = 0
        self._clientes: dict[str, OpenAI] = {}

        for sk in self._pool:
            self._clientes[sk.key] = OpenAI(
                api_key=sk.key,
                base_url=CEREBRAS_BASE_URL,
            )

        logger.info(
            "PoolCerebras iniciado com %d key(s). Modelos: %s",
            len(self._pool), [s.modelo for s in self._pool],
        )

    @classmethod
    def do_env(cls, modelo: str = CEREBRAS_MODELO_PADRAO) -> "PoolCerebras":
        """Cria pool lendo CEREBRAS_API_KEY, CEREBRAS_API_KEY_2, ... do .env."""
        from .config import carregar_env
        carregar_env()
        keys = []
        k1 = os.getenv("CEREBRAS_API_KEY", "")
        if k1:
            keys.append((k1, modelo))
        for i in range(2, 10):
            k = os.getenv(f"CEREBRAS_API_KEY_{i}", "")
            if k:
                keys.append((k, modelo))

        if not keys:
            raise ValueError(
                "Nenhuma CEREBRAS_API_KEY encontrada no .env!\n"
                "Crie uma key gratuita em https://cloud.cerebras.ai\n"
                "e adicione CEREBRAS_API_KEY=csk_... no .env"
            )
        logger.info("PoolCerebras: %d key(s) carregadas do .env.", len(keys))
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
                raise RuntimeError("Todas as keys Cerebras em cooldown.")

            try:
                resp = self._clientes[sk.key].chat.completions.create(
                    model=sk.modelo,
                    messages=mensagens,     # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperatura,
                )
                sk.reqs_total += 1
                conteudo = resp.choices[0].message.content or ""
                logger.debug("Cerebras OK | key=...%s | modelo=%s", sk.key[-6:], sk.modelo)
                return conteudo.strip(), sk.modelo

            except RateLimitError as exc:
                logger.warning("Cerebras rate limit | key=...%s: %s", sk.key[-6:], exc)
                sk.aplicar_cooldown()
                ultimo_erro = exc

            except (APIStatusError, APIConnectionError) as exc:
                logger.error("Cerebras erro API | key=...%s: %s", sk.key[-6:], exc)
                ultimo_erro = exc

        raise RuntimeError(f"Todas as keys Cerebras falharam. Último erro: {ultimo_erro}")

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
            logger.warning("Todas as keys Cerebras em cooldown. Aguardando %.0fs...", espera)
            time.sleep(espera + 1)
        mais_cedo.resetar()
        return mais_cedo
