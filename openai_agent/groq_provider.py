# ============================================================
# groq_provider.py
# Provedor Groq — alternativa ao OpenRouter com limites muito
# maiores no plano gratuito.
#
# Por que Groq?
#   - Free tier: 30 req/min (vs 8 do OpenRouter)
#   - llama-3.1-8b-instant: 14.400 req/DIA gratuitos
#   - Inferência em hardware especializado (muito mais rápido)
#   - 100% compatível com o SDK openai — só muda base_url
#   - Suporta rotação de múltiplas keys (dobra/triplica capacidade)
#
# Como conseguir sua key gratuita:
#   1. Acesse https://console.groq.com
#   2. Crie uma conta (sem cartão de crédito)
#   3. Vá em "API Keys" → "Create API Key"
#   4. Copie e coloque no .env como GROQ_API_KEY
#
# Limites gratuitos reais (fev/2026):
#   llama-3.1-8b-instant      → 30 RPM | 14.400 RPD
#   llama-3.3-70b-versatile   → 30 RPM |  1.000 RPD
#   qwen/qwen3-32b            → 60 RPM |  1.000 RPD
#   moonshotai/kimi-k2        → 60 RPM |  1.000 RPD
# ============================================================

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

logger = logging.getLogger(__name__)

# URL base do Groq (compatível com SDK openai)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Modelo padrão: maior RPD gratuito (14.400/dia = ~600/hora)
GROQ_MODELO_PADRAO = "llama-3.1-8b-instant"

# Modelos gratuitos disponíveis no Groq
GROQ_MODELOS = {
    # nome                              RPM   RPD      contexto
    "llama-3.1-8b-instant":           {"rpm": 30, "rpd": 14400, "ctx": 131072},
    "llama-3.3-70b-versatile":        {"rpm": 30, "rpd":  1000, "ctx": 131072},
    "qwen-qwen3-32b":                 {"rpm": 60, "rpd":  1000, "ctx": 131072},
    "moonshotai-kimi-k2-instruct":    {"rpm": 60, "rpd":  1000, "ctx": 131072},
    "meta-llama-llama-4-scout-17b":   {"rpm": 30, "rpd":  1000, "ctx": 131072},
}

# Cooldown quando bate rate limit por minuto
COOLDOWN_RPM = 62   # segundos


@dataclass
class _StatusKey:
    """Controla o estado de uma API key no pool de rotação."""
    key: str
    modelo: str
    cooldown_ate: float = 0.0
    reqs_hoje: int = 0

    def em_cooldown(self) -> bool:
        return time.time() < self.cooldown_ate

    def aplicar_cooldown(self, segundos: int = COOLDOWN_RPM) -> None:
        self.cooldown_ate = time.time() + segundos
        logger.warning(
            "Key ...%s em cooldown por %ds.", self.key[-6:], segundos
        )

    def resetar(self) -> None:
        self.cooldown_ate = 0.0


class PoolGroq:
    """
    Pool de API keys do Groq com Round-Robin.

    Com uma única key já é muito melhor que o OpenRouter.
    Com 2-3 keys (todas gratuitas) processa 1000+ registros/dia
    sem pagar nada.

    Cada key pode ter um modelo diferente, maximizando os limites:
      key1 → llama-3.1-8b-instant (14.400/dia)
      key2 → llama-3.1-8b-instant (mais 14.400/dia)
      key3 → llama-3.3-70b-versatile (1.000/dia) ← para registros complexos
    """

    def __init__(self, keys_e_modelos: list[tuple[str, str]]):
        """
        Recebe lista de (api_key, modelo).
        Ex: [("gsk_abc...", "llama-3.1-8b-instant"),
             ("gsk_xyz...", "llama-3.1-8b-instant")]
        """
        self._pool = [
            _StatusKey(key=k, modelo=m)
            for k, m in keys_e_modelos
        ]
        self._indice = 0
        self._clientes: dict[str, OpenAI] = {}

        # Pré-cria um cliente por key
        for sk in self._pool:
            self._clientes[sk.key] = OpenAI(
                api_key=sk.key,
                base_url=GROQ_BASE_URL,
            )

        logger.info(
            "PoolGroq iniciado com %d key(s). Modelos: %s",
            len(self._pool),
            [s.modelo for s in self._pool],
        )

    @classmethod
    def do_env(cls, modelo: str = GROQ_MODELO_PADRAO) -> "PoolGroq":
        """
        Cria pool lendo as keys do .env automaticamente.
        Procura GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
        """
        from .config import carregar_env
        carregar_env()
        keys = []

        # Key principal
        k1 = os.getenv("GROQ_API_KEY", "")
        if k1:
            keys.append((k1, modelo))

        # Keys extras (rotação)
        for i in range(2, 10):
            k = os.getenv(f"GROQ_API_KEY_{i}", "")
            if k:
                keys.append((k, modelo))

        if not keys:
            raise ValueError(
                "Nenhuma GROQ_API_KEY encontrada no .env!\n"
                "Crie uma conta gratuita em https://console.groq.com\n"
                "e adicione GROQ_API_KEY=gsk_... no arquivo .env"
            )

        logger.info("PoolGroq: %d key(s) carregadas do .env.", len(keys))
        return cls(keys)

    def completar(
        self,
        mensagens: list[dict],
        max_tokens: int = 2048,
        temperatura: float = 0.7,
    ) -> tuple[str, str]:
        """
        Faz uma chamada com Round-Robin entre as keys.
        Retorna (resposta, modelo_usado).
        Tenta todas as keys antes de desistir.
        """
        total = len(self._pool)
        ultimo_erro: Optional[Exception] = None

        for _ in range(total * 2):  # *2 para dar chance após cooldown
            sk = self._proxima_disponivel()
            if sk is None:
                raise RuntimeError("Todas as keys do Groq em cooldown.")

            cliente = self._clientes[sk.key]
            try:
                resp = cliente.chat.completions.create(
                    model=sk.modelo,
                    messages=mensagens,     # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperatura,
                )
                sk.reqs_hoje += 1
                conteudo = resp.choices[0].message.content or ""
                logger.debug(
                    "Groq OK | key=...%s | modelo=%s | tokens=%s",
                    sk.key[-6:], sk.modelo, resp.usage,
                )
                return conteudo.strip(), sk.modelo

            except RateLimitError as exc:
                logger.warning(
                    "Groq rate limit | key=...%s: %s",
                    sk.key[-6:], exc,
                )
                sk.aplicar_cooldown()
                ultimo_erro = exc

            except (APIStatusError, APIConnectionError) as exc:
                logger.error(
                    "Groq erro de API | key=...%s: %s", sk.key[-6:], exc
                )
                ultimo_erro = exc

        raise RuntimeError(
            f"Todas as keys Groq falharam. Último erro: {ultimo_erro}"
        )

    def status(self) -> list[dict]:
        """Retorna situação atual de cada key."""
        agora = time.time()
        return [
            {
                "key_sufixo":  f"...{s.key[-6:]}",
                "modelo":      s.modelo,
                "disponivel":  not s.em_cooldown(),
                "cooldown_s":  max(0, round(s.cooldown_ate - agora)),
                "reqs_hoje":   s.reqs_hoje,
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

        # Todas em cooldown — espera a mais próxima e retorna
        mais_cedo = min(self._pool, key=lambda s: s.cooldown_ate)
        espera = max(0.0, mais_cedo.cooldown_ate - time.time())
        if espera > 0:
            logger.warning(
                "Todas as keys Groq em cooldown. Aguardando %.0fs...", espera
            )
            time.sleep(espera + 1)
        mais_cedo.resetar()
        return mais_cedo
