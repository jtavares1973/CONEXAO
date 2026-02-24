# ============================================================
# groq_key_rotator.py
#
# Gerenciador de múltiplas API keys para Groq (e outros provedores
# compatíveis com o SDK openai), com:
#   - Round-robin entre keys
#   - Fallback automático em 429 (exponential backoff)
#   - Leitura dos headers de rate limit (com lib groq nativa)
#   - Suporte a múltiplos provedores (Groq + OpenRouter)
#   - Pool compatível com a interface AgenteOpenAI existente
# ============================================================

from __future__ import annotations

import itertools
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Iterator

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurações de provedores conhecidos (base_url)
# ---------------------------------------------------------------------------
BASE_URLS = {
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "huggingface": "https://api-inference.huggingface.co/v1",
    "together":   "https://api.together.xyz/v1",
}

# ---------------------------------------------------------------------------
# Modelos Groq free tier (fev/2026) com seus limites reais
# Fonte: https://console.groq.com/docs/rate-limits
# ---------------------------------------------------------------------------
GROQ_MODELOS_FREE: dict[str, dict] = {
    "llama-3.1-8b-instant": {
        "rpm": 30, "rpd": 14_400, "tpm": 6_000, "tpd": 500_000,
        "descricao": "Llama 3.1 8B – mais req/dia (14.4k), ótimo para volume",
    },
    "llama-3.3-70b-versatile": {
        "rpm": 30, "rpd": 1_000, "tpm": 12_000, "tpd": 100_000,
        "descricao": "Llama 3.3 70B – melhor qualidade, 1k req/dia",
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "rpm": 30, "rpd": 1_000, "tpm": 30_000, "tpd": 500_000,
        "descricao": "Llama 4 Scout 17B – maior TPM (30k/min)",
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "rpm": 30, "rpd": 1_000, "tpm": 6_000, "tpd": 500_000,
        "descricao": "Llama 4 Maverick 17B MoE",
    },
    "qwen/qwen3-32b": {
        "rpm": 60, "rpd": 1_000, "tpm": 6_000, "tpd": 500_000,
        "descricao": "Qwen3 32B – 60 RPM (dobro dos outros)",
    },
    "moonshotai/kimi-k2-instruct": {
        "rpm": 60, "rpd": 1_000, "tpm": 10_000, "tpd": 300_000,
        "descricao": "Kimi K2 1T – 60 RPM, bom para raciocínio",
    },
    "openai/gpt-oss-20b": {
        "rpm": 30, "rpd": 1_000, "tpm": 8_000, "tpd": 200_000,
        "descricao": "GPT OSS 20B (OpenAI open-weight no Groq)",
    },
    "openai/gpt-oss-120b": {
        "rpm": 30, "rpd": 1_000, "tpm": 8_000, "tpd": 200_000,
        "descricao": "GPT OSS 120B – modelo maior, mesmo quota",
    },
}


# ---------------------------------------------------------------------------
# Dataclass que representa uma "fatia" do pool: provedor + key + modelo
# ---------------------------------------------------------------------------
@dataclass
class ProviderSlot:
    """
    Uma combinação de (provedor, api_key, modelo) que representa
    uma fatia de capacidade independente no pool de rotação.
    """
    nome: str           # ex: "groq-key1", "openrouter-key2"
    base_url: str
    api_key: str
    modelo: str
    rpm: int = 30       # requests por minuto
    rpd: int = 1_000    # requests por dia
    # Controle interno
    _req_esta_janela: int = field(default=0, repr=False)
    _ultima_req_ts: float = field(default=0.0, repr=False)
    _bloqueado_ate: float = field(default=0.0, repr=False)

    def esta_disponivel(self) -> bool:
        """Verifica se este slot não está em cooldown."""
        return time.monotonic() >= self._bloqueado_ate

    def marcar_rate_limit(self, retry_after_s: float = 60.0) -> None:
        """Marca este slot como bloqueado por `retry_after_s` segundos."""
        self._bloqueado_ate = time.monotonic() + retry_after_s
        logger.warning(
            "Slot '%s' bloqueado por %.0fs após rate limit 429.",
            self.nome, retry_after_s
        )

    def criar_client(self) -> OpenAI:
        """Instancia um cliente OpenAI apontando para este provedor."""
        return OpenAI(base_url=self.base_url, api_key=self.api_key)


# ---------------------------------------------------------------------------
# Construtor de slots a partir de variáveis de ambiente
# ---------------------------------------------------------------------------
def slots_groq_do_env(modelo: str = "llama-3.1-8b-instant") -> list[ProviderSlot]:
    """
    Lê GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... do ambiente
    e retorna uma lista de ProviderSlots prontos para uso.

    Configure no .env:
        GROQ_API_KEY=gsk_chave1
        GROQ_API_KEY_2=gsk_chave2
        GROQ_API_KEY_3=gsk_chave3

    Args:
        modelo: ID do modelo Groq a usar (ver GROQ_MODELOS_FREE).

    Returns:
        Lista de ProviderSlot (1 por key encontrada).
    """
    slots: list[ProviderSlot] = []
    info = GROQ_MODELOS_FREE.get(modelo, {"rpm": 30, "rpd": 1_000})

    # Primeira key: GROQ_API_KEY
    key1 = os.getenv("GROQ_API_KEY")
    if key1:
        slots.append(ProviderSlot(
            nome="groq-key1",
            base_url=BASE_URLS["groq"],
            api_key=key1,
            modelo=modelo,
            rpm=info["rpm"],
            rpd=info["rpd"],
        ))

    # Keys adicionais: GROQ_API_KEY_2, GROQ_API_KEY_3, ..., GROQ_API_KEY_9
    for i in range(2, 10):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            slots.append(ProviderSlot(
                nome=f"groq-key{i}",
                base_url=BASE_URLS["groq"],
                api_key=key,
                modelo=modelo,
                rpm=info["rpm"],
                rpd=info["rpd"],
            ))

    if not slots:
        logger.warning(
            "Nenhuma GROQ_API_KEY encontrada no ambiente. "
            "Adicione ao .env: GROQ_API_KEY=gsk_..."
        )
    else:
        total_rpd = sum(s.rpd for s in slots)
        logger.info(
            "%d slot(s) Groq carregados com modelo '%s'. "
            "Capacidade total: %d req/dia.",
            len(slots), modelo, total_rpd,
        )
    return slots


def slots_openrouter_do_env(modelo: str = "meta-llama/llama-3.3-70b-instruct:free") -> list[ProviderSlot]:
    """
    Lê OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, ... do ambiente.
    """
    slots: list[ProviderSlot] = []
    for i in range(1, 10):
        sufixo = "" if i == 1 else f"_{i}"
        key = os.getenv(f"OPENROUTER_API_KEY{sufixo}")
        if key:
            slots.append(ProviderSlot(
                nome=f"openrouter-key{i}",
                base_url=BASE_URLS["openrouter"],
                api_key=key,
                modelo=modelo,
                rpm=8,    # free tier OpenRouter ~8-20 RPM
                rpd=200,  # estimativa conservadora
            ))
    return slots


# ---------------------------------------------------------------------------
# Pool de Rotação Principal
# ---------------------------------------------------------------------------
class KeyRotatorPool:
    """
    Pool de múltiplas ProviderSlots com rotação automática e fallback.

    Uso básico:
        pool = KeyRotatorPool.do_groq_env(modelo="llama-3.1-8b-instant")
        resposta = pool.completar("Resuma este texto: ...")

    Uso avançado (múltiplos provedores):
        slots = slots_groq_do_env("llama-3.1-8b-instant")
        slots += slots_openrouter_do_env("meta-llama/llama-3.3-70b-instruct:free")
        pool = KeyRotatorPool(slots, max_tentativas=6)
    """

    def __init__(
        self,
        slots: list[ProviderSlot],
        max_tentativas: int = 5,
        intervalo_minimo_s: float = 2.0,
    ):
        if not slots:
            raise ValueError("KeyRotatorPool precisa de pelo menos 1 ProviderSlot.")
        self.slots = slots
        self.max_tentativas = max_tentativas
        self.intervalo_minimo_s = intervalo_minimo_s
        self._ciclo: Iterator[ProviderSlot] = itertools.cycle(slots)
        self._ultimo_ts: float = 0.0
        logger.info(
            "KeyRotatorPool iniciado com %d slot(s). Throughput estimado: %.0f req/min.",
            len(slots),
            sum(s.rpm for s in slots),
        )

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def do_groq_env(cls, modelo: str = "llama-3.1-8b-instant", **kwargs) -> "KeyRotatorPool":
        """Cria pool a partir das GROQ_API_KEY* no ambiente."""
        slots = slots_groq_do_env(modelo)
        if not slots:
            raise EnvironmentError(
                "Nenhuma GROQ_API_KEY encontrada. "
                "Adicione ao .env ou defina a variável de ambiente."
            )
        return cls(slots, **kwargs)

    @classmethod
    def multi_provedor_do_env(
        cls,
        modelo_groq: str = "llama-3.1-8b-instant",
        modelo_openrouter: str = "meta-llama/llama-3.3-70b-instruct:free",
        **kwargs,
    ) -> "KeyRotatorPool":
        """Cria pool combinando Groq + OpenRouter automaticamente."""
        slots = slots_groq_do_env(modelo_groq)
        slots += slots_openrouter_do_env(modelo_openrouter)
        if not slots:
            raise EnvironmentError("Nenhuma API key encontrada em nenhum provedor.")
        return cls(slots, **kwargs)

    # ── Seleção de slot ──────────────────────────────────────────────────────

    def _proximo_slot_disponivel(self) -> ProviderSlot:
        """
        Itera pelo ciclo até encontrar um slot sem cooldown.
        Lança RuntimeError se todos estiverem bloqueados.
        """
        visitados = 0
        total = len(self.slots)
        while visitados < total:
            slot = next(self._ciclo)
            if slot.esta_disponivel():
                return slot
            visitados += 1

        # Todos bloqueados → espera o menor tempo de cooldown restante
        menor_espera = min(
            max(0.0, s._bloqueado_ate - time.monotonic())
            for s in self.slots
        )
        logger.info("Todos os slots bloqueados. Aguardando %.1fs...", menor_espera)
        time.sleep(menor_espera + 0.5)  # +0.5s de margem
        return next(self._ciclo)

    # ── Chamada principal ────────────────────────────────────────────────────

    def completar(
        self,
        prompt: str,
        system: str = "Você é um assistente prestativo.",
        temperatura: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Envia uma mensagem e retorna o conteúdo da resposta.

        Aplica:
        - Throttling mínimo de `intervalo_minimo_s` entre chamadas
        - Round-robin entre slots disponíveis
        - Backoff exponencial em erro 429
        - Skip automático de slots bloqueados

        Args:
            prompt: Texto do usuário.
            system: System prompt.
            temperatura: Temperatura do modelo.
            max_tokens: Limite de tokens na resposta.

        Returns:
            Conteúdo da resposta como string.

        Raises:
            RuntimeError: Se todas as tentativas falharem.
        """
        ultimo_erro: Exception | None = None

        for tentativa in range(self.max_tentativas):
            # Throttle global mínimo
            decorrido = time.monotonic() - self._ultimo_ts
            if decorrido < self.intervalo_minimo_s:
                time.sleep(self.intervalo_minimo_s - decorrido)

            slot = self._proximo_slot_disponivel()
            client = slot.criar_client()

            try:
                logger.debug("Tentativa %d via slot '%s'...", tentativa + 1, slot.nome)
                resp = client.chat.completions.create(
                    model=slot.modelo,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperatura,
                    max_tokens=max_tokens,
                )
                self._ultimo_ts = time.monotonic()
                return resp.choices[0].message.content or ""

            except Exception as e:
                ultimo_erro = e
                msg = str(e).lower()

                if "429" in str(e) or "rate_limit" in msg or "rate limit" in msg:
                    # Backoff exponencial com jitter
                    espera = (2 ** tentativa) + random.uniform(0, 1)
                    slot.marcar_rate_limit(retry_after_s=espera * 3)
                    logger.warning(
                        "Rate limit no slot '%s' (tentativa %d/%d). "
                        "Aguardando %.1fs antes da próxima tentativa.",
                        slot.nome, tentativa + 1, self.max_tentativas, espera,
                    )
                    time.sleep(espera)

                elif "401" in str(e) or "authentication" in msg:
                    # Key inválida — bloquear permanentemente
                    slot.marcar_rate_limit(retry_after_s=86_400)  # 24h
                    logger.error(
                        "Autenticação falhou para slot '%s'. "
                        "Verifique a API key.", slot.nome
                    )

                else:
                    # Erro inesperado — logar e tentar próximo slot
                    logger.warning(
                        "Erro inesperado no slot '%s': %s",
                        slot.nome, e
                    )

        raise RuntimeError(
            f"Todas as {self.max_tentativas} tentativas falharam. "
            f"Último erro: {ultimo_erro}"
        )

    def listar_status(self) -> None:
        """Imprime o status atual de cada slot no pool."""
        agora = time.monotonic()
        print(f"\n{'='*55}")
        print(f"{'SLOT':<25} {'MODELO':<32} {'STATUS'}")
        print(f"{'='*55}")
        for s in self.slots:
            if agora >= s._bloqueado_ate:
                status = "✓ disponível"
            else:
                restante = s._bloqueado_ate - agora
                status = f"✗ bloqueado ({restante:.0f}s)"
            print(f"{s.nome:<25} {s.modelo:<32} {status}")
        print(f"{'='*55}\n")

    def estatisticas(self) -> dict:
        """Retorna dicionário com stats do pool."""
        agora = time.monotonic()
        disponiveis = sum(1 for s in self.slots if agora >= s._bloqueado_ate)
        return {
            "total_slots": len(self.slots),
            "slots_disponiveis": disponiveis,
            "slots_bloqueados": len(self.slots) - disponiveis,
            "rpm_total_teorico": sum(s.rpm for s in self.slots),
            "rpd_total_teorico": sum(s.rpd for s in self.slots),
        }


# ---------------------------------------------------------------------------
# Wrapper compatível com AgenteOpenAI para uso no batch_processor
# ---------------------------------------------------------------------------
class AgenteGroqRotativo:
    """
    Wrapper que expõe a mesma interface de `AgenteOpenAI.processar()`
    mas usa internamente o KeyRotatorPool com múltiplas keys Groq.

    Uso no batch_processor.py (substituição drop-in):

        from openai_agent.groq_key_rotator import AgenteGroqRotativo

        agente = AgenteGroqRotativo(
            modelo="llama-3.1-8b-instant",
            max_tentativas=5,
        )
        processador = ProcessadorEmLote(agente, skill="chat")
        resultados  = processador.processar(registros)
    """

    def __init__(
        self,
        modelo: str = "llama-3.1-8b-instant",
        max_tentativas: int = 5,
        intervalo_minimo_s: float = 2.0,
    ):
        self.modelo = modelo
        self.pool = KeyRotatorPool.do_groq_env(
            modelo=modelo,
            max_tentativas=max_tentativas,
            intervalo_minimo_s=intervalo_minimo_s,
        )

    def processar(self, texto: str, **kwargs) -> str:
        """Interface compatível com AgenteOpenAI.processar()."""
        system = kwargs.pop("system", "Você é um assistente prestativo.")
        return self.pool.completar(prompt=texto, system=system, **kwargs)


# ---------------------------------------------------------------------------
# Demonstração rápida
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=== KeyRotatorPool — Demo ===\n")

    # Tenta carregar keys do ambiente
    try:
        pool = KeyRotatorPool.do_groq_env(modelo="llama-3.1-8b-instant")
    except EnvironmentError as e:
        print(f"ERRO: {e}")
        print("\nAdicionea ao seu .env:")
        print("  GROQ_API_KEY=gsk_...")
        print("  GROQ_API_KEY_2=gsk_...  (opcional, para mais capacidade)")
        sys.exit(1)

    pool.listar_status()

    # Teste simples
    pergunta = "Em uma frase: o que é rate limiting de API?"
    print(f"Pergunta: {pergunta}")
    resposta = pool.completar(pergunta, max_tokens=100)
    print(f"Resposta: {resposta}\n")

    print("Estatísticas do pool:")
    for k, v in pool.estatisticas().items():
        print(f"  {k}: {v}")
