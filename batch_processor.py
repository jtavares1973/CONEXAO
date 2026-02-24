# ============================================================
# batch_processor.py
# Processa grandes volumes de registros usando a IA
# com controle automático de cadência (rate limit).
#
# Funciona assim:
#   - Você passa uma lista com N registros
#   - Ele processa tudo automaticamente, respeitando o limite
#     de requisições por minuto de cada modelo
#   - Salva o progresso em arquivo .jsonl para não perder nada
#   - Se interrompido, retoma de onde parou
#   - Mostra barra de progresso com tempo estimado
# ============================================================

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from openai_agent import AgenteOpenAI

# ── Configurações de cadência ─────────────────────────────────────────────────
# Cada modelo gratuito suporta 8 req/min.
# Com 2 modelos em Round-Robin = 16 req/min teórico.
# Usamos 14 req/min (margem de segurança de ~12%) para não travar nunca.
REQ_POR_MINUTO_SEGURO = 14
INTERVALO_ENTRE_REQS  = 60 / REQ_POR_MINUTO_SEGURO  # ~4.3 segundos entre cada req


@dataclass
class ResultadoRegistro:
    """Resultado do processamento de um único registro."""
    id: int
    entrada: str
    saida: str = ""
    erro: str  = ""
    modelo_usado: str = ""
    tempo_s: float = 0.0
    sucesso: bool = False


class ProcessadorEmLote:
    """
    Processa uma lista de registros com IA de forma automática e resiliente.

    Características:
      - Cadência controlada: respeita o limite de req/min sem travar
      - Checkpoint: salva progresso a cada registro — retoma se interrompido
      - Barra de progresso no terminal
      - Relatório final com estatísticas

    Exemplo de uso:
        registros = ["Texto 1", "Texto 2", ..., "Texto 1000"]

        processador = ProcessadorEmLote(agente, skill="resumo")
        resultados  = processador.processar(registros)
    """

    def __init__(
        self,
        agente: AgenteOpenAI,
        skill: str = "chat",
        arquivo_checkpoint: str = "progresso.jsonl",
        req_por_minuto: int = REQ_POR_MINUTO_SEGURO,
        **kwargs_skill,
    ):
        self.agente = agente
        self.skill  = skill
        self.kwargs_skill = kwargs_skill
        self.checkpoint_path = Path(arquivo_checkpoint)
        self.intervalo = 60 / req_por_minuto

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def processar(
        self,
        registros: list[str],
        prompt_prefixo: str = "",
    ) -> list[ResultadoRegistro]:
        """
        Processa todos os registros e retorna lista de ResultadoRegistro.

        Parâmetros:
            registros      — lista de textos a processar
            prompt_prefixo — instrução que vai antes de cada registro
                             Ex: "Extraia nome, CPF e endereço do texto abaixo:"
        """
        total = len(registros)
        logger.info("Iniciando processamento de %d registros.", total)
        self._exibir_estimativa(total)

        # Carrega progresso anterior (se existir)
        ja_processados = self._carregar_checkpoint()
        ids_prontos    = {r["id"] for r in ja_processados}

        resultados: list[ResultadoRegistro] = [
            ResultadoRegistro(**r) for r in ja_processados
        ]

        pendentes = [
            (i, texto)
            for i, texto in enumerate(registros)
            if i not in ids_prontos
        ]

        logger.info(
            "%d já processados (checkpoint). %d pendentes.",
            len(ids_prontos), len(pendentes),
        )

        for posicao, (idx, texto) in enumerate(pendentes, 1):
            inicio = time.time()

            # Monta a entrada com o prefixo de instrução
            entrada = f"{prompt_prefixo}\n\n{texto}" if prompt_prefixo else texto

            resultado = ResultadoRegistro(id=idx, entrada=texto)

            try:
                saida = self.agente.executar(self.skill, entrada, **self.kwargs_skill)
                resultado.saida        = saida
                resultado.sucesso      = True
                resultado.modelo_usado = self.agente.model_id

            except Exception as exc:
                resultado.erro    = str(exc)
                resultado.sucesso = False
                logger.error("Erro no registro #%d: %s", idx, exc)

            resultado.tempo_s = round(time.time() - inicio, 2)

            # Salva no checkpoint imediatamente
            resultados.append(resultado)
            self._salvar_checkpoint(resultado)

            # Exibe progresso
            self._exibir_progresso(posicao, len(pendentes), resultado)

            # Cadência controlada — aguarda o tempo necessário entre requisições
            # (descontando o tempo que a própria API já levou)
            tempo_restante = self.intervalo - resultado.tempo_s
            if tempo_restante > 0 and posicao < len(pendentes):
                time.sleep(tempo_restante)

        self._exibir_relatorio(resultados)
        return resultados

    # ------------------------------------------------------------------
    # Checkpoint — salva/carrega progresso em arquivo .jsonl
    # (cada linha é um JSON — robusto a interrupções)
    # ------------------------------------------------------------------
    def _salvar_checkpoint(self, resultado: ResultadoRegistro) -> None:
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(resultado), ensure_ascii=False) + "\n")

    def _carregar_checkpoint(self) -> list[dict]:
        if not self.checkpoint_path.exists():
            return []
        registros = []
        with open(self.checkpoint_path, encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if linha:
                    try:
                        registros.append(json.loads(linha))
                    except json.JSONDecodeError:
                        pass
        return registros

    def limpar_checkpoint(self) -> None:
        """Remove o arquivo de checkpoint para começar do zero."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint removido — próximo processamento começa do zero.")

    # ------------------------------------------------------------------
    # Exibição no terminal
    # ------------------------------------------------------------------
    @staticmethod
    def _exibir_estimativa(total: int) -> None:
        minutos = math.ceil(total / REQ_POR_MINUTO_SEGURO)
        horas   = minutos // 60
        mins    = minutos % 60
        print("\n" + "="*60)
        print(f"  Total de registros : {total}")
        print(f"  Cadência segura    : {REQ_POR_MINUTO_SEGURO} req/min")
        print(f"  Tempo estimado     : {horas}h {mins}min")
        print(f"  (pode rodar em background — retoma se interrompido)")
        print("="*60 + "\n")

    @staticmethod
    def _exibir_progresso(posicao: int, total: int, resultado: ResultadoRegistro) -> None:
        pct      = posicao / total * 100
        barra    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        status   = "✓" if resultado.sucesso else "✗"
        pendente = total - posicao
        logger.info(
            "[%s] |%s| %d/%d (%.0f%%) | #%d | pendentes=%d | %.1fs",
            status, barra, posicao, total, pct,
            resultado.id, pendente, resultado.tempo_s,
        )

    @staticmethod
    def _exibir_relatorio(resultados: list[ResultadoRegistro]) -> None:
        total    = len(resultados)
        ok       = sum(1 for r in resultados if r.sucesso)
        erros    = total - ok
        tempo_t  = sum(r.tempo_s for r in resultados)
        print("\n" + "="*60)
        print("  RELATÓRIO FINAL")
        print("="*60)
        print(f"  Total processado : {total}")
        print(f"  Sucesso          : {ok}  ({ok/total*100:.1f}%)")
        print(f"  Erros            : {erros}")
        print(f"  Tempo total      : {tempo_t/60:.1f} min")
        print(f"  Média por req    : {tempo_t/total:.1f}s")
        print("="*60 + "\n")


# ============================================================
# EXEMPLO: extração de informações de 20 registros fictícios
# (simula o caso real de 1000 registros — só muda o tamanho da lista)
# ============================================================
if __name__ == "__main__":

    # Simula 20 registros de texto (no real seriam seus dados)
    registros_exemplo = [
        f"Cliente {i}: João Silva {i}, CPF 123.456.{i:03d}-00, "
        f"comprou Produto-{i} por R$ {i * 49.90:.2f} em 2026-01-{(i % 28) + 1:02d}."
        for i in range(1, 21)
    ]

    # Instrução que vai na frente de CADA registro
    instrucao = (
        "Extraia as informações abaixo em formato JSON com os campos: "
        "nome, cpf, produto, valor, data. Retorne APENAS o JSON, sem explicações."
    )

    # Cria o agente com Round-Robin automático
    agente = AgenteOpenAI()

    # Cria o processador — cadência de 14 req/min (seguro para 2 modelos gratuitos)
    processador = ProcessadorEmLote(
        agente=agente,
        skill="chat",
        arquivo_checkpoint="extracao_progresso.jsonl",
        req_por_minuto=14,
    )

    # Processa todos os registros
    # Se interromper no meio, rode de novo — continua de onde parou!
    resultados = processador.processar(registros_exemplo, prompt_prefixo=instrucao)

    # Exibe os 5 primeiros resultados
    print("\nPrimeiros 5 resultados:")
    for r in resultados[:5]:
        print(f"\n  Registro #{r.id} | modelo={r.modelo_usado} | {r.tempo_s}s")
        print(f"  {r.saida[:120]}")
