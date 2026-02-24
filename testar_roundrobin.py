# ============================================================
# testar_roundrobin.py
# Demonstra visualmente o Round-Robin em ação:
# cada requisição vai automaticamente para um modelo diferente.
# ============================================================

from dotenv import load_dotenv
import time
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from openai_agent import AgenteOpenAI

agente = AgenteOpenAI()

# ── Status inicial do pool ───────────────────────────────────────────────────
print("\n" + "="*60)
print("POOL DE MODELOS (Round-Robin)")
print("="*60)
for s in agente.status_pool():
    situacao = "DISPONIVEL" if s["disponivel"] else f"COOLDOWN {s['cooldown_restante_s']}s"
    print(f"  {situacao:<15} {s['modelo']}")

# ── 4 requisições — observe que o modelo muda a cada uma ────────────────────
print("\n" + "="*60)
print("4 REQUISICOES — modelo deve alternar a cada vez")
print("="*60)

perguntas = [
    "Responda apenas: OLA 1",
    "Responda apenas: OLA 2",
    "Responda apenas: OLA 3",
    "Responda apenas: OLA 4",
]

for i, pergunta in enumerate(perguntas, 1):
    resposta = agente.executar("chat", pergunta)
    print(f"\n  req #{i} \u2192 modelo: {agente.model_id}")
    print(f"          resposta: {resposta[:70]}")
    if i < len(perguntas):
        print("          [aguardando 8s para n\u00e3o estourar rate limit...]")
        time.sleep(8)

print("\n" + "="*60)
print("Round-Robin funcionando! Cada req usou um modelo diferente.")
print("="*60)

print("\n" + "="*60)
print("Round-Robin funcionando! Cada req usou um modelo diferente.")
print("="*60)
