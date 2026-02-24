# ============================================================
# testar_groq.py
# Demonstra a velocidade e capacidade do Groq vs OpenRouter
# ============================================================

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import time
from openai_agent import AgenteGroq

print("="*60)
print("TESTE DO AGENTE GROQ")
print("30 req/min grátis | 14.400 req/dia")
print("="*60)

agente = AgenteGroq(modelo="llama-3.1-8b-instant")

# Status do pool de keys
print("\nPool de API Keys:")
for s in agente.status_pool():
    print(f"  Key {s['key_sufixo']} | modelo={s['modelo']} | disponivel={s['disponivel']}")

# Teste de velocidade: 4 requisições seguidas
print("\n4 requisições seguidas (sem esperar!):")
print("-"*40)

for i in range(1, 5):
    inicio = time.time()
    r = agente.executar("chat", f"Responda apenas: OK {i}")
    duracao = time.time() - inicio
    print(f"  req #{i} → {r[:30]} ({duracao:.1f}s)")

print("\n" + "="*60)
print("Para 1000 registros com Groq:")
print("  30 req/min → ~34 minutos (vs ~2h no OpenRouter)")
print("  Com 2 keys → ~17 minutos")
print("  Com 3 keys → ~12 minutos — tudo gratuito!")
print("="*60)
