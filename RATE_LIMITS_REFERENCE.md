# Referência: Rate Limits de APIs LLM Gratuitas (2025/2026)

> Pesquisado em fev/2026. Limites mudam com frequência — cheque as documentações oficiais.

---

## 1. Groq (groq.com) — O melhor free tier disponível

### Compatibilidade com SDK `openai`
100% compatível. Basta trocar `base_url` e `api_key`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_SEU_TOKEN_GROQ",
)
```

### Limites do Free Tier (Fev/2026)
Fonte: https://console.groq.com/docs/rate-limits

| Modelo                                  | RPM | RPD    | TPM    | TPD     |
|-----------------------------------------|-----|--------|--------|---------|
| `llama-3.1-8b-instant`                  | 30  | 14.400 | 6.000  | 500.000 |
| `llama-3.3-70b-versatile`               | 30  | 1.000  | 12.000 | 100.000 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 1.000 | 30.000 | 500.000 |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | 30 | 1.000 | 6.000 | 500.000 |
| `qwen/qwen3-32b`                        | 60  | 1.000  | 6.000  | 500.000 |
| `moonshotai/kimi-k2-instruct`           | 60  | 1.000  | 10.000 | 300.000 |
| `openai/gpt-oss-20b`                    | 30  | 1.000  | 8.000  | 200.000 |
| `openai/gpt-oss-120b`                   | 30  | 1.000  | 8.000  | 200.000 |

**Legenda:** RPM = requests/minuto, RPD = requests/dia, TPM = tokens/minuto, TPD = tokens/dia

### Destaques:
- `llama-3.1-8b-instant`: **14.400 req/dia** — o mais generoso para volume
- `llama-3.3-70b-versatile`: 1.000 req/dia mas **12K tokens/min**
- `qwen3-32b`: **60 RPM** (o dobro dos outros)
- Groq usa **LPUs** (não GPUs) → latências de 500-1000ms muito consistentes
- Sem cobrança de cartão para começar — só cadastro

### Batch API do Groq
O Groq tem **Batch Processing** nativo (50% mais barato, sem impacto nos rate limits normais):
- Janela de processamento: 24h a 7 dias
- Não consome sua cota de RPM/TPM padrão
- Docs: https://console.groq.com/docs/batch

---

## 2. OpenRouter (openrouter.ai) — Free tier com muitos modelos

### Limites free tier por modelo `:free`
- Cada modelo `:free` tem seu próprio rate limit (geralmente **8-20 req/min**)
- Rate limit por IP no free tier (não só por key)
- Modelos gratuitos em abr/2025: `deepseek-r1:free`, `llama-3.3-70b:free`, `qwen3-235b:free`

### URL base
```
https://openrouter.ai/api/v1
```

---

## 3. Hugging Face Inference API

### Créditos mensais gratuitos (Fev/2026)
| Plano            | Crédito mensal | Cartão necessário |
|------------------|---------------|-------------------|
| Free             | **$0,10**     | Não               |
| PRO              | $2,00         | Sim               |
| Team/Enterprise  | $2,00/seat    | Sim               |

### Importante (atualização jul/2025)
- O `hf-inference` free tier **agora foca em CPU** (embeddings, classificação, modelos pequenos como BERT/GPT-2)
- Para LLMs grandes gratuitos via HF, a estratégia mudou para **Inference Providers** (Together, Replicate, Nebius, etc.)
- Cada provider tem seus próprios rate limits e a HF cobra os $0,10/mês de crédito deles

### Uso com SDK openai
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1",
    api_key="hf_SEU_TOKEN",
)
```

---

## 4. Técnicas da Comunidade para Contornar Rate Limits

### 4.1 Rotação de Múltiplas API Keys (Round-Robin)
A técnica mais comum no GitHub. Crie N contas (ou N projetos no Groq) e rode um iterador:

```python
import itertools
from openai import OpenAI

GROQ_KEYS = [
    "gsk_key1...",
    "gsk_key2...",
    "gsk_key3...",
]

key_pool = itertools.cycle(GROQ_KEYS)

def get_client() -> OpenAI:
    """Sempre retorna um client com a próxima key da rotação."""
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=next(key_pool),
    )
```

Com 3 keys no Groq free tier = `3 × 14.400 = 43.200 req/dia` para `llama-3.1-8b-instant`.

### 4.2 Rotação com Fallback por Erro 429 (Smarter Round-Robin)
```python
import time
import random

def chamar_com_fallback(prompt: str, tentativas: int = 5) -> str:
    for i in range(tentativas):
        client = get_client()  # próxima key
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** i + random.random()  # backoff exponencial
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Todas as keys esgotadas")
```

### 4.3 Rotação Multi-Provedor (Groq + OpenRouter + HF)
Combine provedores diferentes para maximizar throughput gratuito total:

```python
PROVIDERS = [
    {"base_url": "https://api.groq.com/openai/v1",    "key": "gsk_...", "model": "llama-3.1-8b-instant"},
    {"base_url": "https://openrouter.ai/api/v1",       "key": "sk-or-...", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    {"base_url": "https://api.groq.com/openai/v1",    "key": "gsk_key2", "model": "qwen/qwen3-32b"},
]
provider_pool = itertools.cycle(PROVIDERS)
```

### 4.4 Async com Semáforo (Paralelismo Controlado)
Processa múltiplas requisições em paralelo sem estourar o RPM:

```python
import asyncio
from openai import AsyncOpenAI

async def processar_lote_async(registros: list[str], rpm_max: int = 25) -> list[str]:
    semaforo = asyncio.Semaphore(rpm_max)
    clients = [AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=k) for k in GROQ_KEYS]
    client_pool = itertools.cycle(clients)

    async def processar_um(texto: str) -> str:
        async with semaforo:
            client = next(client_pool)
            resp = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": texto}]
            )
            await asyncio.sleep(60 / rpm_max)  # throttle
            return resp.choices[0].message.content

    tasks = [processar_um(r) for r in registros]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 4.5 Usar Groq Batch API para Volumes Grandes
Para 100-1000 registros sem urgência, o Batch é a melhor opção:
- 50% mais barato (já irrisório no free= de graça)
- **Não consome RPM/TPM normal**
- Você submete um arquivo JSONL e busca o resultado depois

### 4.6 Leitura dos Headers de Rate Limit
O Groq retorna headers que informam exatamente quando a cota reseta:

```python
# Com a lib `groq` nativa (não openai):
from groq import Groq

client = Groq(api_key="gsk_...")
resp = client.chat.completions.with_raw_response.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Olá"}]
)

remaining = resp.headers.get("x-ratelimit-remaining-requests")
reset_in  = resp.headers.get("x-ratelimit-reset-requests")  # ex: "2m59.56s"
print(f"Req restantes hoje: {remaining}. Reset em: {reset_in}")
```

### 4.7 Modelo Menor para Triagem, Maior para Casos Difíceis
Estratégia de "cascade":
1. Envie para `llama-3.1-8b-instant` (8B, rápido, 14.400/dia)
2. Se a resposta for curta/estranha → reenvie para `llama-3.3-70b-versatile`
3. Economize o 70B apenas para casos que precisam

---

## 5. Matemática Prática: 1000 Registros de Graça

### Cenário: só Groq free tier
| Estratégia                       | Req/dia disponíveis | Dias p/ 1000 registros |
|----------------------------------|--------------------|-----------------------|
| 1 key, llama-3.1-8b-instant      | 14.400             | **< 1 dia**           |
| 1 key, llama-3.3-70b-versatile   | 1.000              | 1 dia                 |
| 3 keys, llama-3.1-8b-instant     | 43.200             | < 1 hora              |
| Groq Batch API                   | sem limite diário* | 1-7 dias async        |

*O Batch API tem janela de 24h-7d mas sem quota de RPM.

---

## 6. Instalação

```bash
# Groq SDK oficial (opcional, mas tem acesso aos headers)
pip install groq

# Já funciona com openai SDK (sem instalação extra):
pip install openai
```

---

## 7. Links Úteis

| Recurso | URL |
|---------|-----|
| Groq Console | https://console.groq.com |
| Groq Rate Limits (docs) | https://console.groq.com/docs/rate-limits |
| Groq Batch API | https://console.groq.com/docs/batch |
| Groq OpenAI Compat. | https://console.groq.com/docs/openai |
| OpenRouter Free Models | https://openrouter.ai/models?q=:free |
| HF Inference Providers | https://huggingface.co/docs/inference-providers |
| HF Pricing & Credits | https://huggingface.co/docs/inference-providers/pricing |
