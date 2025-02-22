import asyncio
import random
import aiohttp
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def enviar_sinal(h, g, linha, coluna):
    user = f"user_{random.randint(1, 100)}"
    modelo = {"rows": linha, "cols": coluna}
    alg = "CGNR" if random.randint(1, 2) == 1 else "CGNE"
    sinais = f"{h} e {g}"

    dados = {
        "user": user,
        "alg": alg,
        "modelo": modelo,
        "H": h,
        "g": g,
        "sinais": sinais,
    }

    timeout = aiohttp.ClientTimeout(total=300)  # Timeout de 300 segundos
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                "http://localhost:5001/reconstruir_imagem", json=dados
            ) as response:
                if response.status == 200:
                    resultado = await response.json()
                    logger.info(
                        f"Imagem reconstruída com sucesso para o usuário {user}"
                    )
                else:
                    logger.error(f"Erro ao processar a imagem para o usuário {user}")
        except Exception as e:
            logger.error(f"Erro ao enviar requisição: {e}")


async def monitorar_desempenho():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:5001/status_servidor") as response:
                if response.status == 200:
                    status = await response.json()
                    logger.info(
                        f"Status do servidor - CPU: {status['cpu_percent']}%, Memória usada: {status['memory_used']} bytes"
                    )
                else:
                    logger.error("Erro ao obter status do servidor.")
        except Exception as e:
            logger.error(f"Erro ao monitorar desempenho: {e}")


async def rodar_tarefas():
    iterations = random.randint(1, 8)
    tasks = []
    logger.info(f"Gerando {iterations} imagens...")

    for _ in range(iterations):
        random_number = random.randint(1, 6)
        files = {
            1: ("H-1.csv", "G-1.csv", 60, 60),
            2: ("H-1.csv", "G-2.csv", 60, 60),
            3: ("H-1.csv", "A-60x60-1.csv", 60, 60),
            4: ("H-2.csv", "g-30x30-1.csv", 30, 30),
            5: ("H-2.csv", "g-30x30-2.csv", 30, 30),
            6: ("H-2.csv", "A-30x30-1.csv", 30, 30),
        }
        h, g, rows, cols = files[random_number]
        logger.info(f"Enviando Sinais {h} e {g}")
        tasks.append(enviar_sinal(h, g, rows, cols))
        tasks.append(monitorar_desempenho())

    await asyncio.gather(*tasks)


async def main():
    await rodar_tarefas()


if __name__ == "__main__":
    asyncio.run(main())
