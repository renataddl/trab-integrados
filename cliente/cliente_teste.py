import asyncio
import random
import aiohttp


# Função para simular o envio de dados ao servidor
async def enviar_sinal(h, g, linha, coluna):
    user = f"user_{random.randint(1, 100)}"
    modelo = {"rows": linha, "cols": coluna}
    alg = "CGNR" if random.randint(1, 2) == 1 else "CGNE"
    sinais = str(h) + " e " + str(g)

    dados = {
        "user": user,
        "alg": alg,
        "modelo": modelo,
        "H": h,
        "g": g,
        "sinais": sinais,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:5001/reconstruir_imagem", json=dados
        ) as response:
            if response.status == 200:
                resultado = await response.json()
                print("Imagem reconstruída com sucesso")
            else:
                print("Erro ao processar a imagem.")


# Função para monitorar o desempenho do servidor
async def monitorar_desempenho():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:5001/status_servidor") as response:
            if response.status == 200:
                status = await response.json()
                print(f"Uso de CPU: {status['cpu_percent']}%")
                print(f"Memória usada: {status['memory_used']} bytes")
                print(f"Memória total: {status['memory_total']} bytes")
            else:
                print("Erro ao obter status do servidor.")


# Função principal para rodar as tarefas assíncronas
async def rodar_tarefas():
    iterations = random.randint(1, 8)  # Definindo um número de iterações para testar
    tasks = []  # Lista de tarefas assíncronas a serem executadas
    print("Gerarei " + str(iterations) + " imagens!")

    for i in range(iterations):
        random_number = random.randint(1, 6)
        H_1 = "H-1.csv"
        G_1 = "G-1.csv"
        G_2 = "G-2.csv"
        G_3 = "A-60x60-1.csv"
        H_2 = "H-2.csv"
        g_1 = "g-30x30-1.csv"
        g_2 = "g-30x30-2.csv"
        g_3 = "A-30x30-1.csv"

        if random_number == 1:
            print("Enviando Sinais H_1 e G_1")
            tasks.append(enviar_sinal(H_1, G_1, 60, 60))
        elif random_number == 2:
            print("Enviando Sinais H_1 e G_2")
            tasks.append(enviar_sinal(H_1, G_2, 60, 60))
        elif random_number == 3:
            print("Enviando Sinais H_1 e G_3")
            tasks.append(enviar_sinal(H_1, G_3, 60, 60))
        elif random_number == 4:
            print("Enviando Sinais H_2 e g_1")
            tasks.append(enviar_sinal(H_2, g_1, 30, 30))
        elif random_number == 5:
            print("Enviando Sinais H_2 e g_2")
            tasks.append(enviar_sinal(H_2, g_2, 30, 30))
        elif random_number == 6:
            print("Enviando Sinais H_2 e g_3")
            tasks.append(enviar_sinal(H_2, g_3, 30, 30))

        # Monitorar o desempenho também como uma tarefa assíncrona
        tasks.append(monitorar_desempenho())

    # Usar asyncio.gather para rodar todas as tarefas simultaneamente
    await asyncio.gather(*tasks)


# Função principal para rodar o evento
async def main():
    await rodar_tarefas()


# Iniciar o loop de eventos do asyncio
if __name__ == "__main__":
    asyncio.run(main())
