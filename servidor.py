import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import time
import psutil
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import multiprocessing
import asyncio
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('server.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constantes
MIN_FREE_MEMORY_MB = 4000
app = Flask(__name__)

# Função para criar pastas necessárias
def criar_pastas():
    pastas = ["imagens", "relatorios"]
    for pasta in pastas:
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            logger.info(f"Pasta '{pasta}' criada.")

# Cria as pastas ao iniciar o servidor
criar_pastas()

class MemoryMonitor:
    def __init__(self):
        self.last_log_time = time.time()
        self.log_interval = 60  # Segundos entre logs de memória

    def check_memory(self):
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        
        # Log a cada intervalo definido
        if time.time() - self.last_log_time > self.log_interval:
            logger.info(
                f"Memory Status | Available: {available_mb:.2f}MB, "
                f"Used: {mem.percent}%"
            )
            self.last_log_time = time.time()
        
        return available_mb > MIN_FREE_MEMORY_MB

memory_monitor = MemoryMonitor()

def CGNE(H, g, tol=1e-4, max_iter=20):
    # Inicialização
    m, n = H.shape
    y = np.zeros((m, 1))  # Variável do sistema H H^T y = g
    r = g - H @ (H.T @ y)  # Resíduo do sistema original (g - H f)
    z = r.copy()  # Resíduo das equações normais (H H^T y = g)
    p = z.copy()  # Direção de busca

    f = H.T @ y  # Solução inicial f = H^T y
    residuals = [np.linalg.norm(z)]

    for i in range(max_iter):
        Hp = H @ (H.T @ p)  # Produto H H^T p
        alpha = (z.T @ z) / (p.T @ Hp)

        # Atualiza y e o resíduo
        y += alpha * p
        r -= alpha * Hp  # Resíduo do sistema original
        z = r.copy()  # Resíduo das equações normais

        # Atualiza f (solução do problema original)
        f = H.T @ y

        # Verifica convergência
        residual_norm = np.linalg.norm(z)
        residuals.append(residual_norm)
        if residual_norm < tol:
            break

        # Atualiza beta e a direção de busca
        beta = (z.T @ z) / (residuals[-2] ** 2)
        p = z + beta * p

    return f, residuals

def CGNR(H, g, tol=1e-4, max_iter=20):
    # Certifique-se de que H e g são arrays NumPy
    H = np.array(H, dtype=float)
    g = np.array(g, dtype=float).reshape(-1, 1)  # Transforma g em um vetor coluna

    # Inicialização
    m, n = H.shape
    f = np.zeros((n, 1))  # Solução inicial (vetor coluna)
    r = g - np.dot(H, f)  # Resíduo do sistema original (g - H @ f)
    z = np.dot(H.T, r)  # Resíduo das equações normais (H^T H f = H^T g)
    p = z.copy()  # Direção de busca

    residuals = [np.linalg.norm(z)]

    for i in range(max_iter):
        Hp = np.dot(H, p)  # Produto H p
        alpha = np.dot(z.T, z) / np.dot(Hp.T, Hp)

        # Atualiza f e o resíduo
        f += alpha * p
        r -= alpha * Hp
        z = np.dot(H.T, r)  # Resíduo das equações normais

        # Verifica convergência
        residual_norm = np.linalg.norm(z)
        residuals.append(residual_norm)
        if residual_norm < tol:
            break

        # Atualiza beta e a direção de busca
        beta = np.dot(z.T, z) / (residuals[-2] ** 2)
        p = z + beta * p

    return f, residuals

def gerar_imagem(dados, titulo="ABS", nome_arquivo="imagem.png"):
    caminho_arquivo = os.path.join("imagens", nome_arquivo)  # Caminho completo
    dados_rotacionados = rotate(dados, angle=-90, reshape=True, mode="nearest")
    dados_rotacionados = np.fliplr(dados_rotacionados)
    plt.imshow(np.abs(dados_rotacionados), cmap="gray", interpolation="nearest")
    plt.title(titulo)
    plt.savefig(caminho_arquivo)
    plt.close()  # Fecha a figura para liberar memória

def process(data):
    user = data["user"]
    modelo = data["modelo"]
    alg = data["alg"]
    sinais = data["sinais"]

    while not memory_monitor.check_memory():
        logger.info("Waiting for memory...")
        time.sleep(1)

    try:
        # Carrega H e g como arrays NumPy
        H = pd.read_csv("cliente/dados/" + data["H"], header=None, delimiter=",").to_numpy()
        g = pd.read_csv("cliente/dados/" + data["g"], header=None, delimiter=",").to_numpy()
    except Exception as e:
        logger.error(f"Erro ao carregar arquivos: {e}")
        return jsonify({"error": "Erro ao carregar arquivos. Verifique se os arquivos são válidos."}), 400

    start_time = time.time()
    if alg == "CGNR":
        logger.info(f"Utilizando o CGNR para o usuário {user}")
        resultado, iteracoes = CGNR(H, g)
    else:
        logger.info(f"Utilizando o CGNE para o usuário {user}")
        resultado, iteracoes = CGNE(H, g)
    end_time = time.time()

    # Gerar imagem e salvar
    heatmap_data = resultado.reshape(modelo["rows"], modelo["cols"])
    nome_arquivo = f"imagem_{user}_{int(end_time)}.png"
    process = multiprocessing.Process(
        target=gerar_imagem, args=(heatmap_data,), kwargs={"nome_arquivo": nome_arquivo}
    )
    process.start()

    # Informações adicionais
    tempo_execucao = end_time - start_time
    tamanho_imagem = heatmap_data.shape

    # Criando o JSON com os resultados
    resposta_json = {
        "usuario": user,
        "iteracoes": iteracoes,
        "tempo_execucao": tempo_execucao,
        "tamanho_imagem": tamanho_imagem,
        "arquivo_imagem": nome_arquivo,
        "algoritmo": alg,
        "sinais": sinais,
    }

    # Salvando a resposta como JSON em um arquivo
    timestamp = int(time.time())
    nome_json = f"resposta_{user}_{timestamp}.json"
    caminho_json = os.path.join("relatorios", nome_json)
    with open(caminho_json, "w") as json_file:
        json.dump(resposta_json, json_file, indent=4)

    logger.info(f"Requisição do usuário {user} concluída em {tempo_execucao:.2f} segundos.")
    return jsonify(resposta_json)

@app.route("/reconstruir_imagem", methods=["POST"])
async def reconstruir_imagem():
    data = request.json
    logger.info(f"Requisição recebida do usuário {data['user']}")
    return await asyncio.to_thread(process, data)

@app.route("/status_servidor", methods=["GET"])
def status_servidor():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    return jsonify(
        {
            "cpu_percent": cpu_percent,
            "memory_used": memory_info.used,
            "memory_total": memory_info.total,
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)