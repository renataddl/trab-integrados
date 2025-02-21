from flask import Flask, request, jsonify, make_response
import numpy as np
import pandas as pd
import time
import psutil  # Para monitorar
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import threading
import multiprocessing
import json
import asyncio
import json
from datetime import datetime
import psutil

MIN_FREE_MEMORY_MB = 4000

# import resource

app = Flask(__name__)
database = {}

def CGNE(H, g, tol=1e-4, max_iter=20):
    # Inicialização
    m, n = H.shape
    y = np.zeros((m, 1))         # Variável do sistema H H^T y = g
    r = g - H @ (H.T @ y)        # Resíduo do sistema original (g - H f)
    z = r.copy()                 # Resíduo das equações normais (H H^T y = g)
    p = z.copy()                 # Direção de busca
    
    f = H.T @ y                  # Solução inicial f = H^T y
    residuals = [np.linalg.norm(z)]
    
    for i in range(max_iter):
        Hp = H @ (H.T @ p)       # Produto H H^T p
        alpha = (z.T @ z) / (p.T @ Hp)
        
        # Atualiza y e o resíduo
        y += alpha * p
        r -= alpha * Hp          # Resíduo do sistema original
        z = r.copy()             # Resíduo das equações normais
        
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


def CGNR(H, g, tol=1e-6, max_iter=1000):
     # Inicialização
    f = np.zeros((H.shape[1]))
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    
    residuals = [np.linalg.norm(z)]
    
    for i in range(max_iter):
        w = H @ p
        alpha = np.dot(z, z) / np.dot(w, w)
        
        # Atualiza solução e resíduo
        f += alpha * p
        r -= alpha * w
        
        # Calcula novo resíduo das equações normais
        z_next = H.T @ r
        
        # Verifica convergência
        residual_norm = np.linalg.norm(z_next)
        residuals.append(residual_norm)
        
        if residual_norm < tol:
            break
            
        # Atualiza parâmetros para próxima iteração
        beta = np.dot(z_next, z_next) / np.dot(z, z)
        p = z_next + beta * p
        z = z_next
    
    return f, residuals



def gerar_imagem(dados, titulo="ABS", nome_arquivo="imagem.png"):
    dados_rotacionados = rotate(dados, angle=-90, reshape=True, mode="nearest")
    dados_rotacionados = np.fliplr(dados_rotacionados)
    plt.imshow(np.abs(dados_rotacionados), cmap="gray", interpolation="nearest")
    plt.title(titulo)
    plt.savefig(nome_arquivo)


def process(data):
    user = data["user"]
    modelo = data["modelo"]
    alg = data["alg"]

    while not check_memory():
        print("Waiting for memory...")
        time.sleep(1)


    H = pd.read_csv("cliente/" + data["H"], header=None, delimiter=",").to_numpy()
    g = pd.read_csv("cliente/" + data["g"], header=None, delimiter=",").to_numpy()

    start_time = time.time()
    if alg == "CGNR":
        print("Utilizando o CGNR")
        resultado, iteracoes = CGNR(H, g)
    else:
        print("Utilizando o CGNE")
        resultado, iteracoes = CGNE(H, g)
    end_time = time.time()

    # Gerar imagem e salvar
    heatmap_data = resultado.reshape(modelo["rows"], modelo["cols"])
    nome_arquivo = f"imagem_{user}_{int(end_time)}.png"
    # thread =  threading.Thread(gerar_imagem(heatmap_data, nome_arquivo=nome_arquivo))
    # thread = threading.Thread(target=gerar_imagem, args=(heatmap_data,), kwargs={'nome_arquivo': nome_arquivo})
    # thread.start()
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
    }

    # Salvando a resposta como JSON em um arquivo
    timestamp = int(time.time())
    nome_json = f"resposta_{user}_{timestamp}.json"
    with open(nome_json, "w") as json_file:
        json.dump(resposta_json, json_file, indent=4)

    return jsonify(resposta_json)


def check_memory():
    mem = psutil.virtual_memory()
    available_mb = mem.available / (1024 * 1024)
    total_mb = mem.total / (1024 * 1024)
    used_mb = (mem.total - mem.available) / (1024 * 1024)

    print(
        f"Used memory: {used_mb:.2f} MB | Free memory: {available_mb:.2f} MB | Total memory: {total_mb:.2f} MB "
        f"({(available_mb / total_mb) * 100:.2f}% free)"
    )

    return available_mb > MIN_FREE_MEMORY_MB


@app.route("/reconstruir_imagem", methods=["POST"])
async def reconstruir_imagem():
    data = request.json
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
