import os
import json
import uuid
import random
import time
import subprocess
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm
from faker import Faker

# Imports do LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM, ChatOllama  # Importação correta

# Configuração
OUTPUT_JSONL = "customer_service_dataset.jsonl"
OUTPUT_XLSX = "customer_service_dataset.xlsx"
TOTAL_ENTRIES = 500
MAX_RETRIES = 3
OLLAMA_MODEL = "llama3"  # Você pode alterar para outros modelos como "llama2", "mistral", etc.
OLLAMA_BASE_URL = "http://localhost:11434"  # URL padrão do Ollama

# Inicializando gerador de dados aleatórios
fake = Faker(['pt_BR'])  # Usando locale brasileiro

# Listas de opções para os campos fixos
CUSTOMER_TYPES = ["Individual", "Business"]
SERVICE_CHANNELS = ["Email", "Phone", "WhatsApp", "In Person", "Chat Bot", "Mobile App"]
SERVICE_TYPES = {
    "technical support": ["router issues", "software installation", "hardware repair", "connectivity problems"],
    "complaint": ["delayed service", "quality issues", "rude staff", "billing error", "damaged product"],
    "information request": ["product features", "service details", "pricing inquiries", "availability check"],
    "account management": ["password reset", "profile update", "subscription change", "account closure"],
    "purchase assistance": ["product recommendation", "payment issues", "order tracking", "bulk purchase"],
    "billing inquiry": ["charge explanation", "refund request", "payment plan", "invoice correction"],
    "product return": ["defective product", "wrong item received", "size exchange", "buyer's remorse"],
    "warranty claim": ["product malfunction", "premature failure", "incomplete repair", "missing parts"]
}
SERVICE_CATEGORIES = ["technical", "billing", "product", "account", "general"]
SERVICE_STATUSES = ["resolved", "pending", "open", "escalated", "closed"]
DEPARTMENTS = ["support", "sales", "billing", "customer service", "technical", "returns"]
REPRESENTATIVES = {
    "support": ["Ana Silva", "Bruno Santos", "Carla Oliveira", "Daniel Pereira", "Eduarda Costa"],
    "sales": ["Fernando Lima", "Gabriela Martins", "Henrique Souza", "Isabel Almeida", "João Ferreira"],
    "billing": ["Karina Ribeiro", "Leonardo Gomes", "Marina Dias", "Nelson Rodrigues", "Olívia Carvalho"],
    "customer service": ["Paulo Mendes", "Quitéria Santos", "Rafael Moreira", "Sandra Lopes", "Thiago Barbosa"],
    "technical": ["Ursula Vieira", "Victor Moraes", "Wanda Teixeira", "Xavier Cardoso", "Yasmin Pinto"],
    "returns": ["Zélia Nascimento", "André Campos", "Bianca Nunes", "Carlos Eduardo Ramos", "Débora Machado"]
}

# Função para verificar e instalar o Ollama se necessário
def check_and_install_ollama():
    print("Verificando instalação do Ollama...")
    
    # Verificar se o Ollama está em execução
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            print("Ollama está em execução!")
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            print(f"Modelos disponíveis: {model_names}")
            
            # Verificar se o modelo solicitado já está instalado
            if OLLAMA_MODEL not in str(model_names):
                print(f"Modelo {OLLAMA_MODEL} não encontrado. Iniciando download...")
                download_model()
            else:
                print(f"Modelo {OLLAMA_MODEL} já está disponível.")
                return True
        else:
            print(f"Ollama respondeu com código {response.status_code}.")
            download_model()
    except requests.RequestException as e:
        print(f"Erro ao conectar com Ollama: {str(e)}")
        print("\nVerifique se o Ollama está instalado e em execução:")
        print("1. Instale o Ollama através do site oficial: https://ollama.com/download")
        print("2. Execute o Ollama")
        install_ollama_prompt()
    
    return True

# Função para baixar o modelo automaticamente
def download_model():
    print(f"Tentando baixar o modelo {OLLAMA_MODEL}...")
    try:
        # Verificar se o comando 'ollama' está disponível
        check_command = ["which", "ollama"] if os.name != "nt" else ["where", "ollama"]
        result = subprocess.run(check_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Comando ollama está disponível
            print(f"Baixando modelo {OLLAMA_MODEL}...")
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
            print(f"Modelo {OLLAMA_MODEL} baixado com sucesso!")
            return True
        else:
            print("Comando 'ollama' não encontrado no sistema.")
            install_ollama_prompt()
            return False
    except subprocess.SubprocessError as e:
        print(f"Erro ao baixar o modelo: {str(e)}")
        install_ollama_prompt()
        return False

# Função para mostrar instruções de instalação do Ollama
def install_ollama_prompt():
    print("\nPara instalar o Ollama manualmente:")
    print("1. Acesse https://ollama.com/download")
    print("2. Baixe e instale o Ollama para seu sistema operacional")
    print("3. Após a instalação, abra um terminal e execute: ollama pull", OLLAMA_MODEL)
    print("4. Depois de baixar o modelo, execute este script novamente")
    
    choice = input("\nVocê deseja continuar mesmo sem o Ollama? (s/n): ")
    if choice.lower() != 's':
        exit(1)

# Função para garantir consistência entre categoria, tipo e departamento
def generate_consistent_service_fields():
    # Selecionar categorias e tipos de serviço consistentes
    category = random.choice(SERVICE_CATEGORIES)
    
    if category == "technical":
        service_type = random.choice(["technical support", "warranty claim"])
        department = random.choice(["technical", "support"])
    elif category == "billing":
        service_type = random.choice(["billing inquiry", "complaint"])
        department = "billing"
    elif category == "product":
        service_type = random.choice(["product return", "purchase assistance", "information request"])
        department = random.choice(["sales", "returns", "customer service"])
    elif category == "account":
        service_type = "account management"
        department = random.choice(["account management", "customer service"])
    else:  # "general"
        service_type = random.choice(["information request", "complaint"])
        department = random.choice(DEPARTMENTS)
    
    # Selecionar um subtipo específico
    subtypes = SERVICE_TYPES.get(service_type, ["general inquiry"])
    subtype = random.choice(subtypes)
    
    return category, service_type, department, subtype

# Função para gerar dados fixos aleatórios
def generate_fixed_data():
    # Garantir consistência entre os campos de serviço
    service_category, service_type, department, subtype = generate_consistent_service_fields()
    
    # Gerar data de atendimento (últimos 2 anos)
    service_datetime = fake.date_time_between(start_date="-2y", end_date="now")
    
    # Determinar status do atendimento
    if service_type in ["technical support", "warranty claim"]:
        status_weights = {"resolved": 60, "pending": 20, "open": 10, "escalated": 10}
    elif service_type == "complaint":
        status_weights = {"resolved": 50, "pending": 15, "open": 15, "escalated": 20}
    else:
        status_weights = {"resolved": 70, "pending": 10, "open": 10, "escalated": 5, "closed": 5}
    
    status = random.choices(
        list(status_weights.keys()), 
        weights=list(status_weights.values()), 
        k=1
    )[0]
    
    # Definir data de conclusão se resolvido
    completion_date = ""
    if status in ["resolved", "closed"]:
        days_to_resolve = random.randint(0, 21)  # Até 3 semanas para resolver
        completion_date = (service_datetime + timedelta(days=days_to_resolve)).strftime("%Y-%m-%d")
    
    # Gerar tipo de cliente adequado
    customer_type = "Business" if (service_type in ["account management", "bulk purchase"] and random.random() < 0.7) else random.choice(CUSTOMER_TYPES)
    
    # Gerar nome de cliente apropriado
    customer_name = fake.company() if customer_type == "Business" else fake.name()
    
    # Gerar nome de representante do departamento específico
    representative_name = random.choice(REPRESENTATIVES.get(department, ["Jane Doe"]))
    
    return {
        "customer_id": f"CUST-{str(uuid.uuid4())[:8]}",
        "customer_name": customer_name,
        "phone_number": fake.phone_number(),
        "address": fake.address().replace('\n', ', '),
        "customer_type": customer_type,
        "service_id": f"SRV-{str(uuid.uuid4())[:8]}",
        "service_datetime": service_datetime.strftime("%Y-%m-%dT%H:%M"),
        "service_channel": random.choice(SERVICE_CHANNELS),
        "service_type": service_type,
        "service_category": service_category,
        "service_status": status,
        "representative_name": representative_name,
        "department": department,
        "completion_date": completion_date,
        "subtype": subtype  # Campo auxiliar para o prompt
    }

# Função para configurar LangChain com Ollama
def setup_langchain():
    # Criar o modelo LLM local usando Ollama
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7
    )
    
    # Criar o template do prompt
    prompt_template = PromptTemplate(
        input_variables=[
            "customer_type", "customer_name", "service_type", "service_channel", 
            "department", "service_status", "subtype", "representative_name"
        ],
        template="""
        Você é um assistente especializado em gerar dados realistas para um dataset de atendimento ao cliente.
        
        Por favor, gere informações coerentes e realistas para um atendimento ao cliente com as seguintes características:
        - Nome do cliente: {customer_name}
        - Tipo de cliente: {customer_type}
        - Tipo de serviço: {service_type}
        - Subtipo específico: {subtype}
        - Canal de atendimento: {service_channel}
        - Departamento: {department}
        - Status do atendimento: {service_status}
        - Nome do atendente: {representative_name}
        
        Retorne apenas o seguinte formato JSON, sem explicações adicionais:
        ```json
        {{
            "problem_description": "Uma descrição detalhada e realista do problema que o cliente está enfrentando",
            "applied_solution": "A solução aplicada pelo atendente (ou em andamento se não estiver resolvido)",
            "customer_satisfaction": número entre 0 e 5 que reflete a satisfação do cliente,
            "customer_comment": "Um comentário realista do cliente sobre o atendimento"
        }}
        ```
        
        Regras importantes:
        1. O rating de satisfação deve fazer sentido com o comentário e o status.
        2. Se o status for "pending", "open" ou "escalated", a satisfação deve ser mais baixa (0-3).
        3. Se o status for "resolved" ou "closed", a satisfação deve variar mais (1-5).
        4. Gere descrições e comentários realistas, evitando exageros.
        5. Para empresas, use um tom mais formal nos comentários.
        6. Seja consistente com o tipo de problema descrito pelo subtipo.
        """
    )
    
    # Criar a cadeia LLM
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain

# Função para extrair JSON da resposta do LLM
def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].strip()
        else:
            # Tentar extrair o texto entre { e }
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end] if start >= 0 and end > 0 else response_text
            
        # Tentar carregar o JSON
        return json.loads(json_str)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"Erro ao extrair JSON: {str(e)[:100]}...")
        # Fallback com valores padrão
        return {
            "problem_description": "O cliente relatou um problema.",
            "applied_solution": "O atendente está trabalhando na solução.",
            "customer_satisfaction": 3,
            "customer_comment": "Aguardando resolução."
        }

# Função para gerar um registro completo
def generate_record(langchain):
    # Gerar dados fixos
    fixed_data = generate_fixed_data()
    
    # Usar LangChain para gerar os campos descritivos
    for attempt in range(MAX_RETRIES):
        try:
            llm_response = langchain.run(
                customer_type=fixed_data["customer_type"],
                customer_name=fixed_data["customer_name"],
                service_type=fixed_data["service_type"],
                service_channel=fixed_data["service_channel"],
                department=fixed_data["department"],
                service_status=fixed_data["service_status"],
                subtype=fixed_data["subtype"],
                representative_name=fixed_data["representative_name"]
            )
            
            # Extrair o JSON da resposta
            llm_data = extract_json_from_response(llm_response)
            
            # Verificar campos obrigatórios
            if all(k in llm_data for k in ["problem_description", "applied_solution", "customer_satisfaction", "customer_comment"]):
                break
        except Exception as e:
            print(f"Erro na tentativa {attempt+1}: {str(e)[:100]}...")
            time.sleep(1)
    else:
        # Fallback se todas as tentativas falharem
        llm_data = {
            "problem_description": f"O cliente relatou um problema com {fixed_data['service_type']}.",
            "applied_solution": "O atendente está trabalhando na solução." if fixed_data["service_status"] in ["pending", "open", "escalated"] else "Problema resolvido pelo atendente.",
            "customer_satisfaction": 2 if fixed_data["service_status"] in ["pending", "open", "escalated"] else 4,
            "customer_comment": "Aguardando resolução." if fixed_data["service_status"] in ["pending", "open", "escalated"] else "Atendimento satisfatório."
        }
    
    # Garantir que a satisfação do cliente seja um número inteiro entre 0 e 5
    try:
        satisfaction = int(float(llm_data["customer_satisfaction"]))
        llm_data["customer_satisfaction"] = max(0, min(5, satisfaction))
    except (ValueError, TypeError):
        llm_data["customer_satisfaction"] = 3
    
    # Remover campos auxiliares que não queremos no output final
    fixed_data.pop("subtype", None)
    
    # Combinar dados fixos com dados gerados pelo LLM
    record = {**fixed_data, **llm_data}
    return record

# Função para salvar um registro no arquivo JSONL
def save_to_jsonl(record, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

# Função para converter JSONL para XLSX
def convert_jsonl_to_xlsx(jsonl_path, xlsx_path):
    # Ler o arquivo JSONL
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    
    # Converter para DataFrame e salvar como XLSX
    df = pd.DataFrame(records)
    df.to_excel(xlsx_path, index=False)
    print(f"Dataset convertido para Excel: {xlsx_path}")

# Função principal
def main():
    # Verificar se o Ollama está instalado e em execução, e baixar o modelo se necessário
    check_and_install_ollama()
    
    print(f"Iniciando geração de {TOTAL_ENTRIES} registros de atendimento ao cliente...")
    
    # Certificar-se de que o arquivo JSONL está vazio no início
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)
    
    # Configurar LangChain
    langchain = setup_langchain()
    
    # Gerar os registros com barra de progresso
    with tqdm(total=TOTAL_ENTRIES, desc="Gerando registros") as pbar:
        for i in range(TOTAL_ENTRIES):
            record = generate_record(langchain)
            save_to_jsonl(record, OUTPUT_JSONL)
            pbar.update(1)
    
    print(f"Geração de registros JSONL concluída: {OUTPUT_JSONL}")
    
    # Converter para XLSX
    convert_jsonl_to_xlsx(OUTPUT_JSONL, OUTPUT_XLSX)
    print("Processo completo!")

if __name__ == "__main__":
    # Executar o pipeline
    main()