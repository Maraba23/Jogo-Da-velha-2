import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import time
import json
import copy
import math
import tempfile, os, torch
import ai  # Importa o módulo de IA do jogo da velha avançado
import sys
from colorama import Fore, Style, init

def exibir_status(episode, total_reward_x, total_reward_o, moves_made, stats_x, stats_o, clear_previous=True):
    import sys
    import os
    
    
    # Tenta usar colorama para cores (se estiver instalado)
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)  # Autoreset ajuda a evitar problemas com cores
        # Definição das cores
        LABEL_COLOR = Fore.CYAN
        VALUE_COLOR = Fore.WHITE
        HIGHLIGHT = Style.BRIGHT
        RESET = Style.RESET_ALL
    except ImportError:
        LABEL_COLOR = VALUE_COLOR = HIGHLIGHT = RESET = ""
    
    # Calcula estatísticas
    total_games = stats_x['x_wins'] + stats_o['o_wins'] + stats_x['draws']
    pct_x_wins = (stats_x['x_wins'] / total_games) * 100 if total_games else 0
    pct_o_wins = (stats_o['o_wins'] / total_games) * 100 if total_games else 0
    pct_draws  = (stats_x['draws']  / total_games) * 100 if total_games else 0

    # Formatação da saída com caracteres de escape para sobrescrever
    output = (
        f"\r{LABEL_COLOR}🧠 Episódio:{RESET} {HIGHLIGHT}{episode:6d}{RESET} | "
        f"{LABEL_COLOR}🎯 Jogadas:{RESET} {HIGHLIGHT}{moves_made:3d}{RESET} | "
        f"{LABEL_COLOR}🏆 Recompensas - X:{RESET} {HIGHLIGHT}{total_reward_x:7.2f}{RESET} | "
        f"{LABEL_COLOR}O:{RESET} {HIGHLIGHT}{total_reward_o:7.2f}{RESET} || "
        f"{LABEL_COLOR}📊 Vitórias - X:{RESET} {HIGHLIGHT}{stats_x['x_wins']:5d}{RESET} "
        f"({pct_x_wins:5.1f}%) | "
        f"{LABEL_COLOR}O:{RESET} {HIGHLIGHT}{stats_o['o_wins']:5d}{RESET} "
        f"({pct_o_wins:5.1f}%) | "
        f"{LABEL_COLOR}Empates:{RESET} {HIGHLIGHT}{stats_x['draws']:5d}{RESET} "
        f"({pct_draws:5.1f}%)"
    )
    
    # Escreve e força a atualização imediatamente
    sys.stdout.write(output)
    sys.stdout.flush()
    
    # Desativa outros prints de status padrão (adicione isso ao seu código principal)
    # Comentar ou remover os prints existentes como:
    # print(f"Episódio {episode} - Recompensa X: {total_reward_x}, Recompensa O: {total_reward_o}, Jogadas: {moves_made}")
    # print(f"Status: X ganhou {stats_x['x_wins']}, O ganhou {stats_o['o_wins']}, Empates {stats_x['draws']}")


# Configurações de treinamento
CONFIG = {
    "learning_rate": 0.0001,      # Taxa de aprendizado
    "gamma": 0.95,                # Fator de desconto para recompensas futuras
    "epsilon_start": 1.0,         # Exploração inicial (1 = 100% aleatório)
    "epsilon_end": 0.05,          # Exploração final (0.05 = 5% aleatório)
    "epsilon_decay": 40000,       # Quantas etapas para reduzir epsilon
    "batch_size": 64,             # Tamanho do lote para treinamento
    "memory_size": 20000,         # Tamanho da memória de experiência
    "target_update": 500,         # Frequência para atualizar a rede alvo
    "save_interval": 5000,        # Frequência para salvar o modelo
    "checkpoint_dir": "model_checkpoints",  # Diretório para salvar checkpoints
    "episodes": 12_000            # Número de jogos a serem treinados
}

# Garante que o diretório de checkpoints existe
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# Classe para representar o estado do jogo
class GameState:
    def __init__(self):
        # Inicializa tabuleiro principal e sub-tabuleiros (todos vazios no começo)
        self.main_board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.won_sub_boards = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # X sempre começa
        self.next_sub_board = None  # Qualquer um pode ser jogado inicialmente
        self.game_over = False
    
    def reset(self):
        """Reinicia o jogo para um novo episódio"""
        self.main_board = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.won_sub_boards = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # X sempre começa
        self.next_sub_board = None  # Qualquer um pode ser jogado inicialmente
        self.game_over = False
        return self.get_state_tensor()

    def get_state_tensor(self):
        """Converte o estado do jogo para um tensor compatível com a rede neural"""
        # Criamos um tensor 3D de tamanho 3x9x9 (3 camadas, 9x9 grid)
        # Camada 0: Posições onde está X (1 onde tiver X, 0 caso contrário)
        # Camada 1: Posições onde está O (1 onde tiver O, 0 caso contrário)
        # Camada 2: Posições onde é possível jogar (1 onde for possível, 0 caso contrário)
        
        state = np.zeros((3, 9, 9), dtype=np.float32)
        
        # Preenchendo as camadas 0 e 1 (posições de X e O)
        for row in range(3):
            for col in range(3):
                for sub_row in range(3):
                    for sub_col in range(3):
                        # Calculando a posição linear no grid 9x9
                        grid_row = row * 3 + sub_row
                        grid_col = col * 3 + sub_col
                        
                        # Marcando posições de X e O
                        if self.main_board[row][col][sub_row][sub_col] == 'X':
                            state[0, grid_row, grid_col] = 1
                        elif self.main_board[row][col][sub_row][sub_col] == 'O':
                            state[1, grid_row, grid_col] = 1
        
        # Preenchendo a camada 2 (jogadas possíveis)
        possible_moves = self.get_valid_moves()
        for move in possible_moves:
            row, col, sub_row, sub_col = move
            grid_row = row * 3 + sub_row
            grid_col = col * 3 + sub_col
            state[2, grid_row, grid_col] = 1
            
        return torch.FloatTensor(state)
    
    def get_valid_moves(self):
        """Retorna todas as jogadas válidas no formato [(row, col, sub_row, sub_col), ...]"""
        valid_moves = []
        
        if self.game_over:
            return valid_moves
            
        # Se o próximo sub-tabuleiro é None, qualquer sub-tabuleiro não completo pode ser jogado
        if self.next_sub_board is None:
            for row in range(3):
                for col in range(3):
                    # Verifica se o sub-tabuleiro já foi ganho
                    if self.won_sub_boards[row][col] is not None:
                        continue
                    
                    # Verifica se o sub-tabuleiro está completo
                    full = True
                    for sub_row in range(3):
                        for sub_col in range(3):
                            if self.main_board[row][col][sub_row][sub_col] is None:
                                valid_moves.append((row, col, sub_row, sub_col))
                                full = False
                    
                    if full and self.won_sub_boards[row][col] is None:
                        self.won_sub_boards[row][col] = 'draw'
        else:
            # Se temos um sub-tabuleiro específico para jogar
            row, col = self.next_sub_board
            
            # Verifica se o sub-tabuleiro já foi ganho ou está completo
            if self.won_sub_boards[row][col] is not None:
                # Se o sub-tabuleiro já foi ganho, podemos jogar em qualquer outro
                return self.get_valid_moves()  # Chama recursivamente sem próximo sub-tabuleiro definido
            
            # Adiciona todas as células vazias neste sub-tabuleiro
            for sub_row in range(3):
                for sub_col in range(3):
                    if self.main_board[row][col][sub_row][sub_col] is None:
                        valid_moves.append((row, col, sub_row, sub_col))
            
            # Se não houver jogadas válidas neste sub-tabuleiro, liberamos todos
            if not valid_moves:
                self.next_sub_board = None
                return self.get_valid_moves()
                
        return valid_moves

    def make_move(self, move):
        """
        Realiza uma jogada no tabuleiro e retorna (nova_observação, recompensa, terminado)
        """
        row, col, sub_row, sub_col = move
        reward = 0
        
        # Verifica se a jogada é válida
        valid_moves = self.get_valid_moves()
        if move not in valid_moves or self.game_over:
            return self.get_state_tensor(), -10, True  # Penalidade por jogada inválida
        
        # Marca a posição
        self.main_board[row][col][sub_row][sub_col] = self.current_player
        
        # ------- INÍCIO DO REWARD SHAPING -------
        # Apenas analisa se o sub-tabuleiro ainda não foi ganho
        if self.won_sub_boards[row][col] is None:
            # Determina o oponente
            opponent = 'X' if self.current_player == 'O' else 'O'
            
            # Verifica todas as linhas horizontais, verticais e diagonais
            # do sub-tabuleiro onde a peça foi colocada
            sub_board = self.main_board[row][col]
            
            # Linhas horizontais
            for r in range(3):
                line = [sub_board[r][0], sub_board[r][1], sub_board[r][2]]
                # Bônus se o jogador tem duas peças em linha com uma vazia
                if line.count(self.current_player) == 2 and line.count(None) == 1:
                    reward += 0.2
                # Penalidade se o oponente tinha duas peças em linha e não foi bloqueado
                if line.count(opponent) == 2 and line.count(None) == 1:
                    reward -= 0.2
            
            # Colunas verticais
            for c in range(3):
                line = [sub_board[0][c], sub_board[1][c], sub_board[2][c]]
                if line.count(self.current_player) == 2 and line.count(None) == 1:
                    reward += 0.2
                if line.count(opponent) == 2 and line.count(None) == 1:
                    reward -= 0.2
            
            # Diagonal principal
            line = [sub_board[0][0], sub_board[1][1], sub_board[2][2]]
            if line.count(self.current_player) == 2 and line.count(None) == 1:
                reward += 0.2
            if line.count(opponent) == 2 and line.count(None) == 1:
                reward -= 0.2
            
            # Diagonal secundária
            line = [sub_board[0][2], sub_board[1][1], sub_board[2][0]]
            if line.count(self.current_player) == 2 and line.count(None) == 1:
                reward += 0.2
            if line.count(opponent) == 2 and line.count(None) == 1:
                reward -= 0.2
        # ------- FIM DO REWARD SHAPING -------
        
        # Verifica se o sub-tabuleiro foi ganho
        sub_tabuleiro_ganho = False
        if ai.sub_winner(self.main_board[row][col], self.current_player):
            self.won_sub_boards[row][col] = self.current_player
            reward += 1  # Recompensa por ganhar um sub-tabuleiro
            sub_tabuleiro_ganho = True
        
        # Determina o próximo sub-tabuleiro
        next_row, next_col = sub_row, sub_col
        
        # Se o próximo sub-tabuleiro já está ganho ou completo, liberamos para qualquer sub-tabuleiro
        if (self.won_sub_boards[next_row][next_col] is not None or 
            ai.is_sub_board_full(self.main_board[next_row][next_col])):
            self.next_sub_board = None
        else:
            self.next_sub_board = (next_row, next_col)
        
        # Verifica se o jogo acabou
        x_wins = sum(row.count('X') for row in self.won_sub_boards)
        o_wins = sum(row.count('O') for row in self.won_sub_boards)
        
        # Jogo termina se todos os sub-tabuleiros forem preenchidos ou se um jogador ganhar 5
        if x_wins + o_wins == 9 or x_wins > 4 or o_wins > 4:
            self.game_over = True
            
            # Recompensa final com base no resultado do jogo
            if self.current_player == 'X':
                if x_wins > o_wins:
                    # Vitória: +10 pontos
                    reward += 10
                    # Não precisamos adicionar +1 por sub-tabuleiro vencido pois já fizemos isso acima
                    # quando cada sub-tabuleiro foi conquistado
                    
                    # Subtrai -1 por cada sub-tabuleiro perdido
                    reward -= o_wins
                elif x_wins < o_wins:
                    # Derrota: -10 pontos
                    reward -= 10
                    # Já adicionamos +1 por cada sub-tabuleiro vencido quando aconteceu
                    
                    # Subtrai -1 por cada sub-tabuleiro perdido
                    reward -= o_wins
            else:  # O jogador é 'O'
                if o_wins > x_wins:
                    # Vitória: +10 pontos
                    reward += 10
                    # Não precisamos adicionar +1 por sub-tabuleiro vencido pois já fizemos isso acima
                    
                    # Subtrai -1 por cada sub-tabuleiro perdido
                    reward -= x_wins
                elif o_wins < x_wins:
                    # Derrota: -10 pontos
                    reward -= 10
                    # Já adicionamos +1 por cada sub-tabuleiro vencido quando aconteceu
                    
                    # Subtrai -1 por cada sub-tabuleiro perdido
                    reward -= x_wins
        
        # Clip de reward para ficar entre -1.0 e 1.0
        reward /= 10.0  # Normaliza a recompensa
        reward = max(-1.0, min(1.0, reward))
        
        # Troca o jogador
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return self.get_state_tensor(), reward, self.game_over    

# Classe para a memória de experiência (armazena as transições para treinamento)
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, next_state, reward, done):
        """Adiciona uma transição à memória"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Amostra aleatória da memória"""
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

# Classe para a rede neural Q (Deep Q-Network)
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        # convoluções menores
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(128)

        flat = 128 * 9 * 9  # 128 canais, 9×9

        # cabeça Advantage
        self.fc_a1 = nn.Linear(flat, 256)
        self.fc_a2 = nn.Linear(256, 81)   # 81 ações

        # cabeça Value
        self.fc_v1 = nn.Linear(flat, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        a = F.relu(self.fc_a1(x))
        a = self.fc_a2(a)

        v = F.relu(self.fc_v1(x))
        v = self.fc_v2(v).expand_as(a)

        q = v + a - a.mean(1, keepdim=True)
        return q

# Função para selecionar uma ação epsilon-greedy
def select_action(state, policy_net, epsilon, valid_moves, device):
    """
    Seleciona uma ação usando a política epsilon-greedy
    """
    # Convertendo movimentos válidos para índices no grid 9x9
    valid_indices = []
    for move in valid_moves:
        row, col, sub_row, sub_col = move
        index = (row * 3 + sub_row) * 9 + (col * 3 + sub_col)
        valid_indices.append(index)
    
    # Se não houver movimentos válidos, retorna None
    if not valid_indices:
        return None
    
    # Com probabilidade epsilon, escolhe uma ação aleatória
    if random.random() < epsilon:
        return random.choice(valid_moves)
    
    # Caso contrário, escolhe a melhor ação de acordo com a rede
    with torch.no_grad():
        state = state.to(device)
        q_values = policy_net(state.unsqueeze(0)).squeeze()
        
        # Filtra apenas as ações válidas
        valid_q_values = [(i, q_values[i].item()) for i in valid_indices]
        
        # Escolhe a ação com maior valor Q entre as válidas
        best_index, _ = max(valid_q_values, key=lambda x: x[1])
        
        # Converte de volta para o formato (row, col, sub_row, sub_col)
        grid_row = best_index // 9
        grid_col = best_index % 9
        row = grid_row // 3
        sub_row = grid_row % 3
        col = grid_col // 3
        sub_col = grid_col % 3
        
        return (row, col, sub_row, sub_col)

# Função para otimizar o modelo
def optimize_model(policy_net, target_net, optimizer, memory, device, batch_size, gamma):
    """
    Realiza uma etapa de otimização no modelo
    """
    if len(memory) < batch_size:
        return
    
    # Amostra uma transição da memória
    transitions = memory.sample(batch_size)
    
    # Converte a amostra para o formato batch
    batch = list(zip(*transitions))
    
    # Extrai os componentes
    state_batch = torch.stack([s.to(device, non_blocking=True) for s in batch[0]])
    action_batch = batch[1]
    next_state_batch = torch.stack([ns.to(device, non_blocking=True) for ns in batch[2]])
    reward_batch = torch.tensor(batch[3], device=device, dtype=torch.float32)
    done_batch = torch.tensor(batch[4], device=device, dtype=torch.bool)
    
    # Calcula os valores Q atuais para as ações tomadas
    q_values = policy_net(state_batch)
    action_indices = []
    
    for i, action in enumerate(action_batch):
        if action is None:
            action_indices.append(0)  # Valor padrão, será mascarado depois
        else:
            row, col, sub_row, sub_col = action
            index = (row * 3 + sub_row) * 9 + (col * 3 + sub_col)
            action_indices.append(index)
    
    action_indices = torch.LongTensor(action_indices).to(device)
    q_values = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    
    # Calcula os valores Q esperados usando a rede alvo
    next_q_values = torch.zeros(batch_size, device=device)
    # ------------------------------------------------------------------
    # next_q_values com máscara, sem depender do objeto env
    # ------------------------------------------------------------------
    with torch.no_grad():
        q_target_all = target_net(next_state_batch)        # shape (B, 81)

        # Canal 2 contém jogadas possíveis; transformamos em máscara booleana
        # next_state_batch shape: (B, 3, 9, 9)
        valid_mask = next_state_batch[:, 2, :, :]          # (B, 9, 9)
        valid_mask = valid_mask.view(valid_mask.size(0), -1).bool()  # (B, 81)

        # Atribui -inf às ações ilegais
        q_target_all[~valid_mask] = -float('inf')

        # Double-DQN opcional: usa policy para escolher a-max
        # a_max = policy_net(next_state_batch).masked_fill(~valid_mask, -float('inf')).argmax(1, keepdim=True)
        # next_q_values = q_target_all.gather(1, a_max).squeeze(1)

        # DQN simples:
        next_q_values = q_target_all.max(1)[0]
    
    # Mascara os estados terminais
    next_q_values[done_batch] = 0.0
    
    # Calcula os valores Q esperados
    expected_q_values = reward_batch + gamma * next_q_values
    
    # Calcula a perda Huber
    loss = F.smooth_l1_loss(q_values, expected_q_values)
    
    # Otimiza o modelo
    optimizer.zero_grad()
    loss.backward()
    
    # Limitador de gradiente (para estabilidade)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    
    optimizer.step()
    
    return loss.item()

# Função para salvar o modelo e estado do treinamento
def save_checkpoint(policy_net, target_net, optimizer, memory, episode, step, stats, filename):
    """
    Salva o estado atual do treinamento
    """
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory': memory.memory,
        'memory_position': memory.position,
        'episode': episode,
        'step': step,
        'stats': stats
    }, filename)

# Função para carregar o checkpoint
def load_checkpoint(filename, policy_net, target_net, optimizer, memory, device):
    """
    Carrega um checkpoint salvo
    """
    if os.path.isfile(filename):
        print(f"Carregando checkpoint '{filename}'...")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        memory.memory = checkpoint['memory']
        memory.position = checkpoint['memory_position']
        for i, (s, a, ns, r, d) in enumerate(memory.memory):
            memory.memory[i] = (s.cpu(), a, ns.cpu(), r, d)
        
        episode = checkpoint['episode']
        step = checkpoint['step']
        stats = checkpoint['stats']
        
        print(f"Checkpoint carregado. Continuando do episódio {episode}, passo {step}")
        
        return episode, step, stats
    else:
        print(f"Nenhum checkpoint encontrado em '{filename}'")
        return 0, 0, {'x_wins': 0, 'o_wins': 0, 'draws': 0, 'episodes': 0, 'epsilon_x': [], 'epsilon_o': []}

# Função principal de treinamento com self-play
def train_self_play():
    """
    Função principal para treinar o modelo usando self-play
    """
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Inicialização do ambiente
    env = GameState()
    
    # Inicialização das memórias separadas para X e O
    memory_x = ReplayMemory(CONFIG["memory_size"])
    memory_o = ReplayMemory(CONFIG["memory_size"])
    
    # Inicialização das redes para X e O
    policy_net_x = DQN().to(device)
    target_net_x = DQN().to(device)
    target_net_x.load_state_dict(policy_net_x.state_dict())
    target_net_x.eval()  # Modo de avaliação (não treinamento)
    
    policy_net_o = DQN().to(device)
    target_net_o = DQN().to(device)
    target_net_o.load_state_dict(policy_net_o.state_dict())
    target_net_o.eval()  # Modo de avaliação (não treinamento)
    
    # Inicialização dos otimizadores
    optimizer_x = optim.Adam(policy_net_x.parameters(), lr=CONFIG["learning_rate"])
    optimizer_o = optim.Adam(policy_net_o.parameters(), lr=CONFIG["learning_rate"])
    
    # Carregamento do checkpoint (se existir)
    checkpoint_path_x = os.path.join(CONFIG["checkpoint_dir"], "latest_checkpoint_x.pt")
    checkpoint_path_o = os.path.join(CONFIG["checkpoint_dir"], "latest_checkpoint_o.pt")
    
    # Tenta carregar checkpoints
    start_episode_x, step_x, stats_x = load_checkpoint(checkpoint_path_x, policy_net_x, target_net_x, optimizer_x, memory_x, device)
    start_episode_o, step_o, stats_o = load_checkpoint(checkpoint_path_o, policy_net_o, target_net_o, optimizer_o, memory_o, device)
    
    # Usa o maior episódio como ponto de partida
    start_episode = max(start_episode_x, start_episode_o)
    
    # Se não houver estatísticas carregadas, inicializa-as
    if not stats_x:
        stats_x = {
            'x_wins': 0,
            'o_wins': 0,
            'draws': 0,
            'episodes': 0,
            'epsilon_x': []
        }
    
    if not stats_o:
        stats_o = {
            'x_wins': 0,
            'o_wins': 0,
            'draws': 0,
            'episodes': 0,
            'epsilon_o': []
        }
    
    # Para rastrear os passos de otimização
    if step_x == 0:
        step_x = 1
    if step_o == 0:
        step_o = 1
    
    # Loop principal de treinamento
    for episode in range(start_episode, CONFIG["episodes"]):
        # Reinicia o ambiente
        state = env.reset()
        total_reward_x = 0
        total_reward_o = 0
        game_over = False
        moves_made = 0
        
        # Estado anterior e ação para armazenar na memória
        prev_state_x = None
        prev_action_x = None
        prev_state_o = None
        prev_action_o = None
        
        # Loop do jogo (até terminar)
        while not game_over:
            # Calcula epsilon para este passo (exploração vs aproveitamento)
            epsilon_x = CONFIG["epsilon_end"] + (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]) * \
                      math.exp(-1. * step_x / CONFIG["epsilon_decay"])
            epsilon_o = CONFIG["epsilon_end"] + (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]) * \
                      math.exp(-1. * step_o / CONFIG["epsilon_decay"])
            
            # Obtém as jogadas válidas
            valid_moves = env.get_valid_moves()
            
            # Se não houver jogadas possíveis, terminar episódio de forma limpa
            if not valid_moves:
                game_over = True
                break
            
            # Seleciona a ação usando a rede adequada para o jogador atual
            if env.current_player == 'X':
                action = select_action(state, policy_net_x, epsilon_x, valid_moves, device)
            else:  # O
                action = select_action(state, policy_net_o, epsilon_o, valid_moves, device)
            
            # Segurança extra – select_action NUNCA deveria devolver None aqui
            if action is None:
                game_over = True
                break
            
            # Armazena o estado e ação atuais para o jogador atual
            if env.current_player == 'X':
                prev_state_x = state
                prev_action_x = action
            else:  # O
                prev_state_o = state
                prev_action_o = action
            
            # Executa a ação no ambiente
            next_state, reward, done = env.make_move(action)
            
            # Armazena a recompensa para estatísticas
            if env.current_player == 'O':  # Jogador que acabou de mudar (era X)
                total_reward_x += reward
                
                # Armazena transição na memória de X (se já tivermos estado e ação anteriores)
                if prev_state_x is not None and prev_action_x is not None:
                    memory_x.push(prev_state_x, prev_action_x, next_state, reward, done)
                
                # Otimiza o modelo X
                if len(memory_x) > CONFIG["batch_size"]:
                    loss_x = optimize_model(policy_net_x, target_net_x, optimizer_x, memory_x, device, 
                                         CONFIG["batch_size"], CONFIG["gamma"])
                    
                    # Atualiza a rede alvo X periodicamente
                    if step_x % CONFIG["target_update"] == 0:
                        target_net_x.load_state_dict(policy_net_x.state_dict())
                    
                    # Incrementa o contador de passos para X
                    step_x += 1
            else:  # Jogador que acabou de mudar (era O)
                total_reward_o += reward
                
                # Armazena transição na memória de O (se já tivermos estado e ação anteriores)
                if prev_state_o is not None and prev_action_o is not None:
                    memory_o.push(prev_state_o, prev_action_o, next_state, reward, done)
                
                # Otimiza o modelo O
                if len(memory_o) > CONFIG["batch_size"]:
                    loss_o = optimize_model(policy_net_o, target_net_o, optimizer_o, memory_o, device, 
                                         CONFIG["batch_size"], CONFIG["gamma"])
                    
                    # Atualiza a rede alvo O periodicamente
                    if step_o % CONFIG["target_update"] == 0:
                        target_net_o.load_state_dict(policy_net_o.state_dict())
                    
                    # Incrementa o contador de passos para O
                    step_o += 1
            
            # Atualiza o estado atual
            state = next_state
            game_over = done
            moves_made += 1
            
            # Salva os modelos periodicamente
            if (step_x % CONFIG["save_interval"] == 0) or (step_o % CONFIG["save_interval"] == 0):
                # Salva o modelo X
                save_checkpoint(policy_net_x, target_net_x, optimizer_x, memory_x, episode, step_x, stats_x,
                               os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_x_step_{step_x}.pt"))
                save_checkpoint(policy_net_x, target_net_x, optimizer_x, memory_x, episode, step_x, stats_x,
                               checkpoint_path_x)
                
                # Salva o modelo O
                save_checkpoint(policy_net_o, target_net_o, optimizer_o, memory_o, episode, step_o, stats_o,
                               os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_o_step_{step_o}.pt"))
                save_checkpoint(policy_net_o, target_net_o, optimizer_o, memory_o, episode, step_o, stats_o,
                               checkpoint_path_o)
                
                # Exibe estatísticas
                win_rate_x = stats_x['x_wins'] / max(1, stats_x['episodes']) * 100
                win_rate_o = stats_o['o_wins'] / max(1, stats_o['episodes']) * 100
                draw_rate = stats_x['draws'] / max(1, stats_x['episodes']) * 100
                
                print(f"Passo X: {step_x}, Passo O: {step_o}")
                print(f"Epsilon X: {epsilon_x:.4f}, Epsilon O: {epsilon_o:.4f}")
                print(f"Vitórias X: {win_rate_x:.1f}%, Vitórias O: {win_rate_o:.1f}%, Empates: {draw_rate:.1f}%")
                
                # Salva estatísticas em arquivos JSON
                with open(os.path.join(CONFIG["checkpoint_dir"], "stats_x.json"), 'w') as f:
                    json.dump(stats_x, f)
                with open(os.path.join(CONFIG["checkpoint_dir"], "stats_o.json"), 'w') as f:
                    json.dump(stats_o, f)
        
        # Fim do episódio: atualiza as estatísticas
        # Determina quem ganhou
        x_wins = sum(row.count('X') for row in env.won_sub_boards)
        o_wins = sum(row.count('O') for row in env.won_sub_boards)
        
        if x_wins > o_wins:
            stats_x['x_wins'] += 1
            stats_o['x_wins'] += 1
        elif o_wins > x_wins:
            stats_x['o_wins'] += 1
            stats_o['o_wins'] += 1
        else:
            stats_x['draws'] += 1
            stats_o['draws'] += 1
        
        stats_x['episodes'] += 1
        stats_o['episodes'] += 1
        stats_x['epsilon_x'].append(epsilon_x)
        stats_o['epsilon_o'].append(epsilon_o)
        
        # Mostra resultado a cada episódio
        print(f"Episódio {episode} - Recompensa X: {total_reward_x}, Recompensa O: {total_reward_o}, Jogadas: {moves_made}")
        print(f"Status: X ganhou {stats_x['x_wins']}, O ganhou {stats_o['o_wins']}, Empates {stats_x['draws']}")
        # def exibir_status(episode, total_reward_x, total_reward_o, moves_made, stats_x, stats_o):
        #exibir_status(episode, total_reward_x, total_reward_o, moves_made, stats_x, stats_o)
        
        # Salva checkpoints a cada episódio
        ckpt_dir = CONFIG["checkpoint_dir"]
        latest_x = os.path.join(ckpt_dir, "latest_checkpoint_x.pt")
        previous_x = os.path.join(ckpt_dir, "prev_checkpoint_x.pt")
        latest_o = os.path.join(ckpt_dir, "latest_checkpoint_o.pt")
        previous_o = os.path.join(ckpt_dir, "prev_checkpoint_o.pt")
        
        # Salva checkpoint para X
        fd_x, tmp_x = tempfile.mkstemp(dir=ckpt_dir, suffix="_x.tmp")
        os.close(fd_x)
        save_checkpoint(policy_net_x, target_net_x, optimizer_x, memory_x, episode, step_x, stats_x, tmp_x)
        
        if os.path.exists(latest_x):
            os.replace(latest_x, previous_x)
        os.replace(tmp_x, latest_x)
        
        # Salva checkpoint para O
        fd_o, tmp_o = tempfile.mkstemp(dir=ckpt_dir, suffix="_o.tmp")
        os.close(fd_o)
        save_checkpoint(policy_net_o, target_net_o, optimizer_o, memory_o, episode, step_o, stats_o, tmp_o)
        
        if os.path.exists(latest_o):
            os.replace(latest_o, previous_o)
        os.replace(tmp_o, latest_o)
        
        # Salva os modelos finais separadamente
        save_checkpoint(policy_net_x, target_net_x, optimizer_x, memory_x, episode, step_x, stats_x,
                      os.path.join(ckpt_dir, "model_x_final.pt"))
        save_checkpoint(policy_net_o, target_net_o, optimizer_o, memory_o, episode, step_o, stats_o,
                      os.path.join(ckpt_dir, "model_o_final.pt"))
        
        # Salva estatísticas finais
        with open(os.path.join(ckpt_dir, "stats_x_final.json"), "w") as f:
            json.dump(stats_x, f)
        with open(os.path.join(ckpt_dir, "stats_o_final.json"), "w") as f:
            json.dump(stats_o, f)

if __name__ == "__main__":
    train_self_play()