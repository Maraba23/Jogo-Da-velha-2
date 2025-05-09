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

# Configurações de treinamento
CONFIG = {
    "model_symbol": "O",          # Símbolo que o modelo vai usar para treinar (X ou O)
    "opponent_difficulty": 4,     # Dificuldade do oponente (1-8)
    "learning_rate": 0.0001,      # Taxa de aprendizado
    "gamma": 0.95,                # Fator de desconto para recompensas futuras
    "epsilon_start": 1.0,         # Exploração inicial (1 = 100% aleatório)
    "epsilon_end": 0.05,           # Exploração final (0.1 = 10% aleatório)
    "epsilon_decay": 40000,      # Quantas etapas para reduzir epsilon
    "batch_size": 64,            # Tamanho do lote para treinamento
    "memory_size": 20000,         # Tamanho da memória de experiência
    "target_update": 500,        # Frequência para atualizar a rede alvo
    "save_interval": 5000,        # Frequência para salvar o modelo
    "checkpoint_dir": "model_checkpoints",  # Diretório para salvar checkpoints
    "episodes": 10_000            # Número de jogos a serem treinados
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
        
        # Troca o jogador
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return self.get_state_tensor(), reward, self.game_over

    def opponent_move(self, difficulty):
        """Faz uma jogada utilizando o algoritmo minimax do oponente"""
        # Se o jogo já terminou, não faz nada
        if self.game_over:
            return self.get_state_tensor(), 0, True
            
        # Determina se o oponente está maximizando ou minimizando
        is_maximizing = self.current_player == 'O'
        
        try:
            # Chama o algoritmo minimax
            _, best_move = ai.minmax(
                self.main_board, 
                difficulty, 
                float('-inf'), 
                float('inf'), 
                is_maximizing, 
                self.next_sub_board
            )
            
            if best_move:
                # Faz a jogada do oponente
                return self.make_move(best_move)
            else:
                # Se não houver jogadas possíveis, o jogo termina
                self.game_over = True
                return self.get_state_tensor(), 0, True
                
        except Exception as e:
            print(f"Erro no movimento do oponente: {e}")
            # Em caso de erro, faz uma jogada aleatória
            valid_moves = self.get_valid_moves()
            if valid_moves:
                return self.make_move(random.choice(valid_moves))
            else:
                self.game_over = True
                return self.get_state_tensor(), 0, True

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
# class DQN(nn.Module):
#     def __init__(self):
#         super(DQN, self).__init__()
        
#         # Camadas convolucionais
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
        
#         # Camadas densas
#         self.fc1 = nn.Linear(256 * 9 * 9, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 81)  # 81 possíveis saídas (9x9 grid)
        
#     def forward(self, x):
#         # Camadas convolucionais
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
        
#         # Achatando para alimentar nas camadas densas
#         x = x.view(x.size(0), -1)
        
#         # Camadas densas
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x
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
    state_batch = torch.stack([s.to(device, non_blocking=True)     for s in batch[0]])
    action_batch = batch[1]
    next_state_batch = torch.stack([ns.to(device, non_blocking=True) for ns in batch[2]])
    reward_batch = torch.tensor(batch[3], device=device, dtype=torch.float32)
    done_batch   = torch.tensor(batch[4], device=device, dtype=torch.bool)
    
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
        return 0, 0, {'wins': 0, 'losses': 0, 'draws': 0, 'episodes': 0, 'epsilon': []}

# Função principal de treinamento
def train():
    """
    Função principal para treinar o modelo
    """
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Inicialização do ambiente e da memória
    env = GameState()
    memory = ReplayMemory(CONFIG["memory_size"])
    
    # Inicialização das redes
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Modo de avaliação (não treinamento)
    
    # Inicialização do otimizador
    optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG["learning_rate"])
    
    # Carregamento do checkpoint (se existir)
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "latest_checkpoint.pt")
    start_episode, step, stats = load_checkpoint(checkpoint_path, policy_net, target_net, optimizer, memory, device)
    
    # Se não houver estatísticas carregadas, inicializa-as
    if not stats:
        stats = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'episodes': 0,
            'epsilon': []
        }
    
    # Loop principal de treinamento
    model_symbol = CONFIG["model_symbol"]
    opponent_symbol = "O" if model_symbol == "X" else "X"
    
    for episode in range(start_episode, CONFIG["episodes"]):
        # Reinicia o ambiente
        state = env.reset()
        total_reward = 0
        game_over = False
        moves_made = 0
        
        # Se o oponente começa (modelo é O e X sempre começa), faz a primeira jogada
        if model_symbol == "O" and env.current_player == "X":
            _, reward, game_over = env.opponent_move(CONFIG["opponent_difficulty"])
            total_reward += reward
        
        # Loop do jogo (até terminar)
        while not game_over:
            # Calcula epsilon para este passo (exploração vs aproveitamento)
            epsilon = CONFIG["epsilon_end"] + (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]) * \
                      math.exp(-1. * step / CONFIG["epsilon_decay"])
                      
            # Só faz jogada se for a vez do modelo
            if env.current_player == model_symbol:
                # Obtém as jogadas válidas
                valid_moves = env.get_valid_moves()

                # Se não houver jogadas possíveis, terminar episódio de forma limpa
                if not valid_moves:
                    game_over = True
                    break

                action = select_action(state, policy_net, epsilon, valid_moves, device)
                # Segurança extra – select_action NUNCA deveria devolver None aqui
                if action is None:
                    game_over = True
                    break

                next_state, reward, done = env.make_move(action)
                
                # Armazena a transição na memória
                memory.push(state, action, next_state, reward, done)
                
                # Atualiza o estado atual
                state = next_state
                total_reward += reward
                game_over = done
                moves_made += 1
                
                # Realiza uma etapa de otimização
                loss = optimize_model(policy_net, target_net, optimizer, memory, device, 
                                     CONFIG["batch_size"], CONFIG["gamma"])
                
                # Atualiza a rede alvo periodicamente
                if step % CONFIG["target_update"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
                # Incrementa o contador de passos
                step += 1
                
                # Salva o modelo periodicamente
                # if step % CONFIG["save_interval"] == 0:
                #     save_checkpoint(policy_net, target_net, optimizer, memory, episode, step, stats,
                #                    os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_step_{step}.pt"))
                #     save_checkpoint(policy_net, target_net, optimizer, memory, episode, step, stats,
                #                    checkpoint_path)
                    
                #     # Exibe estatísticas
                #     win_rate = stats['wins'] / max(1, stats['episodes']) * 100
                #     loss_rate = stats['losses'] / max(1, stats['episodes']) * 100
                #     draw_rate = stats['draws'] / max(1, stats['episodes']) * 100
                    
                #     print(f"Passo {step}, Epsilon {epsilon:.4f}, Vitórias {win_rate:.1f}%, Derrotas {loss_rate:.1f}%, Empates {draw_rate:.1f}%")
                    
                #     # Salva estatísticas em um arquivo JSON
                #     with open(os.path.join(CONFIG["checkpoint_dir"], "stats.json"), 'w') as f:
                #         json.dump(stats, f)
            
            # Se for a vez do oponente, faz a jogada usando minimax
            else:
                next_state, reward, done = env.opponent_move(CONFIG["opponent_difficulty"])
                state = next_state
                total_reward -= reward  # Inverte a recompensa (vitória do oponente é ruim para o modelo)
                game_over = done
            
            # Se o jogo terminou, atualiza as estatísticas
            if game_over:
                # Determina quem ganhou
                x_wins = sum(row.count('X') for row in env.won_sub_boards)
                o_wins = sum(row.count('O') for row in env.won_sub_boards)
                
                if x_wins > o_wins and model_symbol == "X":
                    stats['wins'] += 1
                elif o_wins > x_wins and model_symbol == "O":
                    stats['wins'] += 1
                elif x_wins == o_wins:
                    stats['draws'] += 1
                else:
                    stats['losses'] += 1
                
                stats['episodes'] += 1
                stats['epsilon'].append(epsilon)
                
                # Mostra resultado a cada 100 episódios
                #if episode % 100 == 0:
                print(f"Episódio {episode} - Recompensa total: {total_reward}, Jogadas: {moves_made}")
                print(f"Status: Vitórias {stats['wins']}, Derrotas {stats['losses']}, Empates {stats['draws']}")
                # --- dentro do loop principal, depois de atualizar stats e imprimir resultado ---
                # grava sempre no fim de cada episódio

                ckpt_dir  = CONFIG["checkpoint_dir"]
                latest    = os.path.join(ckpt_dir, "latest_checkpoint.pt")
                previous  = os.path.join(ckpt_dir, "prev_checkpoint.pt")

                # 1) grava em tmp
                fd, tmp = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
                os.close(fd)
                save_checkpoint(policy_net, target_net, optimizer,
                                memory, episode, step, stats, tmp)

                # 2) move antigo latest -> prev (atômico)
                if os.path.exists(latest):
                    os.replace(latest, previous)            # sobrescreve prev se existir

                # 3) move tmp -> latest (atômico)
                os.replace(tmp, latest)

                # opcional: snapshot só‑pesos / stats legíveis por fora
                save_checkpoint(policy_net, target_net, optimizer,
                                memory, episode, step, stats,
                                os.path.join(ckpt_dir, "model_final.pt"))
                with open(os.path.join(ckpt_dir, "stats_final.json"), "w") as f:
                    json.dump(stats, f)


    # # Final do treinamento
    # print("Treinamento concluído!")
    
    # # Salva o modelo final
    # save_checkpoint(policy_net, target_net, optimizer, memory, CONFIG["episodes"], step, stats,
    #                os.path.join(CONFIG["checkpoint_dir"], "model_final.pt"))
    # save_checkpoint(policy_net, target_net, optimizer, memory, CONFIG["episodes"], step, stats,
    #                checkpoint_path)
    
    # # Salva estatísticas finais
    # with open(os.path.join(CONFIG["checkpoint_dir"], "stats_final.json"), 'w') as f:
    #     json.dump(stats, f)

if __name__ == "__main__":
    train()