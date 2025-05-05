# ai.py - Módulo de Inteligência Artificial para o Jogo da Velha Avançado

MAX_RECURSION_DEPTH = 7000

def minmax(board, depth, alpha, beta, maximizing_player, next_sub_board, _lvl=0):
    """
    Implementação do algoritmo minimax com poda alfa-beta.
    
    Args:
        board: O tabuleiro de jogo
        depth: Profundidade máxima de busca
        alpha: Valor alfa para poda
        beta: Valor beta para poda
        maximizing_player: True se é o jogador que maximiza (IA jogando como 'O'), 
                          False se é o jogador que minimiza (IA jogando como 'X')
        next_sub_board: Tupla (row, col) indicando o próximo sub-tabuleiro a jogar
        
    Returns:
        Tupla (score, move) com a pontuação e o melhor movimento encontrado
    """
    if _lvl >= MAX_RECURSION_DEPTH:
        return evaluate(board, next_sub_board), None
    # Verificar condições de término
    if is_game_over(board) or depth == 0:
        return evaluate(board), None

    if maximizing_player:  # IA jogando como 'O'
        max_eval = float('-inf')
        best_move = None
        for move in get_possible_moves(board, next_sub_board):
            # Simula o movimento
            board[move[0]][move[1]][move[2]][move[3]] = 'O'
            eval, _ = minmax(board, depth - 1, alpha, beta, False, (move[2], move[3]), _lvl=_lvl + 1)
            #print(board)
            # Desfaz o movimento
            board[move[0]][move[1]][move[2]][move[3]] = None
            
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:  # IA jogando como 'X' ou simulando jogador humano
        min_eval = float('inf')
        best_move = None
        for move in get_possible_moves(board, next_sub_board):
            # Simula o movimento
            board[move[0]][move[1]][move[2]][move[3]] = 'X'
            eval, _ = minmax(board, depth - 1, alpha, beta, True, (move[2], move[3]), _lvl=_lvl + 1)
            # Desfaz o movimento
            board[move[0]][move[1]][move[2]][move[3]] = None
            
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def is_game_over(board):
    """Verifica se o jogo terminou."""
    # Conta os sub-tabuleiros ganhos por cada jogador
    player_X_wins = count_won_sub_boards(board, 'X')
    player_O_wins = count_won_sub_boards(board, 'O')
    
    # Se todos os sub-tabuleiros estão cheios ou alguém ganhou a maioria
    filled_boards = 0
    for row in range(3):
        for col in range(3):
            if is_sub_board_full(board[row][col]) or sub_winner(board[row][col], 'X') or sub_winner(board[row][col], 'O'):
                filled_boards += 1
    
    if filled_boards == 9:
        return True
    
    # Se um jogador ganhou 5 ou mais sub-tabuleiros (maioria)
    if player_X_wins >= 5 or player_O_wins >= 5:
        return True
    
    return False

def count_won_sub_boards(board, player):
    """Conta quantos sub-tabuleiros foram ganhos por um jogador."""
    count = 0
    for row in range(3):
        for col in range(3):
            if sub_winner(board[row][col], player):
                count += 1
    return count

def sub_winner(sub_board, player):
    """Verifica se um jogador ganhou um sub-tabuleiro."""
    # Verifica linhas horizontais e verticais
    for col_row in range(3):
        if all([sub_board[col_row][i] == player for i in range(3)]) or all([sub_board[i][col_row] == player for i in range(3)]):
            return True
    # Verifica diagonais
    if sub_board[0][0] == sub_board[1][1] == sub_board[2][2] == player or sub_board[0][2] == sub_board[1][1] == sub_board[2][0] == player:
        return True
    return False

def is_sub_board_full(sub_board):
    """Verifica se um sub-tabuleiro está completamente preenchido."""
    for row in sub_board:
        if None in row:
            return False
    return True

def evaluate(board):
    """
    Função de avaliação heurística do tabuleiro.
    Retorna um valor numérico representando quão bom é o estado atual para a IA.
    Valores positivos favorecem a IA (O), valores negativos favorecem o jogador humano (X).
    """
    score = 0

    # Avaliar vencedor final (maior prioridade)
    player_X_wins = count_won_sub_boards(board, 'X')
    player_O_wins = count_won_sub_boards(board, 'O')
    
    if player_O_wins > player_X_wins:
        score += 10000
    elif player_X_wins > player_O_wins:
        score -= 10000

    # Avaliar cada sub-tabuleiro
    for row in range(3):
        for col in range(3):
            sub_board = board[row][col]
            
            # Sub-tabuleiro ganho vale mais
            if sub_winner(sub_board, 'O'):
                score += 1000
            elif sub_winner(sub_board, 'X'):
                score -= 2000
            elif is_sub_board_full(sub_board):
                score += 100  # Empate é ligeiramente positivo para IA
            else:
                # Avaliar situações de quase-vitória
                score += evaluate_sub_board(sub_board)
    
    return score

def evaluate_sub_board(sub_board):
    """Avalia um sub-tabuleiro específico."""
    score = 0
    
    # Avaliar linhas, colunas e diagonais
    # Linhas
    for row in range(3):
        o_count = x_count = 0
        for col in range(3):
            if sub_board[row][col] == 'O':
                o_count += 1
            elif sub_board[row][col] == 'X':
                x_count += 1
                
        score += evaluate_line(o_count, x_count)
    
    # Colunas
    for col in range(3):
        o_count = x_count = 0
        for row in range(3):
            if sub_board[row][col] == 'O':
                o_count += 1
            elif sub_board[row][col] == 'X':
                x_count += 1
                
        score += evaluate_line(o_count, x_count)
    
    # Diagonal principal
    o_count = x_count = 0
    for i in range(3):
        if sub_board[i][i] == 'O':
            o_count += 1
        elif sub_board[i][i] == 'X':
            x_count += 1
    score += evaluate_line(o_count, x_count)
    
    # Diagonal secundária
    o_count = x_count = 0
    for i in range(3):
        if sub_board[i][2-i] == 'O':
            o_count += 1
        elif sub_board[i][2-i] == 'X':
            x_count += 1
    score += evaluate_line(o_count, x_count)
    
    return score

def evaluate_line(o_count, x_count):
    """Avalia uma linha, coluna ou diagonal."""
    if o_count == 0 and x_count == 0:
        return 0  # Linha vazia
    
    if o_count > 0 and x_count > 0:
        return 0  # Bloqueada, ninguém pode ganhar
    
    if o_count == 2 and x_count == 0:
        return 30  # IA pode ganhar no próximo movimento
    if o_count == 1 and x_count == 0:
        return 10  # IA tem potencial
        
    if x_count == 2 and o_count == 0:
        return -30  # Jogador humano pode ganhar no próximo movimento (prioridade de bloqueio)
    if x_count == 1 and o_count == 0:
        return -10  # Jogador humano tem potencial
    
    return 0

def get_possible_moves(board, next_sub_board):
    """
    Retorna uma lista de todas as jogadas possíveis no formato (row, col, sub_row, sub_col).
    
    Args:
        board: O tabuleiro de jogo
        next_sub_board: Tupla (row, col) indicando o próximo sub-tabuleiro a jogar ou None
    """
    possible_moves = []
    
    # Se não há restrição de sub-tabuleiro ou o sub-tabuleiro indicado já está completo
    if next_sub_board is None:
        for row in range(3):
            for col in range(3):
                # Só considera sub-tabuleiros que não estão completos ou ganhos
                if not (sub_winner(board[row][col], 'X') or 
                        sub_winner(board[row][col], 'O') or 
                        is_sub_board_full(board[row][col])):
                    for sub_row in range(3):
                        for sub_col in range(3):
                            if board[row][col][sub_row][sub_col] is None:
                                possible_moves.append((row, col, sub_row, sub_col))
    else:
        row, col = next_sub_board
        # Verifica se o sub-tabuleiro ainda está disponível para jogadas
        if not (sub_winner(board[row][col], 'X') or 
                sub_winner(board[row][col], 'O') or 
                is_sub_board_full(board[row][col])):
            for sub_row in range(3):
                for sub_col in range(3):
                    if board[row][col][sub_row][sub_col] is None:
                        possible_moves.append((row, col, sub_row, sub_col))
        else:
            # Se o sub-tabuleiro já está completo, considera todos os outros
            for row in range(3):
                for col in range(3):
                    if not (sub_winner(board[row][col], 'X') or 
                            sub_winner(board[row][col], 'O') or 
                            is_sub_board_full(board[row][col])):
                        for sub_row in range(3):
                            for sub_col in range(3):
                                if board[row][col][sub_row][sub_col] is None:
                                    possible_moves.append((row, col, sub_row, sub_col))
    
    return possible_moves