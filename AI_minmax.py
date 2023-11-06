import copy

def generate_valid_moves(main_board, last_move):
    valid_moves = []
    # Caso seja o primeiro movimento, todos os movimentos são válidos.
    if last_move is None:
        for i, row in enumerate(main_board):
            for j, sub_board in enumerate(row):
                for k in range(3):
                    for l in range(3):
                        if sub_board[k][l] == 0:
                            valid_moves.append(((i, j), (k, l)))
    else:
        # Identifique o sub-tabuleiro onde o próximo movimento deve ser feito.
        last_sub_board_row, last_sub_board_col = last_move
        target_sub_board = main_board[last_sub_board_row][last_sub_board_col]

        # Se o sub-tabuleiro alvo já está resolvido ou completamente cheio, o jogador pode escolher qualquer sub-tabuleiro não resolvido.
        if check_winner(target_sub_board) is not None or not any(0 in row for row in target_sub_board):
            for i, row in enumerate(main_board):
                for j, sub_board in enumerate(row):
                    if check_winner(sub_board) is None:
                        for k in range(3):
                            for l in range(3):
                                if sub_board[k][l] == 0:
                                    valid_moves.append(((i, j), (k, l)))
        else:
            # Gere movimentos válidos apenas dentro do sub-tabuleiro alvo.
            for k in range(3):
                for l in range(3):
                    if target_sub_board[k][l] == 0:
                        valid_moves.append(((last_sub_board_row, last_sub_board_col), (k, l)))
    
    return valid_moves


# Modifique minimax_alpha_beta para usar a função generate_valid_moves
def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, last_move):
    game_over_result, winner = game_over(board)
    if depth == 0 or game_over_result:
        return value_of_board(board, winner)

    if maximizing_player:
        max_eval = float('-inf')
        for each_possible_move in generate_valid_moves(board, last_move):
            evaluation = minimax_alpha_beta(new_board_from_move(board, each_possible_move[0], each_possible_move[1], 'X'), depth - 1, alpha, beta, False, each_possible_move)
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break  # Poda alpha
        return max_eval
    else:
        min_eval = float('inf')
        for each_possible_move in generate_valid_moves(board, last_move):
            evaluation = minimax_alpha_beta(new_board_from_move(board, each_possible_move[0], each_possible_move[1], 'O'), depth - 1, alpha, beta, True, each_possible_move)
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break  # Poda beta
        return min_eval

# Função auxiliar para determinar se o jogo terminou
def game_over(main_board):
    """
    Verifica se o jogo acabou e retorna o estado do jogo junto com o vencedor.
    
    :param main_board: O tabuleiro principal contendo todos os sub-tabuleiros
    :return: Uma tupla (game_over, winner), onde `game_over` é um booleano indicando se o jogo acabou,
             e `winner` é 'X', 'O', ou 'None' indicando o vencedor.
    """
    # Inicializa a contagem de vitórias para cada jogador
    wins = {'X': 0, 'O': 0}

    # Verifica cada sub-tabuleiro para contabilizar as vitórias
    for row in main_board:
        for sub_board in row:
            winner = check_winner(sub_board)  # Supõe-se que você tenha uma função que verifica o vencedor em um sub-tabuleiro
            if winner == 'X':
                wins['X'] += 1
            elif winner == 'O':
                wins['O'] += 1

    # Se todos os sub-tabuleiros forem ganhos ou empates, o jogo acabou
    total_sub_boards = 9
    if sum(wins.values()) == total_sub_boards:
        game_over = True
        winner = 'X' if wins['X'] > wins['O'] else 'O' if wins['O'] > wins['X'] else 'None'
        return game_over, winner
    
    # Se ainda houver movimentos possíveis, o jogo não acabou
    return False, 'None'

# Função que verifica se há um vencedor em um sub-tabuleiro
def check_winner(sub_board):
    # Verifica as linhas
    for row in sub_board:
        if row[0] == row[1] == row[2] != 0:
            return row[0]

    # Verifica as colunas
    for col in range(3):
        if sub_board[0][col] == sub_board[1][col] == sub_board[2][col] != 0:
            return sub_board[0][col]

    # Verifica as diagonais
    if sub_board[0][0] == sub_board[1][1] == sub_board[2][2] != 0:
        return sub_board[0][0]
    if sub_board[0][2] == sub_board[1][1] == sub_board[2][0] != 0:
        return sub_board[0][2]

    # Não há vencedores ainda
    return None


# Função auxiliar para avaliar o valor do tabuleiro
def value_of_board(main_board):
    # Inicializa a pontuação
    score = 0

    # Pontuação para cada jogador
    player_scores = {'X': 0, 'O': 0}

    # Atribuir valores para cada vitória em sub-tabuleiros
    value_for_win = 1  # Este valor pode ser ajustado se outras heurísticas forem aplicadas

    # Verificar cada sub-tabuleiro
    for row in main_board:
        for sub_board in row:
            winner = check_winner(sub_board)
            if winner:
                player_scores[winner] += value_for_win

    # A avaliação é a diferença de pontuação
    score = player_scores['X'] - player_scores['O']

    return score


# Função auxiliar para obter um novo tabuleiro com a jogada aplicada
def new_board_from_move(main_board, board_coords, move_coords, player):
    # Crie uma cópia profunda do tabuleiro para não alterar o original
    new_board = copy.deepcopy(main_board)
    
    # Extrai as coordenadas do sub-tabuleiro e do movimento
    sub_board_row, sub_board_col = board_coords
    move_row, move_col = move_coords
    
    # Aplica o movimento no novo tabuleiro
    new_board[sub_board_row][sub_board_col][move_row][move_col] = player
    
    return new_board



# Para iniciar o algoritmo, você chamaria assim:
#best_move = minimax_alpha_beta(main_board, depth, float('-inf'), float('inf'), True)
