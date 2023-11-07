import pygame
import sys
import copy

# Inicializa o Pygame
pygame.init()

# Constantes
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
# rgb: red green blue
RED = (255, 0, 0)
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# Definindo o display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Jogo da Velha Avançado')
screen.fill(BG_COLOR)

# Tabuleiro principal e secundário
main_board = [[None]*3 for _ in range(3)]
sub_boards = [[[[None]*3 for _ in range(3)] for _ in range(3)] for _ in range(3)]

# Inicializa os sub-tabuleiros (todos vazios no começo)
for x in range(3):
    for y in range(3):
        main_board[x][y] = [[None]*3 for _ in range(3)]

# Função para desenhar as linhas do tabuleiro
def draw_lines():
    # Desenhando as linhas do tabuleiro principal
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(screen, LINE_COLOR, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    
    # Desenhando as linhas para os sub-tabuleiros
    for x in range(3):
        for y in range(3):
            for i in range(1, BOARD_ROWS):
                pygame.draw.line(screen, LINE_COLOR, (x * SQUARE_SIZE, y * SQUARE_SIZE + i * SQUARE_SIZE // 3), (x * SQUARE_SIZE + SQUARE_SIZE, y * SQUARE_SIZE + i * SQUARE_SIZE // 3), LINE_WIDTH // 4)
                pygame.draw.line(screen, LINE_COLOR, (x * SQUARE_SIZE + i * SQUARE_SIZE // 3, y * SQUARE_SIZE), (x * SQUARE_SIZE + i * SQUARE_SIZE // 3, y * SQUARE_SIZE + SQUARE_SIZE), LINE_WIDTH // 4)

# Função para desenhar os símbolos (X e O)
def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            for sub_row in range(BOARD_ROWS):
                for sub_col in range(BOARD_COLS):
                    if main_board[row][col][sub_row][sub_col] == 'O':
                       pygame.draw.circle(screen, CIRCLE_COLOR, (col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SQUARE_SIZE/6, row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SQUARE_SIZE/6), CIRCLE_RADIUS/2, CIRCLE_WIDTH)
                    elif main_board[row][col][sub_row][sub_col] == 'X':
                        start_desc = (col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SPACE, row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SPACE)
                        end_desc = (col * SQUARE_SIZE + (sub_col + 1) * SQUARE_SIZE/3 - SPACE, row * SQUARE_SIZE + (sub_row + 1) * SQUARE_SIZE/3 - SPACE)
                        pygame.draw.line(screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)
                        start_asc = (col * SQUARE_SIZE + (sub_col + 1) * SQUARE_SIZE/3 - SPACE, row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SPACE)
                        end_asc = (col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SPACE, row * SQUARE_SIZE + (sub_row + 1) * SQUARE_SIZE/3 - SPACE)
                        pygame.draw.line(screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)


# Função para marcar um movimento no tabuleiro
def mark_square(row, col, sub_row, sub_col, player):
    # Verifica se o quadrado está vazio
    if main_board[row][col][sub_row][sub_col] is None:
        main_board[row][col][sub_row][sub_col] = player
        return True
    return False


# Verifica se o tabuleiro está cheio
def is_board_full(board):
    for row in board:
        if None in row:
            return False
    return True

# Função principal para verificar o estado do jogo
# Checa se há um vencedor no sub-tabuleiro
def check_sub_winner(board, player):
    # Verifica linhas horizontais e verticais
    for col_row in range(3):
        if all([board[col_row][i] == player for i in range(3)]) or all([board[i][col_row] == player for i in range(3)]):
            return True
    # Verifica diagonais
    if board[0][0] == board[1][1] == board[2][2] == player or board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

# Conta os sub-tabuleiros ganhos por um jogador
def count_won_sub_boards(main_board, player):
    count = 0
    for row in range(3):
        for col in range(3):
            if check_sub_winner(main_board[row][col], player):
                count += 1
    return count

# Agora a função check_winner que verifica o estado do jogo principal e sub-tabuleiros
def check_winner():
    player_X_wins = count_won_sub_boards(main_board, 'X')
    player_O_wins = count_won_sub_boards(main_board, 'O')

    if player_X_wins > player_O_wins:
        return 'X'
    elif player_O_wins > player_X_wins:
        return 'O'
    else: # No caso de empate em número de sub-tabuleiros ganhos, ou se ainda não houver vencedores
        return None
    

# Função para jogar o próximo movimento no tabuleiro correto
def play_move(current_row, current_col, sub_row, sub_col, player):
    global main_board, current_player, game_over, last_move
    
    # Verifica se a jogada é válida
    if main_board[current_row][current_col][sub_row][sub_col] is None and not game_over:
        # Marca o quadrado para o jogador atual
        mark_square(current_row, current_col, sub_row, sub_col, player)
        
        # Verifica se a jogada atual ganhou o sub-tabuleiro
        if check_sub_winner(main_board[current_row][current_col], player):
            # Poderíamos aqui marcar este sub-tabuleiro como ganho no tabuleiro principal, se necessário
            pygame.draw.rect(screen, RED, (current_col * SQUARE_SIZE, current_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        # Verifica se este movimento levou a um tabuleiro já completo
        next_row, next_col = sub_row, sub_col
        if check_sub_winner(main_board[next_row][next_col], 'X') or check_sub_winner(main_board[next_row][next_col], 'O'):
            game_over = True  # Fim do jogo de acordo com a regra 4

        # Se o jogo não terminou, muda o jogador
        if not game_over:
            last_move = (sub_row, sub_col)
            current_player = 'O' if player == 'X' else 'X'
            
        # Se o jogo terminou, determina o vencedor
        if game_over:
            winner = check_winner()
            # Você pode aqui exibir o vencedor ou fazer o que for necessário para finalizar o jogo
    else:
        # A jogada não é válida
        print("Invalid move")
        # Você pode tratar uma jogada inválida como preferir


# minmax com poda alfa-beta
def minmax(board, depth, alpha, beta, maximizing_player, next_sub_board):
    global game_over
    if game_over or depth == 0:
        return evaluate(board), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in get_possible_moves(board, next_sub_board):
            board[move[0]][move[1]][move[2]][move[3]] = 'O'
            eval, _ = minmax(board, depth - 1, alpha, beta, False, (move[2], move[3]))
            print(f"Move: {move} Eval: {eval} MaxEval: {max_eval} Alpha: {alpha} Beta: {beta}")
            board[move[0]][move[1]][move[2]][move[3]] = None
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in get_possible_moves(board, next_sub_board):
            board[move[0]][move[1]][move[2]][move[3]] = 'X'
            eval, _ = minmax(board, depth - 1, alpha, beta, True, (move[2], move[3]))
            board[move[0]][move[1]][move[2]][move[3]] = None
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

    

# Função para obter as jogadas possíveis
def get_possible_moves(board, next_sub_board):
    possible_moves = []
    if next_sub_board is None:
        for row in range(3):
            for col in range(3):
                for sub_row in range(3):
                    for sub_col in range(3):
                        if board[row][col][sub_row][sub_col] is None:
                            possible_moves.append((row, col, sub_row, sub_col))
    else:
        row, col = next_sub_board
        for sub_row in range(3):
            for sub_col in range(3):
                if board[row][col][sub_row][sub_col] is None:
                    possible_moves.append((row, col, sub_row, sub_col))
    #print(f"Movimentos possiveis: {possible_moves}")
    return possible_moves



def evaluate(board):
    # Os pontos para a IA são calculados da seguinte forma:
    # 1. Cada sub-tabuleiro ganho vale 1000 pontos
    # 2. Cada sub-tabuleiro perdido vale -1000 pontos
    # 3. Cada sub-tabuleiro empatado vale 100 pontos
    # 4. Cada jogada no sub-tabuleiro que esta perto de ser ganho vale 10
    # 5. Cada jogada no sub-tabuleiro que esta perto de ser perdido vale -10
    # 6. Se o jogo acabou e a IA ganhou, vale 10000 pontos
    # 7. Se o jogo acabou e a IA perdeu, vale -10000 pontos
    # 8. Se o jogo acabou e empatou, vale 0 pontos


    # Pontuação inicial
    score = 0

    # Verifica se o jogo acabou
    global game_over
    if game_over:
        winner = check_winner()
        if winner == 'O':
            score += 10000
        elif winner == 'X':
            score -= 10000
    else:
        winner = check_winner()
        if winner == 'O':
            score += 10000
        elif winner == 'X':
            score -= 10000

    

    # Verifica os sub-tabuleiros ganhos e perdidos
    score += count_won_sub_boards(board, 'O') * 500
    score -= count_won_sub_boards(board, 'X') * 1000

    # Verifica os sub-tabuleiros empatados
    for row in range(3):
        for col in range(3):
            if is_board_full(board[row][col]):
                score += 100

    # Verifica as jogadas que podem levar a um sub-tabuleiro ganho ou perdido
    # for row in range(3):
    #     for col in range(3):
    #         for sub_row in range(3):
    #             for sub_col in range(3):
    #                 if board[row][col][sub_row][sub_col] is None:
    #                     board[row][col][sub_row][sub_col] = 'O'
    #                     if check_sub_winner(board[row][col], 'O'):
    #                         score += 10
    #                     board[row][col][sub_row][sub_col] = 'X'
    #                     if check_sub_winner(board[row][col], 'X'):
    #                         score -= 10
    #                     board[row][col][sub_row][sub_col] = None
    

    return score



current_player = 'X'  # X começa
next_sub_board = None  # Qualquer um pode ser jogado inicialmente
game_over = False

# Main loop do jogo
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX, mouseY = pygame.mouse.get_pos()
            
            clicked_row = int(mouseY // SQUARE_SIZE)
            clicked_col = int(mouseX // SQUARE_SIZE)
            
            sub_clicked_row = (mouseY % SQUARE_SIZE) // (SQUARE_SIZE / 3)
            sub_clicked_col = (mouseX % SQUARE_SIZE) // (SQUARE_SIZE / 3)
            
            # Se o próximo sub-tabuleiro é None, qualquer um pode ser jogado
            if next_sub_board is None or (clicked_row, clicked_col) == next_sub_board:
                if main_board[clicked_row][clicked_col][int(sub_clicked_row)][int(sub_clicked_col)] is None:

                    if current_player == 'X':
                        play_move(clicked_row, clicked_col, int(sub_clicked_row), int(sub_clicked_col), current_player)
                        # A próxima jogada deve ser no sub-tabuleiro correspondente à última jogada feita
                        next_sub_board = (int(sub_clicked_row), int(sub_clicked_col))

                    else:
                        # Chama a função minmax para obter a melhor jogada
                        print(next_sub_board)
                        score, best_move = minmax(main_board, 9, float('-inf'), float('inf'), True, next_sub_board)
                        print(score, best_move)
                        # Faz a jogada
                        play_move(best_move[0], best_move[1], best_move[2], best_move[3], current_player)
                        # A próxima jogada deve ser no sub-tabuleiro correspondente à última jogada feita
                        next_sub_board = (best_move[2], best_move[3])
            else:
                print(f"Você deve jogar no sub-tabuleiro {next_sub_board}")



    # Checar se o jogo acabou devido a um sub-tabuleiro cheio e início de um novo turno
    if game_over:
        winner = check_winner()
        # Código para exibir o vencedor ou finalizar o jogo
        if winner is None:
            print("Empate")
        print(f"O vencedor é {winner}")


    # Desenho do jogo
    draw_lines()
    draw_figures()

    if game_over:
        pygame.display.update()
        pygame.time.wait(10000)
        pygame.quit()

    # Atualiza o display
    pygame.display.update()

