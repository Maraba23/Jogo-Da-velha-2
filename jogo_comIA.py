import pygame
import sys
import random
import math
import ai  # Importa o módulo de IA que criamos

# Inicializa o Pygame
pygame.init()
pygame.font.init()  # Inicializa o sistema de fontes

# Constantes
WIDTH, HEIGHT = 600, 650  # Altura extra para o painel de status
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Cores com esquema mais agradável
BG_COLOR = (240, 240, 250)  # Azul muito claro
LINE_COLOR = (70, 80, 100)   # Azul escuro
CIRCLE_COLOR = (0, 120, 215)  # Azul médio
CROSS_COLOR = (200, 60, 50)   # Vermelho
HIGHLIGHT_COLOR = (255, 215, 0, 80)  # Dourado com transparência
TEXT_COLOR = (40, 40, 40)     # Quase preto
STATUS_BG_COLOR = (230, 230, 240)  # Ligeiramente mais escuro que o fundo
WON_COLORS = {
    'X': (255, 100, 100, 60),  # Vermelho transparente
    'O': (100, 170, 255, 60)   # Azul transparente
}

# Configuração das fontes
title_font = pygame.font.SysFont('Arial Black', 36)
font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 16)

# Definindo o display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Jogo da Velha Avançado')
screen.fill(BG_COLOR)

# Tabuleiro principal
main_board = [[[[None]*3 for _ in range(3)] for _ in range(3)] for _ in range(3)]

# Rastreia quais sub-tabuleiros foram ganhos
won_sub_boards = [[None for _ in range(3)] for _ in range(3)]

# Efeitos visuais
particles = []

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-2, 0)
        self.radius = random.uniform(2, 5)
        self.life = 30  # Duração em frames
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravidade
        self.life -= 1
        self.radius = max(0, self.radius - 0.1)
        
    def draw(self, screen):
        alpha = int(255 * (self.life / 30))
        color_with_alpha = (self.color[0], self.color[1], self.color[2], alpha)
        surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color_with_alpha, (self.radius, self.radius), self.radius)
        screen.blit(surf, (self.x - self.radius, self.y - self.radius))

def create_particles(x, y, color, count=20):
    for _ in range(count):
        particles.append(Particle(x, y, color))

def update_particles():
    for particle in particles[:]:
        particle.update()
        if particle.life <= 0:
            particles.remove(particle)

def draw_particles():
    for particle in particles:
        particle.draw(screen)

# Função para desenhar o fundo do tabuleiro com textura
def draw_board_background():
    # Desenha o fundo com gradiente sutil
    for y in range(HEIGHT):
        # Gradiente muito sutil de cima para baixo
        color_value = 240 - int(y / HEIGHT * 20)  # Varia de 240 a 220
        pygame.draw.line(screen, (color_value, color_value, color_value+10), (0, y), (WIDTH, y))
    
    # Desenha o painel de status
    pygame.draw.rect(screen, STATUS_BG_COLOR, (0, HEIGHT-50, WIDTH, 50))
    pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT-50), (WIDTH, HEIGHT-50), 2)

# Função para desenhar as linhas do tabuleiro
def draw_lines():
    # Desenhando as linhas do tabuleiro principal
    for i in range(1, BOARD_ROWS):
        # Linhas horizontais principais
        pygame.draw.line(screen, LINE_COLOR, 
                       (0, i * SQUARE_SIZE), 
                       (WIDTH, i * SQUARE_SIZE), 
                       LINE_WIDTH)
        # Linhas verticais principais
        pygame.draw.line(screen, LINE_COLOR, 
                       (i * SQUARE_SIZE, 0), 
                       (i * SQUARE_SIZE, HEIGHT-50), 
                       LINE_WIDTH)
    
    # Desenhando as linhas para os sub-tabuleiros com linhas mais finas
    for x in range(3):
        for y in range(3):
            # Se o sub-tabuleiro foi ganho, desenha um fundo colorido
            if won_sub_boards[y][x] is not None:
                color = WON_COLORS[won_sub_boards[y][x]]
                surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                surf.fill(color)
                screen.blit(surf, (x * SQUARE_SIZE, y * SQUARE_SIZE))
            
            for i in range(1, BOARD_ROWS):
                # Linhas horizontais dos sub-tabuleiros
                pygame.draw.line(screen, (100, 100, 120, 150), 
                               (x * SQUARE_SIZE, y * SQUARE_SIZE + i * SQUARE_SIZE // 3), 
                               (x * SQUARE_SIZE + SQUARE_SIZE, y * SQUARE_SIZE + i * SQUARE_SIZE // 3), 
                               max(1, LINE_WIDTH // 4))
                # Linhas verticais dos sub-tabuleiros
                pygame.draw.line(screen, (100, 100, 120, 150), 
                               (x * SQUARE_SIZE + i * SQUARE_SIZE // 3, y * SQUARE_SIZE), 
                               (x * SQUARE_SIZE + i * SQUARE_SIZE // 3, y * SQUARE_SIZE + SQUARE_SIZE), 
                               max(1, LINE_WIDTH // 4))

# Função para desenhar os símbolos (X e O) com animações aprimoradas
def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            for sub_row in range(BOARD_ROWS):
                for sub_col in range(BOARD_COLS):
                    if main_board[row][col][sub_row][sub_col] == 'O':
                        center = (col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SQUARE_SIZE/6, 
                                 row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SQUARE_SIZE/6)
                        # Desenha um círculo com efeito de brilho
                        pygame.draw.circle(screen, CIRCLE_COLOR, center, CIRCLE_RADIUS/2, 
                                         max(2, CIRCLE_WIDTH//2))
                        
                        # Adiciona um pequeno brilho interno
                        highlight_pos = (center[0] - CIRCLE_RADIUS/8, center[1] - CIRCLE_RADIUS/8)
                        highlight_radius = CIRCLE_RADIUS/8
                        surf = pygame.Surface((highlight_radius*2, highlight_radius*2), pygame.SRCALPHA)
                        pygame.draw.circle(surf, (255, 255, 255, 120), 
                                        (highlight_radius, highlight_radius), highlight_radius)
                        screen.blit(surf, (highlight_pos[0] - highlight_radius, 
                                        highlight_pos[1] - highlight_radius))
                        
                    elif main_board[row][col][sub_row][sub_col] == 'X':
                        # Posições para as linhas do X
                        start_desc = (col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SPACE/2, 
                                     row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SPACE/2)
                        end_desc = (col * SQUARE_SIZE + (sub_col + 1) * SQUARE_SIZE/3 - SPACE/2, 
                                   row * SQUARE_SIZE + (sub_row + 1) * SQUARE_SIZE/3 - SPACE/2)
                        start_asc = (col * SQUARE_SIZE + (sub_col + 1) * SQUARE_SIZE/3 - SPACE/2, 
                                    row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SPACE/2)
                        end_asc = (col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SPACE/2, 
                                  row * SQUARE_SIZE + (sub_row + 1) * SQUARE_SIZE/3 - SPACE/2)
                        
                        # Desenha sombras para efeito 3D
                        shadow_offset = 2
                        shadow_color = (20, 20, 20, 100)
                        
                        # Sombra da primeira linha
                        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                        pygame.draw.line(surf, shadow_color, 
                                      (start_desc[0]+shadow_offset, start_desc[1]+shadow_offset), 
                                      (end_desc[0]+shadow_offset, end_desc[1]+shadow_offset), 
                                      max(2, CROSS_WIDTH//2))
                        screen.blit(surf, (0, 0))
                        
                        # Primeira linha
                        pygame.draw.line(screen, CROSS_COLOR, start_desc, end_desc, 
                                       max(2, CROSS_WIDTH//2))
                        
                        # Sombra da segunda linha
                        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                        pygame.draw.line(surf, shadow_color, 
                                      (start_asc[0]+shadow_offset, start_asc[1]+shadow_offset), 
                                      (end_asc[0]+shadow_offset, end_asc[1]+shadow_offset), 
                                      max(2, CROSS_WIDTH//2))
                        screen.blit(surf, (0, 0))
                        
                        # Segunda linha
                        pygame.draw.line(screen, CROSS_COLOR, start_asc, end_asc, 
                                       max(2, CROSS_WIDTH//2))

# Função para destacar o próximo sub-tabuleiro a ser jogado
def highlight_next_board(next_sub_board):
    if next_sub_board is not None:
        row, col = next_sub_board
        if won_sub_boards[row][col] is None:  # Só destaca se o sub-tabuleiro não estiver ganho
            # Cria uma superfície com transparência para o destaque
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Função para exibir mensagens na tela
def display_message(message, y_pos=HEIGHT/2, font_size='normal', color=TEXT_COLOR):
    if font_size == 'large':
        font_to_use = title_font
    elif font_size == 'small':
        font_to_use = small_font
    else:
        font_to_use = font
        
    message_surface = font_to_use.render(message, True, color)
    message_rect = message_surface.get_rect(center=(WIDTH/2, y_pos))
    
    # Fundo com transparência para a mensagem
    padding = 10
    bg_surface = pygame.Surface(
        (message_rect.width + padding*2, message_rect.height + padding), 
        pygame.SRCALPHA
    )
    bg_surface.fill((255, 255, 255, 200))
    screen.blit(bg_surface, (message_rect.x - padding, message_rect.y - padding//2))
    
    screen.blit(message_surface, message_rect)

# Função para exibir o status do jogo
def display_game_status(current_player, next_sub_board, x_wins, o_wins):
    # Área de status na parte inferior
    status_rect = pygame.Rect(0, HEIGHT-50, WIDTH, 50)
    pygame.draw.rect(screen, STATUS_BG_COLOR, status_rect)
    pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT-50), (WIDTH, HEIGHT-50), 2)
    
    # Quem é o jogador atual
    if current_player == human_player:
        player_text = f"Deep: {difficulty} | {current_player} (Você)"
    else:
        player_text = f"Deep: {difficulty} | {current_player} (IA)"
    
    player_surface = font.render(player_text, True, TEXT_COLOR)
    screen.blit(player_surface, (10, HEIGHT-40))
    
    # Pontuação
    score_text = f"X: {x_wins}  |  O: {o_wins}"
    score_surface = font.render(score_text, True, TEXT_COLOR)
    score_rect = score_surface.get_rect(center=(WIDTH/2, HEIGHT-25))
    screen.blit(score_surface, score_rect)
    
    # Próximo tabuleiro
    if next_sub_board is not None:
        next_board_text = f"Próximo: ({next_sub_board[0]}, {next_sub_board[1]})"
    else:
        next_board_text = "Próximo: qualquer um"
    next_board_surface = small_font.render(next_board_text, True, TEXT_COLOR)
    screen.blit(next_board_surface, (WIDTH - next_board_surface.get_width() - 10, HEIGHT-40))

# Função para marcar um movimento no tabuleiro
def mark_square(row, col, sub_row, sub_col, player):
    # Verifica se o quadrado está vazio
    if main_board[row][col][sub_row][sub_col] is None:
        main_board[row][col][sub_row][sub_col] = player
        # Cria efeitos de partículas na posição marcada
        particle_x = col * SQUARE_SIZE + sub_col * SQUARE_SIZE/3 + SQUARE_SIZE/6
        particle_y = row * SQUARE_SIZE + sub_row * SQUARE_SIZE/3 + SQUARE_SIZE/6
        if player == 'X':
            create_particles(particle_x, particle_y, CROSS_COLOR)
        else:
            create_particles(particle_x, particle_y, CIRCLE_COLOR)
        return True
    return False

# Função para jogar um movimento
def play_move(current_row, current_col, sub_row, sub_col, player):
    global main_board, won_sub_boards, current_player, game_over, next_sub_board
    
    # Verifica se a jogada é válida
    if main_board[current_row][current_col][sub_row][sub_col] is None and not game_over:
        # Marca o quadrado para o jogador atual
        mark_square(current_row, current_col, sub_row, sub_col, player)
        
        # Verifica se a jogada atual ganhou o sub-tabuleiro
        if ai.sub_winner(main_board[current_row][current_col], player):
            won_sub_boards[current_row][current_col] = player
            # Cria efeito de partículas extras para celebrar
            particle_x = current_col * SQUARE_SIZE + SQUARE_SIZE/2
            particle_y = current_row * SQUARE_SIZE + SQUARE_SIZE/2
            color = CROSS_COLOR if player == 'X' else CIRCLE_COLOR
            create_particles(particle_x, particle_y, color, 50)
            
            # Reproduz efeito sonoro (opcional)
            # Se você tiver sons: win_sound.play()
        
        # Determina o próximo sub-tabuleiro
        next_row, next_col = sub_row, sub_col
        
        # Se o próximo sub-tabuleiro já está ganho ou completo, jogador pode escolher qualquer um
        if (won_sub_boards[next_row][next_col] is not None or 
            ai.is_sub_board_full(main_board[next_row][next_col])):
            next_sub_board = None
        else:
            next_sub_board = (next_row, next_col)
        
        # Verifica se o jogo acabou
        x_wins = sum(row.count('X') for row in won_sub_boards)
        o_wins = sum(row.count('O') for row in won_sub_boards)
        
        if x_wins + o_wins == 9 or x_wins > 4 or o_wins > 4:
            game_over = True
        else:
            # Muda o jogador se o jogo não acabou
            current_player = 'O' if player == 'X' else 'X'
        
        return True
    else:
        # A jogada não é válida
        return False

# Função de animação quando o jogo termina
def animate_game_end(winner):
    start_time = pygame.time.get_ticks()
    duration = 2000  # 2 segundos
    
    while pygame.time.get_ticks() - start_time < duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        # Cria partículas em posições aleatórias
        if pygame.time.get_ticks() % 100 < 20:  # Controla a taxa de criação
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT-50)
            if winner == 'X':
                color = CROSS_COLOR
            elif winner == 'O':
                color = CIRCLE_COLOR
            else:
                color = (150, 150, 150)  # Cinza para empate
            create_particles(x, y, color, 5)
        
        # Atualiza e desenha partículas
        update_particles()
        
        # Redesenha a tela do jogo
        draw_board_background()
        draw_lines()
        draw_figures()
        draw_particles()
        
        # Mostra mensagem de vitória com efeito pulsante
        elapsed = pygame.time.get_ticks() - start_time
        scale = 1.0 + 0.1 * math.sin(elapsed / 150)  # Pulsação suave
        
        if winner == human_player:
            msg = "VOCÊ VENCEU!"
            color = CROSS_COLOR if human_player == 'X' else CIRCLE_COLOR
        elif winner == ai_player:
            msg = "IA VENCEU!"
            color = CROSS_COLOR if ai_player == 'X' else CIRCLE_COLOR
        else:
            msg = "EMPATE!"
            color = (100, 100, 100)
            
        font_size = int(48 * scale)
        win_font = pygame.font.SysFont('Arial Black', font_size)
        
        text = win_font.render(msg, True, color)
        text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2 - 50))
        
        # Sombra do texto
        shadow = win_font.render(msg, True, (20, 20, 20))
        shadow_rect = shadow.get_rect(center=(WIDTH/2 + 3, HEIGHT/2 - 47))
        screen.blit(shadow, shadow_rect)
        
        # Texto principal
        screen.blit(text, text_rect)
        
        pygame.display.update()
        clock.tick(60)

# Inicialização das variáveis do jogo
current_player = 'X'  # X começa
next_sub_board = None  # Qualquer um pode ser jogado inicialmente
game_over = False
ai_thinking = False
ai_move_delay = 0
difficulty = 3  # Profundidade padrão para o minimax
human_player = 'X'  # Símbolo do jogador humano (padrão)
ai_player = 'O'    # Símbolo da IA (padrão)

# Timer para animação da IA "pensando"
thinking_timer = 0
thinking_dots = ""

# Função para reiniciar o jogo
def reset_game():
    global main_board, won_sub_boards, current_player, next_sub_board, game_over, particles
    
    main_board = [[[[None]*3 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    won_sub_boards = [[None for _ in range(3)] for _ in range(3)]
    current_player = 'X'  # X sempre começa, independente de quem está jogando com X
    next_sub_board = None
    game_over = False
    particles = []
    
    # Se a IA joga como X, ela começa
    if ai_player == 'X':
        ai_thinking = True
        ai_move_delay = pygame.time.get_ticks()

# Tela inicial
def show_title_screen():
    running = True
    selected_difficulty = "Normal"  # Padrão
    player_symbol = "X"  # Padrão - jogador usa X
    
    while running:
        screen.fill((230, 230, 250))
        
        # Título
        title = title_font.render("JOGO DA VELHA AVANÇADO", True, (50, 50, 120))
        title_rect = title.get_rect(center=(WIDTH/2, HEIGHT/4))
        
        # Sombra do título
        title_shadow = title_font.render("JOGO DA VELHA AVANÇADO", True, (100, 100, 170))
        shadow_rect = title_shadow.get_rect(center=(WIDTH/2 + 3, HEIGHT/4 + 3))
        screen.blit(title_shadow, shadow_rect)
        screen.blit(title, title_rect)
        
        # Instruções
        instructions = [
            "Um jogo da velha com tabuleiros aninhados!",
            "Cada vitória em um sub-tabuleiro conta como um ponto.",
            "A posição da sua jogada determina o próximo sub-tabuleiro.",
            "Vence quem ganhar mais sub-tabuleiros."
        ]
        
        y_pos = HEIGHT/4 + 80
        for instruction in instructions:
            text = small_font.render(instruction, True, TEXT_COLOR)
            text_rect = text.get_rect(center=(WIDTH/2, y_pos))
            screen.blit(text, text_rect)
            y_pos += 25
        
        # Seleção de símbolo
        y_pos = HEIGHT/2 + 20
        symbol_text = font.render("Escolha seu símbolo:", True, TEXT_COLOR)
        symbol_text_rect = symbol_text.get_rect(center=(WIDTH/2, y_pos))
        screen.blit(symbol_text, symbol_text_rect)
        
        y_pos += 40
        symbols = ["X", "O"]
        symbol_rects = []
        
        for i, symbol in enumerate(symbols):
            rect_x = WIDTH/2 - 70 + i * 80
            symbol_rect = pygame.Rect(rect_x, y_pos - 20, 60, 40)
            symbol_rects.append(symbol_rect)
            
            if symbol == player_symbol:
                pygame.draw.rect(screen, (200, 220, 255), symbol_rect, border_radius=5)
                color = (50, 120, 200)
            else:
                color = TEXT_COLOR
            
            text = font.render(symbol, True, color)
            text_rect = text.get_rect(center=symbol_rect.center)
            screen.blit(text, text_rect)
            
            # Efeito hover
            if symbol_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(screen, (220, 230, 255), symbol_rect, width=2, border_radius=5)
        
        # Botões de dificuldade
        difficulties = ["Fácil", "Normal", "Difícil"]
        y_pos += 60
        
        for diff in difficulties:
            if diff == selected_difficulty:
                color = (50, 120, 200)
                pygame.draw.rect(screen, (200, 220, 255), 
                               (WIDTH/2 - 60, y_pos - 15, 120, 40), 
                               border_radius=5)
            else:
                color = TEXT_COLOR
                
            text = font.render(diff, True, color)
            text_rect = text.get_rect(center=(WIDTH/2, y_pos))
            screen.blit(text, text_rect)
            
            # Área clicável
            diff_rect = pygame.Rect(WIDTH/2 - 60, y_pos - 15, 120, 30)
            
            # Verifica cliques
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicked = pygame.mouse.get_pressed()[0]
            
            if diff_rect.collidepoint(mouse_pos):
                if not mouse_clicked:
                    pygame.draw.rect(screen, (220, 230, 255), 
                                   (WIDTH/2 - 60, y_pos - 15, 120, 30), 
                                   width=2, border_radius=5)
                elif mouse_clicked:
                    selected_difficulty = diff
            
            y_pos += 50
        
        # Botão de iniciar
        start_rect = pygame.Rect(WIDTH/2 - 80, HEIGHT - 80, 160, 50)
        pygame.draw.rect(screen, (50, 180, 50), start_rect, border_radius=10)
        
        # Efeito hover
        if start_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(screen, (80, 200, 80), start_rect, border_radius=10)
            
        start_text = font.render("INICIAR JOGO", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=start_rect.center)
        screen.blit(start_text, start_text_rect)
        
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Verifica clique nos símbolos
                for i, rect in enumerate(symbol_rects):
                    if rect.collidepoint(event.pos):
                        player_symbol = symbols[i]
                
                # Verifica clique no botão iniciar
                if start_rect.collidepoint(event.pos):
                    # Define a dificuldade
                    global difficulty, ai_player, human_player, ai_thinking, ai_move_delay
                    if selected_difficulty == "Fácil":
                        difficulty = 4
                    elif selected_difficulty == "Normal":
                        difficulty = 6
                    else:  # Difícil
                        difficulty = 8
                    
                    # Define os símbolos dos jogadores
                    human_player = player_symbol
                    ai_player = "O" if human_player == "X" else "X"
                    
                    # Se a IA começa (se jogador humano escolheu 'O')
                    if human_player == 'O':
                        ai_thinking = True
                        ai_move_delay = pygame.time.get_ticks()
                    
                    running = False  # Sai da tela inicial
                    
        pygame.display.update()
        clock.tick(30)

# Main loop do jogo
clock = pygame.time.Clock()

# Mostrar tela inicial
show_title_screen()

while True:
    screen.fill(BG_COLOR)
    draw_board_background()
    
    # Atualiza e desenha partículas
    update_particles()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over and not ai_thinking:
            mouseX, mouseY = pygame.mouse.get_pos()
            
            # Ignora cliques na área de status
            if mouseY >= HEIGHT - 50:
                continue
                
            clicked_row = int(mouseY // SQUARE_SIZE)
            clicked_col = int(mouseX // SQUARE_SIZE)
            
            sub_clicked_row = int((mouseY % SQUARE_SIZE) // (SQUARE_SIZE / 3))
            sub_clicked_col = int((mouseX % SQUARE_SIZE) // (SQUARE_SIZE / 3))
            
            # Jogada do jogador humano (depende do símbolo escolhido)
            if current_player == human_player:
                # Verifica se o sub-tabuleiro é válido
                if next_sub_board is None or (clicked_row, clicked_col) == next_sub_board:
                    # Tenta fazer a jogada
                    success = play_move(clicked_row, clicked_col, sub_clicked_row, sub_clicked_col, current_player)
                    
                    if success and not game_over:
                        ai_thinking = True
                        ai_move_delay = pygame.time.get_ticks()
                else:
                    display_message(f"Jogue no sub-tabuleiro indicado!", HEIGHT - 70, 'small')
    
    # Desenha o tabuleiro e os símbolos
    draw_lines()
    
    # Destaca o próximo sub-tabuleiro se o jogo não acabou
    if not game_over and current_player == human_player:
        highlight_next_board(next_sub_board)
    
    # Desenha os símbolos
    draw_figures()
    draw_particles()
    
    # Exibe o status do jogo
    x_wins = sum(row.count('X') for row in won_sub_boards)
    o_wins = sum(row.count('O') for row in won_sub_boards)
    display_game_status(current_player, next_sub_board, x_wins, o_wins)
    
    # Jogada da IA
    if current_player == ai_player and not game_over and ai_thinking:
        # Animação de "pensando"
        if pygame.time.get_ticks() % 300 < 20:  # Atualiza a cada 300ms
            thinking_dots = "." * ((thinking_timer % 3) + 1)
            thinking_timer += 1
            
        thinking_text = f"IA pensando{thinking_dots}"
        display_message(thinking_text, HEIGHT/2 - 200, 'normal', 
                      CIRCLE_COLOR if ai_player == 'O' else CROSS_COLOR)
        
        # Atraso para dar a impressão de que a IA está "pensando"
        if pygame.time.get_ticks() - ai_move_delay > 800:  # 800 ms de atraso
            # Obtém a melhor jogada
            try:
                # Se a IA joga com 'O', ela quer maximizar. Se joga com 'X', quer minimizar
                is_maximizing = ai_player == 'O'
                
                reward, best_move = ai.minmax(main_board, difficulty, float('-inf'), float('inf'), 
                                       is_maximizing, next_sub_board)
                print(f"Melhor jogada: {best_move} com recompensa: {reward}")
                
                if best_move:
                    play_move(best_move[0], best_move[1], best_move[2], best_move[3], current_player)
                else:
                    # Se não houver jogada válida, declaramos um empate
                    game_over = True
            except Exception as e:
                print(f"Erro na IA: {e}")
                # Em caso de erro, faz uma jogada aleatória
                possible_moves = ai.get_possible_moves(main_board, next_sub_board)
                if possible_moves:
                    move = random.choice(possible_moves)
                    play_move(move[0], move[1], move[2], move[3], current_player)
                else:
                    game_over = True
                    
            ai_thinking = False
            thinking_timer = 0
    
    # Verifica se o jogo acabou
    if game_over:
        # Determina o vencedor
        x_wins = sum(row.count('X') for row in won_sub_boards)
        o_wins = sum(row.count('O') for row in won_sub_boards)
        
        if x_wins > o_wins:
            winner = 'X'
        elif o_wins > x_wins:
            winner = 'O'
        else:
            winner = None  # Empate
            
        # Animação de fim de jogo
        animate_game_end(winner)
            
        # Exibe mensagem para jogar novamente
        display_message("Pressione 'R' para jogar novamente", HEIGHT - 100, 'normal')
        
        # Verifica se o jogador quer reiniciar
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            reset_game()
            
    # Atualiza o display
    pygame.display.update()
    clock.tick(60)  # Limita a 60 FPS