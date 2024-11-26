import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import os
import datetime

def make_teams(positions, conflicts, coefficients, all_players, skills, all_positions):
    # Criar o diretório para guardar o log dos arquivos
    if not os.path.exists('match_history'):
        os.makedirs('match_history')
    start_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    session_filename = f"sessao_jogo_{start_time}.csv"
    session_filepath = f"match_history/{session_filename}"

    # Iniciar o dataframe da sessão de jogo    
    session_df = pd.DataFrame(columns=['Player', 'Games_Played', 'Games_Won'])
    session_df.to_csv(session_filepath, index=False)
    
    # Inicializar as estatísticas dos jogadores
    player_stats = {player: {'Games_Played': 0, 'Games_Won': 0} for player in all_players}

    # Inicializar partidas consecutivas jogadas
    consecutive_matches_played = {player: 0 for player in all_players}

    # Entrada dos nomes de todos os jogadores presentes (apenas uma vez)
    input_players_string = input("Digite os nomes de todos os jogadores presentes, separados por vírgulas:\n")
    input_players_list = [name.strip() for name in input_players_string.split(',')]
    # Priorizar os primeiros 12 jogadores
    priority_players = input_players_list[:12]
    remaining_players = input_players_list[12:]
    all_present_players = priority_players + remaining_players
    new_players_arrived = []

    # Loop Principal
    match_number = 1
    left_out_players = []
    
    # Loop principal do programa
    while True:
        print(f"\n--- Partida {match_number} ---")

        if match_number == 1:
            # Na primeira partida, os primeiros 12 jogadores são obrigatórios
            mandatory_players = priority_players
            players_to_bench = []
        else:
            # Nas partidas subsequentes, jogadores que ficaram de fora são obrigatórios
            mandatory_players = left_out_players + new_players_arrived
            new_players_arrived = []
            # Determinar jogadores que irão para o banco
            num_players_to_bench = len(left_out_players)
            # Excluir jogadores obrigatórios da consideração
            eligible_players_for_benching = [player for player in assigned_players if player not in mandatory_players]
            # Ordenar jogadores elegíveis por partidas consecutivas jogadas (ordem decrescente)
            eligible_players_for_benching.sort(key=lambda x: consecutive_matches_played[x], reverse=True)
            # Selecionar jogadores para o banco
            players_to_bench = eligible_players_for_benching[:num_players_to_bench]
            print(f"Jogadores que irão para o banco com base em partidas consecutivas jogadas: {players_to_bench}")

        # Jogadores restantes são aqueles não em mandatory_players ou players_to_bench
        remaining_players = [player for player in all_present_players if player not in mandatory_players and player not in players_to_bench]

        # Preparar a lista de jogadores para esta partida
        players = mandatory_players + remaining_players
        skills_subset = {player: skills[player] for player in players if player in skills}
        positions_subset = {player: positions[player] for player in players if player in positions}
        conflicts_subset = {player: conflicts[player] for player in players if player in conflicts}

        # Função para tentar formar times com os requisitos de posição dados
        def form_teams(positions_required_list, num_solutions=10, mandatory_players=None):
            num_teams = len(positions_required_list)
            team_sizes = [sum(positions_required.values()) for positions_required in positions_required_list]
            total_players_required = sum(team_sizes)

            if len(players) < total_players_required:
                print(f"Não há jogadores suficientes para formar {num_teams} times com as posições requeridas. São necessários {total_players_required} jogadores.")
                return []
            else:
                # Verificar se há jogadores suficientes para cada posição em todos os times
                total_positions_required = {}
                for positions_required in positions_required_list:
                    for pos, required in positions_required.items():
                        total_positions_required[pos] = total_positions_required.get(pos, 0) + required

                for pos, total_required in total_positions_required.items():
                    players_who_can_play = [player for player in players if pos in positions_subset[player]]
                    if len(players_who_can_play) < total_required:
                        print(f"Não há jogadores suficientes que possam jogar na posição '{pos}'. Necessário: {total_required}, disponível: {len(players_who_can_play)}")
                        return []

                # Calcular o overall primário e ajustado dos jogadores
                scale_factor = 1000  # Fator de escala para lidar com variáveis inteiras

                primary_overall = {}
                adjusted_overall = {}
                for player in players:
                    skills_values = skills_subset[player]  # Dicionário de habilidades do jogador
                    primary_position = positions_subset[player][0]
                    position_coeffs = coefficients[primary_position]

                    numerator = 0
                    denominator = 0
                    for skill_name in ['Saque', 'Recepcao', 'Levantamento', 'Ataque', 'Bloqueio', 'Defesa']:
                        skill_value = skills_values[skill_name]
                        coeff = position_coeffs[skill_name]
                        numerator += skill_value * coeff
                        denominator += coeff
                    overall = numerator / denominator
                    overall_int = int(overall * scale_factor)  # Escalar para inteiro
                    primary_overall[player] = overall_int
                    adjusted_overall[player] = int(overall * 1 * scale_factor)  # Overall ajustado (sem penalização)

                # Variáveis para armazenar as soluções
                solutions = []
                model = cp_model.CpModel()

                # Inicializar variáveis
                assign = {}
                for player in players:
                    for team in range(num_teams):
                        assign[(player, team)] = model.NewBoolVar(f'assign_{player}_team{team}')

                position_vars = {}
                for player in players:
                    for pos in positions_subset[player]:
                        position_vars[(player, pos)] = model.NewBoolVar(f'position_{player}_{pos}')

                # Criar variáveis auxiliares para atribuições combinadas
                assign_pos = {}
                for player in players:
                    for team in range(num_teams):
                        for pos in positions_subset[player]:
                            assign_pos[(player, team, pos)] = model.NewBoolVar(f'assign_{player}_team{team}_pos{pos}')
                            # assign_pos == 1 se e somente se assign[player, team] == 1 e position_vars[player, pos] == 1
                            model.AddBoolAnd([assign[(player, team)], position_vars[(player, pos)]]).OnlyEnforceIf(assign_pos[(player, team, pos)])
                            model.AddBoolOr([assign[(player, team)].Not(), position_vars[(player, pos)].Not()]).OnlyEnforceIf(assign_pos[(player, team, pos)].Not())

                # Variáveis indicando se um jogador joga em sua posição primária ou não
                primary_positions = {player: positions_subset[player][0] for player in players}
                plays_primary = {}
                plays_not_primary = {}

                for player in players:
                    primary_pos = primary_positions[player]
                    for team in range(num_teams):
                        plays_primary[(player, team)] = model.NewBoolVar(f'plays_primary_{player}_team{team}')
                        plays_not_primary[(player, team)] = model.NewBoolVar(f'plays_not_primary_{player}_team{team}')

                        # plays_primary[player, team] == assign[player, team] AND position_vars[player, primary_pos]
                        model.AddBoolAnd([assign[(player, team)], position_vars[(player, primary_pos)]]).OnlyEnforceIf(plays_primary[(player, team)])
                        model.AddBoolOr([assign[(player, team)].Not(), position_vars[(player, primary_pos)].Not()]).OnlyEnforceIf(plays_primary[(player, team)].Not())

                        # plays_not_primary[player, team] == assign[player, team] AND NOT position_vars[player, primary_pos]
                        model.AddBoolAnd([assign[(player, team)], position_vars[(player, primary_pos)].Not()]).OnlyEnforceIf(plays_not_primary[(player, team)])
                        model.AddBoolOr([assign[(player, team)].Not(), position_vars[(player, primary_pos)] ]).OnlyEnforceIf(plays_not_primary[(player, team)].Not())

                # Variáveis de overall do time
                max_overall = sum(primary_overall.values())  # Máximo possível de overall do time
                team_overall = {}
                for team in range(num_teams):
                    team_overall[team] = model.NewIntVar(0, max_overall, f'team_overall_{team}')

                # Diferença nos overalls dos times
                max_overall_difference = max_overall  # Máxima diferença possível
                overall_difference = model.NewIntVar(0, max_overall_difference, 'overall_difference')

                # Definir a diferença como o valor absoluto da diferença entre os overalls dos times
                model.Add(overall_difference == abs(team_overall[0] - team_overall[1]))

                # Restrições
                # Cada jogador é atribuído a no máximo um time
                for player in players:
                    model.Add(sum(assign[(player, team)] for team in range(num_teams)) <= 1)

                # Jogadores obrigatórios devem ser atribuídos a um time
                if mandatory_players:
                    for player in mandatory_players:
                        if player in players:
                            model.Add(sum(assign[(player, team)] for team in range(num_teams)) == 1)

                # Jogadores em conflito estão em times diferentes
                for player in players:
                    for conflict_player in conflicts_subset[player]:
                        if conflict_player in players:
                            for team in range(num_teams):
                                model.Add(assign[(player, team)] + assign[(conflict_player, team)] <= 1)

                # Cada time tem o número requerido de jogadores
                for team in range(num_teams):
                    model.Add(sum(assign[(player, team)] for player in players) == team_sizes[team])

                # Cada jogador é atribuído a no máximo uma posição que pode jogar
                for player in players:
                    model.Add(sum(position_vars[(player, pos)] for pos in positions_subset[player]) <= 1)

                # Vincular atribuição de jogador à atribuição de posição
                for player in players:
                    model.Add(sum(assign[(player, team)] for team in range(num_teams)) == sum(position_vars[(player, pos)] for pos in positions_subset[player]))

                # Cada posição em um time é preenchida pelo número requerido de jogadores
                for team in range(num_teams):
                    positions_required = positions_required_list[team]
                    for pos, required_number in positions_required.items():
                        model.Add(sum(assign_pos[(player, team, pos)] for player in players if pos in positions_subset[player]) == required_number)

                # Calcular overalls dos times com overalls ajustados
                for team in range(num_teams):
                    total_overall = []
                    for player in players:
                        # Contribuição do jogador para o overall do time
                        primary_contrib = primary_overall[player] * plays_primary[(player, team)]
                        adjusted_contrib = adjusted_overall[player] * plays_not_primary[(player, team)]
                        total_overall.append(primary_contrib + adjusted_contrib)
                    model.Add(team_overall[team] == sum(total_overall))

                # Objetivo
                # Primeiro, minimizar a diferença total de overalls
                # Segundo, maximizar o número de jogadores em suas posições primárias (opcional)
                weight_primary_positions = 1  # Peso para o número de posições primárias
                model.Minimize(overall_difference - weight_primary_positions * sum(plays_primary[(player, team)] for player in players for team in range(num_teams)))

                # Criar o solver e o coletor de soluções
                solver = cp_model.CpSolver()
                solution_limit = num_solutions  # Número de soluções a encontrar

                # Opcional: Definir semente aleatória para reprodutibilidade
                solver.parameters.random_seed = 42

                class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
                    """Coletar soluções."""
                    def __init__(self, variables):
                        cp_model.CpSolverSolutionCallback.__init__(self)
                        self._variables = variables
                        self._solution_count = 0
                        self.solutions = []

                    def on_solution_callback(self):
                        self._solution_count += 1
                        # Coletar a solução
                        solution = {}
                        for v in self._variables:
                            solution[str(v)] = self.Value(v)
                        self.solutions.append(solution)
                        if self._solution_count >= solution_limit:
                            self.StopSearch()

                    def solution_count(self):
                        return self._solution_count

                # Preparar a lista de variáveis para monitorar
                variables = []
                for var_dict in [assign, position_vars, assign_pos, plays_primary, plays_not_primary]:
                    variables.extend(var_dict.values())

                # Resolver e coletar soluções
                solution_printer = VarArraySolutionPrinter(variables)
                status = solver.Solve(model, solution_printer)

                if solution_printer.solution_count() > 0:
                    solutions_data = []
                    for idx, solution in enumerate(solution_printer.solutions):
                        # Reconstruir variáveis da solução
                        assign_solution = {}
                        position_vars_solution = {}
                        assign_pos_solution = {}
                        plays_primary_solution = {}
                        plays_not_primary_solution = {}
                        for key in assign:
                            var_name = f"assign_{key[0]}_team{key[1]}"
                            assign_solution[key] = solution[var_name]
                        for key in position_vars:
                            var_name = f"position_{key[0]}_{key[1]}"
                            position_vars_solution[key] = solution[var_name]
                        for key in assign_pos:
                            var_name = f"assign_{key[0]}_team{key[1]}_pos{key[2]}"
                            assign_pos_solution[key] = solution[var_name]
                        for key in plays_primary:
                            var_name = f"plays_primary_{key[0]}_team{key[1]}"
                            plays_primary_solution[key] = solution[var_name]
                        for key in plays_not_primary:
                            var_name = f"plays_not_primary_{key[0]}_team{key[1]}"
                            plays_not_primary_solution[key] = solution[var_name]

                        teams = {team: [] for team in range(num_teams)}
                        team_positions = {team: {} for team in range(num_teams)}
                        team_overalls_result = {team: 0 for team in range(num_teams)}
                        assigned_players = []
                        for player in players:
                            for team in range(num_teams):
                                if assign_solution[(player, team)] == 1:
                                    teams[team].append(player)
                                    assigned_players.append(player)
                                    # Contribuição do jogador para o overall do time
                                    if plays_primary_solution[(player, team)] == 1:
                                        team_overalls_result[team] += primary_overall[player]
                                    elif plays_not_primary_solution[(player, team)] == 1:
                                        team_overalls_result[team] += adjusted_overall[player]
                                    for pos in positions_subset[player]:
                                        if assign_pos_solution.get((player, team, pos), 0) == 1:
                                            team_positions[team].setdefault(pos, []).append(player)
                        # Coletar os dados
                        solution_data = {
                            'teams': teams,
                            'team_positions': team_positions,
                            'team_overalls_result': team_overalls_result,
                            'assigned_players': assigned_players,
                            'left_out_players': [player for player in players if player not in assigned_players],
                            'team_sizes': team_sizes,
                            'primary_positions': primary_positions,
                            'primary_overall': primary_overall,
                            'adjusted_overall': adjusted_overall,
                            'scale_factor': scale_factor,
                            'positions_required_list': positions_required_list,
                            'overall_difference': solver.Value(overall_difference)
                        }
                        solutions_data.append(solution_data)
                        if idx + 1 >= num_solutions:
                            break
                    return solutions_data
                else:
                    return []

        # Lista de configurações disponíveis (mantida igual)
        configurations = [
            # Configuração 1
            [
                {'Levantador': 1, 'Ponta': 2, 'Meio': 1, 'Libero': 1, 'Saida': 1},
                {'Levantador': 1, 'Ponta': 2, 'Meio': 1, 'Libero': 1, 'Saida': 1}
            ],
            # Configuração 2
            [
                {'Levantador': 1, 'Ponta': 2, 'Meio': 1, 'Libero': 1, 'Saida': 1},
                {'Levantador': 1, 'Ponta': 2, 'Meio': 2, 'Saida': 1}
            ],
            # Configuração 3
            [
                {'Levantador': 1, 'Ponta': 2, 'Meio': 2, 'Saida': 1},
                {'Levantador': 1, 'Ponta': 2, 'Meio': 2, 'Saida': 1}
            ]
        ]

        # Exibir as configurações disponíveis
        print("\nConfigurações disponíveis para a formação dos times:")
        for idx, config in enumerate(configurations):
            print(f"\nOpção {idx + 1}:")
            for team_idx, team_config in enumerate(config):
                print(f"  Time {team_idx + 1}:")
                for position, count in team_config.items():
                    print(f"    {position}: {count}")

        # Para cada configuração, gerar soluções
        all_solutions = []
        for idx, positions_required_list in enumerate(configurations):
            print(f"\nGerando soluções para a Opção {idx + 1}...")
            solutions_data = form_teams(positions_required_list, num_solutions=10, mandatory_players=mandatory_players)
            if solutions_data:
                for solution in solutions_data:
                    all_solutions.append((idx + 1, solution))
            else:
                print(f"Não foi possível gerar soluções para a Opção {idx + 1}.")

        if not all_solutions:
            print("\nNão foi possível formar times com as configurações disponíveis e os jogadores presentes.")
            # Decidir se continua ou para
            continue_prompt = input("Deseja tentar novamente? (sim/não): ").lower()
            if continue_prompt != 'sim':
                break
            else:
                continue  # Reiniciar o loop

        # Ordenar as soluções por menor diferença de overall
        all_solutions.sort(key=lambda x: x[1]['overall_difference'])

        # Exibir as melhores soluções geradas e perguntar qual o usuário deseja escolher
        print("\nAs melhores formações de times encontradas (ordenadas pela menor diferença de overall):")
        for idx, (config_number, solution_data) in enumerate(all_solutions):
            print(f"\nOpção {idx + 1} (Configuração {config_number}), Diferença de Overall Total: {solution_data['overall_difference']/solution_data['scale_factor']:.2f}")
            teams = solution_data['teams']
            team_positions = solution_data['team_positions']
            team_overalls_result = solution_data['team_overalls_result']
            assigned_players = solution_data['assigned_players']
            left_out_players = solution_data['left_out_players']
            team_sizes = solution_data['team_sizes']
            primary_positions = solution_data['primary_positions']
            primary_overall = solution_data['primary_overall']
            adjusted_overall = solution_data['adjusted_overall']
            scale_factor = solution_data['scale_factor']
            positions_required_list = solution_data['positions_required_list']

            for team in range(len(positions_required_list)):
                team_total_overall = team_overalls_result[team] / scale_factor
                team_average_overall = team_total_overall / team_sizes[team]
                print(f"\n  Time {team + 1} (Overall Médio: {team_average_overall:.2f}, Overall Total: {team_total_overall:.2f}):")
                print("    Jogadores:", teams[team])
                print("    Posições:")
                positions_required = positions_required_list[team]
                for pos in positions_required.keys():
                    players_in_position = team_positions[team].get(pos, [])
                    for player in players_in_position:
                        if primary_positions[player] == pos:
                            player_overall = primary_overall[player] / scale_factor
                            flag = '(Posição Primária)'
                        else:
                            player_overall = adjusted_overall[player] / scale_factor
                            flag = '(Overall Ajustado)'
                        print(f"      {pos}: {player} {flag}, Overall: {player_overall:.2f}")
            if left_out_players:
                print("    Jogadores não escalados:", left_out_players)
            else:
                print("    Todos os jogadores foram escalados.")

        # Perguntar ao usuário qual opção deseja escolher
        while True:
            try:
                solution_choice = int(input(f"\nEscolha o número da opção desejada (1 a {len(all_solutions)}): "))
                if 1 <= solution_choice <= len(all_solutions):
                    _, solution_data = all_solutions[solution_choice - 1]
                    break
                else:
                    print(f"Por favor, insira um número entre 1 e {len(all_solutions)}.")
            except ValueError:
                print("Entrada inválida. Por favor, insira um número válido.")

        # Utilizar a solução escolhida para prosseguir
        teams = solution_data['teams']
        team_positions = solution_data['team_positions']
        team_overalls_result = solution_data['team_overalls_result']
        assigned_players = solution_data['assigned_players']
        left_out_players = solution_data['left_out_players']
        team_sizes = solution_data['team_sizes']
        primary_positions = solution_data['primary_positions']
        primary_overall = solution_data['primary_overall']
        adjusted_overall = solution_data['adjusted_overall']
        scale_factor = solution_data['scale_factor']
        positions_required_list = solution_data['positions_required_list']

        # Exibir os times escolhidos
        print("\nTimes escolhidos:")
        for team in range(len(positions_required_list)):
            team_total_overall = team_overalls_result[team] / scale_factor
            team_average_overall = team_total_overall / team_sizes[team]
            print(f"\nTime {team + 1} (Overall Médio: {team_average_overall:.2f}, Overall Total: {team_total_overall:.2f}):")
            print("Jogadores:", teams[team])
            print("Posições:")
            positions_required = positions_required_list[team]
            for pos in positions_required.keys():
                players_in_position = team_positions[team].get(pos, [])
                for player in players_in_position:
                    if primary_positions[player] == pos:
                        player_overall = primary_overall[player] / scale_factor
                        flag = '(Posição Primária)'
                    else:
                        player_overall = adjusted_overall[player] / scale_factor
                        flag = '(Overall Ajustado)'
                    print(f"  {pos}: {player} {flag}, Overall: {player_overall:.2f}")

        # Exibir jogadores não escalados
        if left_out_players:
            print("\nJogadores não escalados:")
            for player in left_out_players:
                print(f"  {player}")
        else:
            print("\nTodos os jogadores foram escalados.")
        print("-" * 40)

        # Atualizar estatísticas dos jogadores
        for player in assigned_players:
            player_stats[player]['Games_Played'] += 1
            consecutive_matches_played[player] += 1  # Incrementar partidas consecutivas jogadas
        for player in left_out_players:
            consecutive_matches_played[player] = 0  # Reiniciar partidas consecutivas jogadas

        # Perguntar qual time venceu
        while True:
            winning_team_input = input("Qual time venceu? (1/2): ")
            if winning_team_input in ['1', '2']:
                winning_team = int(winning_team_input)
                break
            else:
                print("Entrada inválida. Por favor, insira 1 ou 2.")

        if winning_team == 1:
            winning_players = teams[0]
        else:
            winning_players = teams[1]

        for player in winning_players:
            player_stats[player]['Games_Won'] += 1

        # Salvar as estatísticas atualizadas dos jogadores no arquivo da sessão
        session_df = pd.DataFrame([
            {'Player': player, 'Games_Played': stats['Games_Played'], 'Games_Won': stats['Games_Won']}
            for player, stats in player_stats.items()
            if stats['Games_Played'] > 0
        ])
        session_df.to_csv(session_filepath, index=False)

        # Perguntar se novos jogadores chegaram
        new_players_prompt = input("Algum novo jogador chegou? (sim/não): ").lower()
        if new_players_prompt == 'sim':
            new_players_string = input("Digite os nomes dos novos jogadores, separados por vírgulas: ")
            new_players_list = [name.strip() for name in new_players_string.split(',')]
            # Adicionar novos jogadores à lista de jogadores presentes
            for player in new_players_list:
                if player not in all_players:
                    print(f"Jogador '{player}' não encontrado nos dados. Certifique-se de que o nome está correto.")
                elif player in all_present_players:
                    print(f"Jogador '{player}' já está na lista de jogadores presentes.")
                else:
                    all_present_players.append(player)
                    new_players_arrived.append(player)
                    print(f"Jogador '{player}' adicionado à lista de jogadores presentes.")

        # Perguntar se o usuário deseja continuar
        continue_prompt = input("Deseja agendar outra partida? (sim/não): ").lower()
        if continue_prompt != 'sim':
            break
        else:
            match_number += 1
            # O loop continuará
            continue

    # Ao final, preparar o dataframe final com as estatísticas
    final_stats_df = pd.DataFrame([
        {'Player': player, 'Games_Played': stats['Games_Played'], 'Games_Won': stats['Games_Won']}
        for player, stats in player_stats.items()
        if stats['Games_Played'] > 0
    ])

    return final_stats_df
