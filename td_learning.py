import json
import copy
import random
import numpy as np
import os

from move import convert_board_format, move_marbles, \
    parse_move_input, move_validation
from next_move_generator import generate_all_next_moves
from AI import find_best_move, board_to_key, WEIGHTS as AI_WEIGHTS
from AI import  evaluate_edge_safety, evaluate_mobility, find_groups_fast


class AbaloneAgent:
    def __init__(self, weights=None, learning_rate=0.0025, discount_factor=0.92, epsilon=0.06, epsilon_decay=0.995):
        """
        아발론 게임을 위한 TD 학습 에이전트 초기화

        매개변수:
            weights (dict): 특성에 대한 가중치 사전 (None이면 기본값 사용)
            learning_rate (float): 학습률
            discount_factor (float): 할인 계수
            epsilon (float): 탐험 확률
            epsilon_decay (float): 탐험 확률 감소율
        """
        # AI.py에서 가중치 복사 (동일한 기본 가중치 사용)
        self.weights = weights if weights is not None else copy.deepcopy(AI_WEIGHTS)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.games_played = 0
        self.wins = {"Black": 0, "White": 0, "Draw": 0}
        self.stats = {"avg_moves": 0, "total_moves": 0}
        self.training_phase = "exploration"  # "exploration", "exploitation", "refinement"
        self.consecutive_draws = 0
        self.best_win_ratio = 0.0

        # 경로 설정 개선
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = os.path.join(self.base_dir, "weights")
        self.games_dir = os.path.join(self.base_dir, "games")

        # 디렉토리 생성 및 권한 확인
        for dir_path in [self.weights_dir, self.games_dir]:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"디렉토리 확인: {dir_path}")
                # 쓰기 권한 테스트
                test_file = os.path.join(dir_path, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"쓰기 권한 확인: {dir_path}")
            except Exception as e:
                print(f"경고: 디렉토리 {dir_path} 생성 또는 쓰기 권한 문제: {e}")

        # 베스트 가중치 저장
        self.best_weights = copy.deepcopy(self.weights)

    def load_weights(self, filename):
        """저장된 가중치 파일 불러오기"""
        full_path = os.path.join(self.weights_dir, os.path.basename(filename))
        try:
            with open(full_path, 'r') as f:
                self.weights = json.load(f)
            print(f"{full_path}에서 가중치 불러옴")
            return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"가중치 로드 오류: {e}")
            print(f"시도한 경로: {full_path}")
            return False

    def save_weights(self, filename, weights_to_save=None):
        """현재 가중치를 파일에 저장"""
        full_path = os.path.join(self.weights_dir, os.path.basename(filename))
        try:
            # 문제가 있는 값 감지 및 처리를 위한 기본 가중치 사전 (AI.py와 동일)
            default_weights = copy.deepcopy(AI_WEIGHTS)

            if weights_to_save is None:
                weights_to_save = self.weights

            # NaN 또는 무한대 값 확인 및 기본값으로 대체
            weights_copy = copy.deepcopy(weights_to_save)
            for key, value in weights_copy.items():
                if np.isnan(value) or np.isinf(value):
                    print(f"경고: {key} 가중치에 잘못된 값 {value}가 있어 기본값으로 재설정")
                    weights_copy[key] = default_weights.get(key, 0.5)  # 키가 없으면 0.5 사용

            with open(full_path, 'w') as f:
                json.dump(weights_copy, f, indent=2)
            print(f"{full_path}에 가중치 저장됨")
            return True
        except Exception as e:
            print(f"가중치 저장 오류: {e}")
            print(f"시도한 경로: {full_path}")
            import traceback
            traceback.print_exc()  # 상세 오류 출력
            return False

    def evaluate_board_with_weights(self, board, player):
        """자체 가중치를 사용하여 보드 평가 - AI.py의 최적화된 평가 함수 사용"""
        features = self.get_features(board, player)

        # NaN 및 Inf 방지를 위한 계산 보호
        score = 0.0
        for name, value in features.items():
            if name in self.weights and not np.isnan(value) and not np.isinf(value):
                weight = self.weights.get(name, 0)
                if not np.isnan(weight) and not np.isinf(weight):
                    score += weight * value

        return score

    def get_features(self, board, player):
        """
        AI.py의 최적화된 평가 함수와 동일한 특성 추출
        """
        # 플레이어/상대방 구슬 인덱스 결정
        friend_idx = 0 if player.lower() == "black" else 1
        enemy_idx = 1 if player.lower() == "black" else 0

        friend_marbles = board[friend_idx]
        enemy_marbles = board[enemy_idx]

        # 더 빠른 집합 연산을 위해 튜플로 변환
        friend_positions = [tuple(pos) for pos in friend_marbles]
        enemy_positions = [tuple(pos) for pos in enemy_marbles]
        friend_set = set(friend_positions)
        enemy_set = set(enemy_positions)

        # 특성 1: 게임 진행에 따른 가중치를 적용한 구슬 수 차이
        friend_count = len(friend_positions)
        enemy_count = len(enemy_positions)
        marble_diff = friend_count - enemy_count
        total_lost = (10 * 2) - (friend_count + enemy_count)
        progress_factor = 1.0 + total_lost / 6.0  # 게임 후반에 더 중요해짐
        marble_feature = marble_diff * progress_factor

        # 특성 2: 중앙성
        from AI import coord_to_index, CENTRALITY_VALUES
        friend_centrality = sum(
            CENTRALITY_VALUES[coord_to_index(pos)] for pos in friend_positions if coord_to_index(pos) >= 0)
        enemy_centrality = sum(
            CENTRALITY_VALUES[coord_to_index(pos)] for pos in enemy_positions if coord_to_index(pos) >= 0)
        centrality_score = friend_centrality - enemy_centrality

        # 그룹 찾기 (한 번만 계산)
        friend_groups = find_groups_fast(friend_marbles)
        enemy_groups = find_groups_fast(enemy_marbles)

        # 특성 3: 밀기 능력 평가 (sumito)
        friend_sumito = 0
        enemy_sumito = 0
        sumito_score = friend_sumito - enemy_sumito

        # 특성 4: 가장자리 안전성 (edge safety)
        friend_edge_safety = evaluate_edge_safety(friend_positions, enemy_marbles)
        enemy_edge_safety = evaluate_edge_safety(enemy_positions, friend_marbles)
        edge_safety_score = friend_edge_safety - enemy_edge_safety

        # 특성 5: 구슬 진형 평가 (formation)
        friend_formation = 0
        enemy_formation = 0
        formation_score = friend_formation - enemy_formation

        # 특성 6: 이동성 (mobility)
        friend_mobility = evaluate_mobility(friend_positions, friend_set, enemy_set)
        enemy_mobility = evaluate_mobility(enemy_positions, enemy_set, friend_set)
        mobility_score = friend_mobility - enemy_mobility

        # 최종 특성 반환
        return {
            'marble_diff': marble_feature,
            'centrality': centrality_score,
            'sumito': sumito_score,
            'edge_safety': edge_safety_score,
            'formation': formation_score,
            'mobility': mobility_score
        }

    def _check_edge_threat(self, board, player):
        """
        상대 구슬이 가장자리에 위협받고 있는지 확인
        """
        friend_idx = 0 if player.lower() == "black" else 1
        enemy_idx = 1 if player.lower() == "black" else 0

        friend_positions = [tuple(pos) for pos in board[friend_idx]]
        enemy_positions = [tuple(pos) for pos in board[enemy_idx]]

        # evaluate_edge_safety 함수를 직접 활용
        enemy_danger = -evaluate_edge_safety(enemy_positions, board[friend_idx])

        return max(0, enemy_danger)  # 음수 값은 0으로 처리 (위험이 없음)

    def choose_move(self, board, player, depth=3, time_limit=8.0, recent_boards=None, strategy_bias=None):
        """
        현재 보드 상태에서 다음 이동 선택 - 전략 편향 옵션 추가, 딕셔너리 형식 반환 처리
        """
        # 전략 편향 적용
        original_weights = None
        if strategy_bias:
            original_weights = copy.deepcopy(self.weights)
            if strategy_bias == "aggressive":
                # 공격적 전략: 밀기 중시, 구슬 차이 중시, 가장자리 안전성 경시
                self.weights["sumito"] = self.weights["sumito"] * 1.5
                self.weights["marble_diff"] = self.weights["marble_diff"] * 1.3
                self.weights["edge_safety"] = self.weights["edge_safety"] * 0.7
            elif strategy_bias == "defensive":
                # 수비적 전략: 구슬 보존, 안전성 강화
                self.weights["marble_diff"] = self.weights["marble_diff"] * 1.5
                self.weights["edge_safety"] = self.weights["edge_safety"] * 1.3
                self.weights["formation"] = self.weights["formation"] * 1.2
            elif strategy_bias == "balanced":
                # 균형잡힌 전략: 원래 가중치 사용
                pass

        # 게임 단계 감지 및 깊이 조정
        black_lost = 14 - len(board[0])
        white_lost = 14 - len(board[1])
        total_lost = black_lost + white_lost

        adaptive_depth = 3

        try:
            # 탐험 확률에 따른 무작위 이동
            if random.random() < self.epsilon:
                print(f"epsilon = {self.epsilon:.4f}로 탐험 중")
                color = "BLACK" if player.lower() == "black" else "WHITE"
                all_moves_dict = generate_all_next_moves(board, color)

                if all_moves_dict:
                    # 공격적인 이동 식별 및 우선순위
                    aggressive_moves = []
                    defensive_moves = []

                    for move_key, move in all_moves_dict.items():
                        # 구슬 수 비교로 잡기 확인
                        captures = (player.lower() == "black" and len(move[1]) < len(board[1])) or \
                                   (player.lower() == "white" and len(move[0]) < len(board[0]))

                        if captures:
                            # 잡는 수에 높은 우선순위
                            aggressive_moves.append((move_key, move, 3))  # 확실히 높은 우선순위
                        else:
                            # 밀기 능력 또는 가장자리 위협 평가
                            edge_threat = self._check_edge_threat(move, player)
                            if edge_threat > 0:
                                aggressive_moves.append((move_key, move, 2))  # 중간 우선순위

                            # 좋은 평가점수를 갖는 방어적 이동 찾기
                            if len(move[0 if player.lower() == "black" else 1]) == len(
                                    board[0 if player.lower() == "black" else 1]):
                                # 구슬을 잃지 않는 이동
                                eval_score = self.evaluate_board_with_weights(move, player)
                                if eval_score > 0:
                                    defensive_moves.append((move_key, move, eval_score))

                    # 반복 상태를 피하는 이동 필터링
                    valid_moves = [(move_key, move) for move_key, move in all_moves_dict.items()
                                   if self._is_non_repetitive(move, recent_boards)]

                    if valid_moves:
                        # 상황에 따라 공격적 또는 방어적 이동 선호
                        if self.training_phase == "exploration" or total_lost >= 4:
                            # 초기 훈련 또는 게임 중반 이상: 공격적인 플레이 선호
                            if aggressive_moves:
                                aggressive_moves.sort(key=lambda x: x[2], reverse=True)
                                valid_aggressive = [(k, m) for k, m, _ in aggressive_moves
                                                    if self._is_non_repetitive(m, recent_boards)]
                                if valid_aggressive:
                                    print(f"공격적인 이동 선택!")
                                    return valid_aggressive[0][1]  # 이동 상태 반환

                        if defensive_moves and total_lost <= 3:
                            # 게임 초반: 방어적 플레이 고려
                            defensive_moves.sort(key=lambda x: x[2], reverse=True)
                            valid_defensive = [(k, m) for k, m, _ in defensive_moves[:3]
                                               if self._is_non_repetitive(m, recent_boards)]
                            if valid_defensive:
                                print(f"방어적인 이동 선택!")
                                return valid_defensive[0][1]  # 이동 상태 반환

                        # 위 특수 케이스가 없으면 랜덤 선택
                        chosen_key, chosen_move = random.choice(valid_moves)
                        return chosen_move

                    # 반복 회피 이동이 없으면 그냥 무작위 선택
                    chosen_key, chosen_move = random.choice(list(all_moves_dict.items()))
                    return chosen_move
                else:
                    # 유효한 이동이 없음 - 최선의 이동 찾기
                    return find_best_move(board, player, adaptive_depth, time_limit, generate_all_next_moves)[0]

            # 알파-베타 탐색으로 최적 이동 찾기
            best_move, _ = find_best_move(board, player, adaptive_depth, time_limit, generate_all_next_moves)

            # 최적 이동이 반복 상태를 만들면 대안 찾기
            if recent_boards and not self._is_non_repetitive(best_move, recent_boards):
                color = "BLACK" if player.lower() == "black" else "WHITE"
                all_moves_dict = generate_all_next_moves(board, color)

                valid_moves = [(move_key, move) for move_key, move in all_moves_dict.items()
                               if self._is_non_repetitive(move, recent_boards)]

                if valid_moves:
                    # 가장 좋은 반복되지 않는 이동 선택, 잡는 수 우선
                    best_alternative = None
                    best_alt_key = None
                    best_alt_score = float('-inf')

                    for move_key, move in valid_moves:
                        # 이동이 상대방 구슬을 잡는지 확인
                        captures = (player.lower() == "black" and len(move[1]) < len(board[1])) or \
                                   (player.lower() == "white" and len(move[0]) < len(board[0]))

                        score = self.evaluate_board_with_weights(move, player)
                        # 잡는 수에 보너스 부여
                        if captures:
                            score += 3.0  # 더 높은 보너스

                        if score > best_alt_score:
                            best_alt_score = score
                            best_alternative = move
                            best_alt_key = move_key

                    if best_alternative:
                        print("반복 회피 대안 이동 선택!")
                        return best_alternative

            return best_move

        finally:
            # 전략 편향 제거 및 원래 가중치로 복원
            if original_weights:
                self.weights = original_weights

    def _is_non_repetitive(self, move, recent_boards):
        """최근 상태로 돌아가는 이동인지 확인"""
        if not recent_boards:
            return True

        # 보드 상태를 문자열로 변환하여 비교
        move_key = board_to_key(move)
        return move_key not in recent_boards

    def update_weights(self, state, next_state, player, reward=0):
        """
        향상된 TD 학습으로 가중치 업데이트
        """
        if state is None or next_state is None:
            return

        try:
            current_value = self.evaluate_board_with_weights(state, player)
            next_value = self.evaluate_board_with_weights(next_state, player)

            # 정규화된 TD 오차
            td_error = reward + self.discount_factor * next_value - current_value

            # NaN 또는 Inf 값 처리
            if np.isnan(td_error) or np.isinf(td_error):
                print(f"경고: TD 오차가 {td_error}입니다. 업데이트 건너뜀")
                return

            # 안정적인 학습을 위한 클리핑
            td_error = max(min(td_error, 5.0), -5.0)  # 더 넓은 범위로 조정

            features = self.get_features(state, player)

            # 학습률 동적 조정
            adaptive_learning_rate = self.learning_rate
            if self.training_phase == "exploitation":
                adaptive_learning_rate *= 0.7  # 학습률 감소
            elif self.training_phase == "refinement":
                adaptive_learning_rate *= 0.5  # 더 작은 학습률

            for name, value in features.items():
                if name in self.weights:
                    # 가중치 업데이트 계산
                    delta = adaptive_learning_rate * td_error * value

                    # 유효하지 않은 업데이트 건너뜀
                    if np.isnan(delta) or np.isinf(delta):
                        continue

                    # 특성별 차등 학습
                    if name == "sumito":
                        # sumito는 중요한 특성이므로 학습 강화
                        delta *= 1.2
                    elif name == "edge_safety":
                        # edge_safety도 중요한 특성
                        delta *= 1.1

                    # 큰 변화 제한
                    delta = max(min(delta, 0.1), -0.1)

                    # 업데이트 적용
                    self.weights[name] += delta

                    # 가중치를 합리적인 범위 내로 유지
                    if abs(self.weights[name]) > 10.0:
                        self.weights[name] = np.sign(self.weights[name]) * 10.0

                    # 각 특성에 대한 가중치 범위 제한 및 조정
                    if name == 'marble_diff' and self.weights[name] < 0.1:
                        self.weights[name] = 0.1  # 항상 양수 유지

                    # sumito 가중치는 항상 양수로 유지
                    if name == 'sumito' and self.weights[name] < 0.1:
                        self.weights[name] = 0.1

                    # edge_safety는 항상 음수로 유지 (위험에 패널티)
                    if name == 'edge_safety' and self.weights[name] > -0.1:
                        self.weights[name] = -0.1
        except Exception as e:
            print(f"update_weights 오류: {e}")

    def play_self_game(self, initial_board=None, max_moves=150, strategy_bias=None):
        """
        향상된 학습 기능을 갖춘 자가 학습 게임 - 딕셔너리 형식 이동 처리
        """
        from IO import BoardIO

        if initial_board is None:
            board_dict = copy.deepcopy(BoardIO.BOARD_SAMPLE)
            board = convert_board_format(board_dict)
        else:
            board = copy.deepcopy(initial_board)

        current_player = "Black"
        moves_made = 0
        game_history = []
        black_lost = white_lost = 0

        # 반복 상태 감지용
        recent_board_states = {}  # 키: 보드 상태, 값: 횟수

        game_history.append((copy.deepcopy(board), current_player))

        print(f"게임 {self.games_played + 1} 시작...")

        # 공격적 플레이 장려를 위한 카운터
        no_capture_turns = 0

        # 각 플레이어의 전략 설정
        black_strategy = strategy_bias
        white_strategy = strategy_bias

        # 게임 단계별로 전략 변경
        if strategy_bias is None:
            # 무작위로 다양한 전략 시도
            strategies = ["aggressive", "defensive", "balanced"]
            if random.random() < 0.3:  # 60% 확률로 상대 전략
                black_strategy = random.choice(strategies)
                white_strategy = random.choice(strategies)

        while moves_made < max_moves:
            start_time = time.time()

            # 20턴마다 상태 요약 로그 추가
            if moves_made % 20 == 0 and moves_made > 0:
                print(f"\n=== 게임 상태 요약 (턴 {moves_made}) ===")
                print(f"흑 구슬: {len(board[0])}, 백 구슬: {len(board[1])}")
                print(f"흑 잃은 구슬: {black_lost}, 백 잃은 구슬: {white_lost}")
                print(f"잡기 없는 턴: {no_capture_turns}")
                print(f"현재 epsilon: {self.epsilon:.4f}")
                print("현재 가중치:")
                for k, v in self.weights.items():
                    print(f"  {k}: {v:.4f}")
                print("================================\n")

            # 현재 보드 상태 추적
            board_key = board_to_key(board)
            if board_key in recent_board_states:
                recent_board_states[board_key] += 1
            else:
                recent_board_states[board_key] = 1

            # 메모리 관리 - 최근 30개 상태까지 기억
            if len(recent_board_states) > 30:
                # 가장 낮은 빈도의 상태를 삭제
                min_count_key = min(recent_board_states.items(), key=lambda x: x[1])[0]
                del recent_board_states[min_count_key]

            # 현재 플레이어에 맞는 전략 선택
            current_strategy = black_strategy if current_player == "Black" else white_strategy

            # 게임 진행 상황에 따른 전략 조정
            if moves_made > max_moves * 0.7:  # 게임 후반
                if black_lost > white_lost + 1:  # 흑이 지고 있으면
                    if current_player == "Black":
                        current_strategy = "aggressive"  # 더 공격적으로
                elif white_lost > black_lost + 1:  # 백이 지고 있으면
                    if current_player == "White":
                        current_strategy = "aggressive"  # 더 공격적으로

            # 깊이를 동적으로 조정
            depth = 2  # 기본 깊이
            if no_capture_turns > 30:
                depth = 3  # 잡기가 없으면 더 깊게 탐색


            next_board = self.choose_move(board, current_player, depth=depth,
                                          time_limit=8.0,
                                          recent_boards=recent_board_states,
                                          strategy_bias=current_strategy)

            end_time = time.time()

            if next_board is None or next_board == board:
                print(f"{current_player}의 유효한 이동이 없습니다. 게임 종료.")
                winner = "White" if current_player == "Black" else "Black"
                break

            # 새로운 형식에 맞게 이동 문자열 표시 - next_board에서 찾을 수 없기 때문에 보드 간 차이 비교로 검출
            color = "BLACK" if current_player == "Black" else "WHITE"
            all_moves_dict = generate_all_next_moves(board, color)

            # 현재 보드와 같은 다음 보드 상태 찾기
            move_str = None
            for move_key, move_board in all_moves_dict.items():
                if board_to_key(move_board) == board_to_key(next_board):
                    from AI import get_move_string_from_key
                    move_str = get_move_string_from_key(move_key)
                    break

            # 이동 키를 찾지 못한 경우 기본 설명 제공
            if not move_str:
                # 구슬 개수 변화로 이동 추론
                if len(next_board[0]) != len(board[0]) or len(next_board[1]) != len(board[1]):
                    if current_player == "Black" and len(next_board[1]) < len(board[1]):
                        move_str = f"백 구슬 {len(board[1]) - len(next_board[1])}개 밀어냄"
                    elif current_player == "White" and len(next_board[0]) < len(board[0]):
                        move_str = f"흑 구슬 {len(board[0]) - len(next_board[0])}개 밀어냄"
                    else:
                        move_str = "구슬 이동"
                else:
                    move_str = "구슬 이동"

            print(f"이동 {moves_made + 1}: {current_player}가 {move_str} 실행 ({end_time - start_time:.2f}초)")

            game_history.append((copy.deepcopy(next_board), current_player))

            # 나머지 함수 내용은 변경하지 않음...
            # 반복 상태 감지 및 페널티 적용
            next_board_key = board_to_key(next_board)
            repetition_count = recent_board_states.get(next_board_key, 0)
            repetition_penalty = 0

            if repetition_count > 0:
                # 선형 증가 페널티로 변경
                repetition_penalty = -0.3 * repetition_count
                print(f"반복 감지 ({repetition_count}회). 페널티: {repetition_penalty:.2f}")

                # 심각한 반복에 대한 페널티 (완화됨)
                if repetition_count >= 3:
                    print(f"심각한 반복 상태 감지! 강력한 페널티 적용")
                    repetition_penalty = -1.5  # 완화된 고정 페널티

            # 잡기 확인 - 보상 증가
            prev_black_count = len(board[0])
            prev_white_count = len(board[1])
            next_black_count = len(next_board[0])
            next_white_count = len(next_board[1])

            capture_happened = False
            capture_reward = 0

            if next_black_count < prev_black_count:
                black_lost += (prev_black_count - next_black_count)
                capture_reward = 9.0 * (prev_black_count - next_black_count)  # 증가된 잡기 보상
                print(f"흑이 {prev_black_count - next_black_count}개 구슬을 잃었습니다. 총: {black_lost}")
                capture_happened = True

            if next_white_count < prev_white_count:
                white_lost += (prev_white_count - next_white_count)
                capture_reward =9.0 * (prev_white_count - next_white_count)  # 증가된 잡기 보상
                print(f"백이 {prev_white_count - next_white_count}개 구슬을 잃었습니다. 총: {white_lost}")
                capture_happened = True

            # 소극적 플레이 추적 및 페널티 증가
            if capture_happened:
                no_capture_turns = 0
            else:
                no_capture_turns += 1

            # 소극적 플레이 페널티
            passive_play_penalty = 0
            if no_capture_turns > 10:
                # 기본 페널티
                passive_play_penalty = -0.5 * (no_capture_turns - 10) / 10

                # 장기간 소극적 플레이에 대한 추가 페널티
                if no_capture_turns > 20:
                    passive_play_penalty *= 2.0  # 20턴 이상 소극적 플레이시 페널티 2배

                print(f"{no_capture_turns}턴 동안 잡기 없음. 소극적 플레이 페널티: {passive_play_penalty:.2f}")

            # 게임 종료 조건 확인
            if black_lost >= 6 or white_lost >= 6:
                winner = "White" if black_lost >= 6 else "Black"
                print(f"게임 종료! {winner}가 6개 구슬을 밀어내 승리했습니다.")
                break

            # 게임 진행 상황에 따른 보상 및 페널티 동적 조정
            game_progress = moves_made / max_moves
            if game_progress > 0.7:  # 게임 후반
                # 반복 페널티 감소 (게임 종료를 향한 진행 유도)
                repetition_penalty *= 0.7
                # 잡기 보상 증가 (결정적 행동 장려)
                capture_reward *= 2
                print(f"게임 후반부 - 잡기 보상 증가, 반복 페널티 감소")

            # 총 보상 계산 (잡기 + 소극적 페널티 + 반복 페널티)
            total_reward = capture_reward + passive_play_penalty + repetition_penalty

            # 보상으로 TD 학습 업데이트
            self.update_weights(board, next_board, current_player, total_reward)
            # 상대방에게 반대 보상 적용
            self.update_weights(board, next_board,
                                "White" if current_player == "Black" else "Black",
                                -total_reward)

            board = next_board
            current_player = "White" if current_player == "Black" else "Black"
            moves_made += 1

        # 나머지 함수 내용은 변경하지 않음...
        # 게임 결과 처리
        if moves_made >= max_moves:
            print(f"게임이 최대 이동 수({max_moves})에 도달했습니다. 무승부입니다.")
            winner = "Draw"

            # 무승부 페널티 (마블 차이에 따라 다른 페널티 적용)
            black_count = len(board[0])
            white_count = len(board[1])
            marble_diff = abs(black_count - white_count)

            if marble_diff <= 1:  # 거의 대등한 상태
                # 무승부 페널티를 강화하여 경쟁적 플레이 유도
                draw_penalty = -2.0
            else:
                # 마블 수 차이가 있는 경우, 우세한 쪽에는 더 작은 페널티
                draw_penalty = -1.0

                # 소극적 플레이에 대한 추가 페널티
                if no_capture_turns > 10:
                    draw_penalty -= 0.5 * min(1.0, (no_capture_turns - 10) / 10)

                # 이전에 연속 무승부가 있었다면 더 큰 페널티
                if self.consecutive_draws > 2:
                    draw_penalty *= (1.0 + 0.2 * min(5, self.consecutive_draws))

            final_state = game_history[-1][0]

            # 양쪽에 무승부 페널티 적용
            self.update_weights(board, final_state, "Black", draw_penalty)
            self.update_weights(board, final_state, "White", draw_penalty)

            # 연속 무승부 카운터 증가
            self.consecutive_draws += 1
        else:
            # 승/패 보상 처리
            self.consecutive_draws = 0  # 무승부가 아니므로 리셋

            final_state = game_history[-1][0]

            # 기본 승/패 보상
            base_win_reward = 50

            # 승리 스타일에 따른 보상 조정
            if winner == "Black":
                # 빠른 승리에 추가 보상
                moves_bonus = max(0, 20 - (moves_made / (max_moves * 1)))
                # 구슬 우세에 따른 추가 보상
                margin_bonus = min(10.0, white_lost / 2)

                reward_black = base_win_reward + moves_bonus + margin_bonus
                reward_white = -base_win_reward
            else:  # winner == "White"
                moves_bonus = max(0, 20 - (moves_made / (max_moves * 1)))
                margin_bonus = min(10.0, black_lost / 2)

                reward_white = base_win_reward + moves_bonus + margin_bonus
                reward_black = -base_win_reward

            print(f"승자 {winner}에게 보상 {reward_black if winner == 'Black' else reward_white:.2f} 적용")

            self.update_weights(board, final_state, "Black", reward_black)
            self.update_weights(board, final_state, "White", reward_white)

        self.games_played += 1
        self.wins[winner] += 1
        self.stats["total_moves"] += moves_made
        self.stats["avg_moves"] = self.stats["total_moves"] / self.games_played

        # 탐험 확률 감소 - 훈련 단계에 따라 차등 적용
        if self.training_phase == "exploration":
            self.epsilon *= self.epsilon_decay
        elif self.training_phase == "exploitation":
            self.epsilon *= (self.epsilon_decay ** 1.2)  # 빠르게 감소
        elif self.training_phase == "refinement":
            self.epsilon *= (self.epsilon_decay ** 0.8)  # 천천히 감소

        # 최소 탐험 확률 설정
        if self.training_phase == "exploration":
            min_epsilon = 0.06
        elif self.training_phase == "exploitation":
            min_epsilon = 0.035
        else:  # refinement
            min_epsilon = 0.020

        self.epsilon = max(min_epsilon, self.epsilon)

        # 승률 계산 및 베스트 가중치 저장
        if self.games_played % 10 == 0:
            win_ratio = (self.wins["Black"] + self.wins["White"]) / max(1, self.games_played)
            if win_ratio > self.best_win_ratio:
                self.best_win_ratio = win_ratio
                self.best_weights = copy.deepcopy(self.weights)
                print(f"새로운 최고 승률: {win_ratio:.2f}, 가중치 저장")
                self.save_weights("best_weights.json", self.best_weights)

        # 학습 단계 전환
        if self.games_played % 100 == 0:
            if self.training_phase == "exploration" and self.games_played >= 300:
                self.training_phase = "exploitation"
                print("학습 단계 전환: 탐색 → 활용")
                # 일시적 탐험률 증가로 지역 최적해 탈출
                self.epsilon = min(0.4, self.epsilon * 1.5)
            elif self.training_phase == "exploitation" and self.games_played >= 700:
                self.training_phase = "refinement"
                print("학습 단계 전환: 활용 → 정제")
                # 베스트 가중치로 복원
                if self.best_win_ratio > 0:
                    print(f"베스트 가중치 복원 (승률: {self.best_win_ratio:.2f})")
                    self.weights = copy.deepcopy(self.best_weights)

        self.save_game_history(game_history, winner, moves_made)

        return winner


    def train(self, num_games, save_interval=10):
        """
        자가 대국을 통한 학습 수행 - 개선된 전략과 학습 방법 사용

        매개변수:
            num_games: 플레이할 게임 수
            save_interval: 가중치를 저장할 게임 간격
        """
        start_time = time.time()

        # 디렉토리 및 파일 시스템 상태 출력
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        print(f"가중치 저장 경로: {os.path.abspath(self.weights_dir)}")
        print(f"게임 저장 경로: {os.path.abspath(self.games_dir)}")

        # 디렉토리가 존재하고 쓰기 가능한지 확인
        for dir_path in [self.weights_dir, self.games_dir]:
            if os.path.exists(dir_path):
                if os.access(dir_path, os.W_OK):
                    print(f"디렉토리 {dir_path}가 존재하고 쓰기 가능합니다.")
                else:
                    print(f"경고: 디렉토리 {dir_path}가 존재하지만 쓰기 권한이 없습니다!")
            else:
                print(f"경고: 디렉토리 {dir_path}가 존재하지 않습니다!")
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"디렉토리 {dir_path}를 생성했습니다.")
                except Exception as e:
                    print(f"디렉토리 {dir_path} 생성 실패: {e}")

        from IO import BoardIO
        board_dict = copy.deepcopy(BoardIO.BOARD_SAMPLE)
        initial_board = convert_board_format(board_dict)

        # 잘못된 값 재설정을 위한 기본 가중치
        default_weights = copy.deepcopy(AI_WEIGHTS)

        # 전략 순환 주기 설정
        strategy_cycle = ["balanced", "aggressive", "defensive"]

        # 주기적 학습률 및 가중치 재설정 변수
        last_reset = 0
        draw_streak = 0

        # 다양한 학습 상황 시도를 위한 변형 보드 생성
        modified_boards = []

        # 기본 보드 및 변형 보드 준비
        for _ in range(3):
            mod_board = copy.deepcopy(initial_board)
            # 약간의 비대칭성 추가
            if len(mod_board[0]) > 10 and len(mod_board[1]) > 10:
                shift = random.randint(1, 3)
                i = random.randint(0, min(5, len(mod_board[0]) - 1))
                j = random.randint(0, min(5, len(mod_board[1]) - 1))
                mod_board[0][i][0] += shift
                mod_board[1][j][0] -= shift
                # 유효성 검사 (보드 밖으로 나가지 않도록)
                if mod_board[0][i][0] > 9: mod_board[0][i][0] = 9
                if mod_board[1][j][0] < 1: mod_board[1][j][0] = 1
            modified_boards.append(mod_board)

        for game in range(1, num_games + 1):
            game_start = time.time()

            # 학습 진행 상황에 따른 전략 선택
            if game % 10 == 0:  # 주기적으로 다른 전략 시도
                strategy = strategy_cycle[game % 3]
                print(f"게임 {game}: {strategy} 전략으로 학습")
            elif draw_streak >= 5:  # 5판 연속 무승부면 공격적 전략 적용
                strategy = "aggressive"
                print(f"게임 {game}: 연속 무승부 탈출을 위한 공격적 전략")
            else:
                strategy = None  # 기본 전략 (자동 결정)

            # 주기적으로 다른 시작 보드 사용
            if game % 50 == 0 and modified_boards:
                current_board = random.choice(modified_boards)
                print(f"게임 {game}: 변형된 초기 보드로 시작")
            else:
                current_board = initial_board

            # 주기적 가중치 리셋 또는 섭동
            if game - last_reset > 100 and self.consecutive_draws >= 10:
                # 연속 무승부가 많으면 탐색 촉진
                if random.random() < 0.3:  # 30% 확률로 가중치 리셋
                    print(f"게임 {game}: 무승부 회피를 위한 가중치 리셋")
                    # 일부 가중치만 리셋
                    reset_keys = random.sample(list(self.weights.keys()), k=random.randint(1, 3))
                    for key in reset_keys:
                        self.weights[key] = default_weights[key] * (0.5 + random.random())
                    last_reset = game
                else:  # 70% 확률로 일시적 탐험 증가
                    original_epsilon = self.epsilon
                    self.epsilon = min(0.6, self.epsilon * 2.0)
                    print(f"게임 {game}: 무승부 회피를 위한 탐험률 증가 {original_epsilon:.2f} → {self.epsilon:.2f}")

            # 학습 단계별 파라미터 조정
            if self.training_phase == "exploration":
                max_moves = 150  # 탐색 단계에서는 긴 게임 허용
            elif self.training_phase == "exploitation":
                max_moves = 120  # 활용 단계에서는 보통 길이 게임
            else:  # refinement
                max_moves = 100  # 정제 단계에서는 짧은 게임으로 효율성 강화

            # 자가 대국 실행
            winner = self.play_self_game(current_board, max_moves=max_moves, strategy_bias=strategy)

            # 무승부 추적
            if winner == "Draw":
                draw_streak += 1
            else:
                draw_streak = 0

            game_end = time.time()
            game_time = game_end - game_start

            print(f"게임 {game} 완료: {game_time:.1f}초. 승자: {winner}")
            print(f"전적: 흑: {self.wins['Black']}, 백: {self.wins['White']}, 무승부: {self.wins['Draw']}")
            print(f"평균 이동 수: {self.stats['avg_moves']:.1f}")

            if game % save_interval == 0:
                weight_file = f"weights_after_{game}_games.json"
                self.save_weights(weight_file)

                print("현재 가중치:")
                for name, value in self.weights.items():
                    print(f"  {name}: {value:.4f}")

            # 5게임마다 잘못된 가중치 확인
            if game % 5 == 0:
                has_nan = False
                for name, value in self.weights.items():
                    if np.isnan(value) or np.isinf(value):
                        has_nan = True
                        print(f"경고: {name} 가중치가 잘못된 값 {value}를 가집니다. 기본값으로 재설정")
                        self.weights[name] = default_weights.get(name, 0.5)

                if has_nan:
                    print("잘못된 가중치를 기본값으로 재설정했습니다")

            # 주기적으로 무승부 탈출을 위한 학습 촉진
            if self.consecutive_draws >= 15 and game % 20 == 0:
                print(f"연속 {self.consecutive_draws}회 무승부 감지! 학습 촉진 적용")
                # 임시 학습률 증가
                self.learning_rate *= 1.5
                # 일시적으로 공격적인 전략으로 5판 연속 진행
                for i in range(5):
                    boost_winner = self.play_self_game(initial_board, max_moves=80, strategy_bias="aggressive")
                    print(f"학습 촉진 게임 {i + 1}/5: 승자 {boost_winner}")
                # 학습률 원상복구
                self.learning_rate /= 1.5

                # 무승부 탈출에 성공했으면 카운터 리셋
                if boost_winner != "Draw":
                    self.consecutive_draws = 0

        # 최종 가중치 저장
        self.save_weights("final_weights.json")
        # 베스트 가중치 별도 저장
        if self.best_win_ratio > 0:
            self.save_weights("best_weights_final.json", self.best_weights)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n훈련 완료: {num_games}게임, {total_time:.1f}초 소요")
        print(f"최종 통계:")
        print(f"  플레이한 게임: {self.games_played}")
        print(f"  승리: 흑: {self.wins['Black']}, 백: {self.wins['White']}, 무승부: {self.wins['Draw']}")
        print(f"  게임당 평균 이동: {self.stats['avg_moves']:.1f}")
        print(f"  최종 탐험률 (epsilon): {self.epsilon:.4f}")
        print(f"  훈련 단계: {self.training_phase}")
        print(f"  최고 승률: {self.best_win_ratio:.2f}")
        print(f"최종 가중치:")
        for name, value in self.weights.items():
            print(f"  {name}: {value:.4f}")


    def analyze_and_optimize_weights(self, num_test_games=20):
        """
        현재 가중치와 최적 가중치를 비교 분석하고 최적화하는 함수
        """
        print("가중치 분석 및 최적화 시작...")

        # 원래 가중치 저장
        original_weights = copy.deepcopy(self.weights)

        # 1. 각 특성별 민감도 분석
        sensitivity_results = {}

        for feature in self.weights.keys():
            print(f"\n'{feature}' 특성 민감도 분석 중...")

            test_variations = [-0.3, -0.1, 0.1, 0.3]  # 가중치 변화 범위
            variation_results = {}

            for variation in test_variations:
                # 특성 하나만 변화시켜 테스트
                test_weights = copy.deepcopy(original_weights)
                test_weights[feature] += variation

                # 가중치 범위 제한
                if feature == 'marble_diff' and test_weights[feature] < 0.1:
                    test_weights[feature] = 0.1
                if feature == 'sumito' and test_weights[feature] < 0.1:
                    test_weights[feature] = 0.1
                if feature == 'edge_safety' and test_weights[feature] > -0.1:
                    test_weights[feature] = -0.1

                # 원래 가중치를 테스트 가중치로 임시 교체
                self.weights = test_weights

                # 간단히 자가 대국 몇판으로 성능 테스트
                wins_black = wins_white = draws = 0

                for _ in range(num_test_games // 4):  # 빠른 테스트를 위해 짧은 게임
                    result = self.play_self_game(max_moves=100)
                    if result == "Black":
                        wins_black += 1
                    elif result == "White":
                        wins_white += 1
                    else:
                        draws += 1

                win_ratio = (wins_black + wins_white) / max(1, wins_black + wins_white + draws)
                draw_ratio = draws / max(1, wins_black + wins_white + draws)

                variation_results[variation] = {
                    "wins_black": wins_black,
                    "wins_white": wins_white,
                    "draws": draws,
                    "win_ratio": win_ratio,
                    "draw_ratio": draw_ratio
                }

                print(f"  변화량 {variation:+.2f}: 승률 {win_ratio:.2f}, 무승부 비율 {draw_ratio:.2f}")

            # 최적의 변화량 찾기
            best_variation = max(variation_results.items(),
                                 key=lambda x: (x[1]["win_ratio"], -x[1]["draw_ratio"]))

            sensitivity_results[feature] = {
                "variations": variation_results,
                "best_variation": best_variation[0],
                "best_win_ratio": best_variation[1]["win_ratio"]
            }

            print(f"  '{feature}' 최적 변화량: {best_variation[0]:+.2f}, 승률: {best_variation[1]['win_ratio']:.2f}")

        print("\n민감도 분석 결과:")
        for feature, result in sensitivity_results.items():
            print(f"  {feature}: 최적 변화량 {result['best_variation']:+.2f}, 승률 {result['best_win_ratio']:.2f}")

        # 2. 모든 최적 변화량을 동시에 적용
        optimized_weights = copy.deepcopy(original_weights)
        for feature, result in sensitivity_results.items():
            optimized_weights[feature] += result['best_variation'] * 0.7  # 약간 보수적으로 적용

            # 가중치 범위 제한
            if feature == 'marble_diff' and optimized_weights[feature] < 0.1:
                optimized_weights[feature] = 0.1
            if feature == 'sumito' and optimized_weights[feature] < 0.1:
                optimized_weights[feature] = 0.1
            if feature == 'edge_safety' and optimized_weights[feature] > -0.1:
                optimized_weights[feature] = -0.1

        # 3. 최적화된 가중치로 테스트
        print("\n최적화된 가중치 테스트 중...")
        self.weights = optimized_weights
        wins_black = wins_white = draws = 0

        for i in range(num_test_games):
            result = self.play_self_game(max_moves=120)
            if result == "Black":
                wins_black += 1
            elif result == "White":
                wins_white += 1
            else:
                draws += 1

            print(f"  테스트 게임 {i + 1}/{num_test_games}: {result}")

        optimized_win_ratio = (wins_black + wins_white) / max(1, num_test_games)
        optimized_draw_ratio = draws / max(1, num_test_games)

        # 4. 원래 가중치로 테스트
        print("\n원래 가중치 테스트 중...")
        self.weights = original_weights
        orig_wins_black = orig_wins_white = orig_draws = 0

        for i in range(num_test_games):
            result = self.play_self_game(max_moves=120)
            if result == "Black":
                orig_wins_black += 1
            elif result == "White":
                orig_wins_white += 1
            else:
                orig_draws += 1

            print(f"  테스트 게임 {i + 1}/{num_test_games}: {result}")

        original_win_ratio = (orig_wins_black + orig_wins_white) / max(1, num_test_games)
        original_draw_ratio = orig_draws / max(1, num_test_games)

        # 5. 결과 비교 및 저장
        print("\n최적화 결과:")
        print(f"  원래 가중치: 승률 {original_win_ratio:.2f}, 무승부 비율 {original_draw_ratio:.2f}")
        print(f"  최적화 가중치: 승률 {optimized_win_ratio:.2f}, 무승부 비율 {optimized_draw_ratio:.2f}")

        # 원래 가중치로 복구
        self.weights = original_weights

        # 최적화 가중치 저장
        self.save_weights("optimized_weights.json", optimized_weights)

        # 최종 추천 가중치 (원본과 최적화 가중치의 혼합)
        if optimized_win_ratio > original_win_ratio:
            mix_ratio = 0.7  # 최적화 가중치가 더 좋으면 70% 비중으로 적용
            recommended_weights = {}
            for feature in original_weights.keys():
                recommended_weights[feature] = original_weights[feature] * (1 - mix_ratio) + optimized_weights[
                    feature] * mix_ratio

                # 가중치 범위 제한
                if feature == 'marble_diff' and recommended_weights[feature] < 0.1:
                    recommended_weights[feature] = 0.1
                if feature == 'sumito' and recommended_weights[feature] < 0.1:
                    recommended_weights[feature] = 0.1
                if feature == 'edge_safety' and recommended_weights[feature] > -0.1:
                    recommended_weights[feature] = -0.1

            self.save_weights("recommended_weights.json", recommended_weights)

            print("\n최종 추천 가중치:")
            for feature, value in recommended_weights.items():
                print(f"  {feature}: {value:.4f}")

            return recommended_weights
        else:
            print("\n원래 가중치가 더 좋거나 비슷합니다. 원래 가중치 유지.")
            return original_weights

    def save_game_history(self, history, winner, moves):
        """게임 히스토리를 파일로 저장"""
        game_file = os.path.join(self.games_dir, f"game_{self.games_played}_{winner}_{moves}moves.json")

        game_data = {
            "game_number": self.games_played,
            "winner": winner,
            "moves": moves,
            "weights": self.weights,
            "states": []
        }

        for i, (board, player) in enumerate(history):
            black_str = ",".join(f"{pos[0]},{pos[1]}" for pos in board[0])
            white_str = ",".join(f"{pos[0]},{pos[1]}" for pos in board[1])

            state_data = {
                "move_number": i,
                "player": player,
                "black_marbles": black_str,
                "white_marbles": white_str
            }

            game_data["states"].append(state_data)

        try:
            with open(game_file, 'w') as f:
                json.dump(game_data, f, indent=2)
            print(f"게임 히스토리 저장됨: {game_file}")
        except Exception as e:
            print(f"게임 히스토리 저장 오류: {e}")
            print(f"시도한 경로: {game_file}")

    def play_against_agent(self, board_dict=None, player_color="Black", depth=3, time_limit=8.0):
        """
        인간 플레이어가 강화된 에이전트와 대결할 수 있는 인터페이스 - 딕셔너리 형식 이동 처리
        """
        from IO import BoardIO

        if board_dict is None:
            board_dict = copy.deepcopy(BoardIO.BOARD_SAMPLE)

        board = convert_board_format(board_dict)
        current_player = "Black"
        moves_made = 0
        black_lost = white_lost = 0
        recent_board_states = {}  # 반복 상태 방지

        # AI가 사용할 전략 결정
        ai_strategy = None
        if player_color == "Black":
            # AI가 White라면
            ai_strategy = "balanced"  # 기본 전략
        else:
            # AI가 Black이라면
            ai_strategy = "balanced"  # 기본 전략

        while True:
            print("\n현재 보드 상태:")
            for i, color in enumerate(["Black", "White"]):
                pieces = len(board[i])
                print(f"{color}은 {pieces}개 구슬이 남았습니다")

            ai_turn = current_player != player_color

            # 현재 보드 상태 기록
            board_key = board_to_key(board)
            if board_key in recent_board_states:
                recent_board_states[board_key] += 1
            else:
                recent_board_states[board_key] = 1

            if ai_turn:
                print(f"\nAI ({current_player})가 생각 중...")

                # 게임 진행 상황에 따른 AI 전략 조정
                if current_player == "Black" and black_lost > white_lost + 1:
                    # 흑이 지고 있으면 공격적 전략으로 변경
                    ai_strategy = "aggressive"
                elif current_player == "White" and white_lost > black_lost + 1:
                    # 백이 지고 있으면 공격적 전략으로 변경
                    ai_strategy = "aggressive"
                elif moves_made > 100:
                    # 게임이 길어지면 승부를 보기 위해 공격적 전략
                    ai_strategy = "aggressive"
                elif moves_made < 7:
                    # 게임 초반에는 방어적 전략 (인간 플레이어에게 기회 부여)
                    ai_strategy = "defensive"

                # 동적 깊이 조정 - 게임 진행에 따라
                ai_depth = depth
                if (current_player == "Black" and black_lost >= 3) or \
                        (current_player == "White" and white_lost >= 3):
                    ai_depth += 1
                    print(f"AI가 더 깊게 생각합니다 (깊이: {ai_depth})")

                start_time = time.time()
                next_board = self.choose_move(board, current_player, depth=ai_depth,
                                              time_limit=time_limit,
                                              recent_boards=recent_board_states,
                                              strategy_bias=ai_strategy)
                end_time = time.time()

                if next_board is None or next_board == board:
                    print(f"AI ({current_player})의 유효한 이동이 없습니다. 게임 종료.")
                    winner = "White" if current_player == "Black" else "Black"
                    break

                # 이동 문자열 찾기 - 딕셔너리 형식 활용
                color = "BLACK" if current_player == "Black" else "WHITE"
                all_moves_dict = generate_all_next_moves(board, color)

                # 현재 보드와 같은 다음 보드 상태 찾기
                move_str = None
                for move_key, move_board in all_moves_dict.items():
                    if board_to_key(move_board) == board_to_key(next_board):
                        from AI import get_move_string_from_key
                        move_str = get_move_string_from_key(move_key)
                        break

                # 이동 키를 찾지 못한 경우 기본 설명 제공
                if not move_str:
                    # 구슬 개수 변화로 이동 추론
                    if len(next_board[0]) != len(board[0]) or len(next_board[1]) != len(board[1]):
                        if current_player == "Black" and len(next_board[1]) < len(board[1]):
                            move_str = f"백 구슬 {len(board[1]) - len(next_board[1])}개 밀어냄"
                        elif current_player == "White" and len(next_board[0]) < len(board[0]):
                            move_str = f"흑 구슬 {len(board[0]) - len(next_board[0])}개 밀어냄"
                        else:
                            move_str = "구슬 이동"
                    else:
                        move_str = "구슬 이동"

                print(f"AI ({current_player})의 이동: {move_str} ({end_time - start_time:.2f}초)")
            else:
                print(f"\n당신의 차례 ({current_player})")

                while True:
                    try:
                        move_input = input("이동을 입력하세요 (예: 'A1,B1' 또는 'A1A2A3,B1B2B3', 'quit'로 종료): ")

                        if move_input.lower() == 'quit':
                            print("게임을 종료합니다.")
                            return "Quit"

                        source_coords, dest_coords = parse_move_input(move_input)

                        current_color = "BLACK" if current_player == "Black" else "WHITE"
                        is_valid, reason = move_validation(source_coords, dest_coords, board, current_color)

                        if not is_valid:
                            print(f"유효하지 않은 이동: {reason}")
                            continue

                        new_board, _ = move_marbles(source_coords, dest_coords, board, current_color)

                        if new_board is None:
                            print("이동 실행 오류. 다시 시도하세요.")
                            continue

                        next_board = new_board
                        break
                    except Exception as e:
                        print(f"오류: {e}. 다시 시도하세요.")

            prev_black_count = len(board[0])
            prev_white_count = len(board[1])
            next_black_count = len(next_board[0])
            next_white_count = len(next_board[1])

            if next_black_count < prev_black_count:
                black_lost += (prev_black_count - next_black_count)
                print(f"흑이 {prev_black_count - next_black_count}개 구슬을 잃었습니다. 총: {black_lost}")

            if next_white_count < prev_white_count:
                white_lost += (prev_white_count - next_white_count)
                print(f"백이 {prev_white_count - next_white_count}개 구슬을 잃었습니다. 총: {white_lost}")

            if black_lost >= 6 or white_lost >= 6:
                winner = "White" if black_lost >= 6 else "Black"
                print(f"게임 종료! {winner}가 6개 구슬을 밀어내 승리했습니다.")
                break

            board = next_board
            current_player = "White" if current_player == "Black" else "Black"
            moves_made += 1

            if moves_made >= 200:  # 최대 150턴
                print("게임이 최대 이동 수(150)에 도달했습니다. 무승부입니다.")
                return "Draw"

        return winner


if __name__ == "__main__":
    # TD 학습 에이전트 인스턴스 생성
    agent = AbaloneAgent()

    # 자가 대국 횟수 (필요에 따라 조정)
    num_games = 170

    # 각 게임에 시간 제한 설정
    import time

    max_game_time = 360  # 각 게임당 최대 300초(5분) 제한

    for i in range(num_games):
        print(f"게임 {i + 1} 시작합니다...")

        # 게임 시작 시간 기록
        game_start_time = time.time()

        try:
            # 시간 제한 설정 - 게임당 최대 시간과 최대 이동 수 모두 제한
            agent.play_self_game(max_moves=150)  # 이동 수를 줄여 실행 시간 단축

            # 게임이 너무 오래 걸리면 강제 종료
            if time.time() - game_start_time > max_game_time:
                print(f"게임 {i + 1}이 시간 제한({max_game_time}초)을 초과하여 강제 종료합니다.")
        except Exception as e:
            print(f"게임 {i + 1}에서 오류 발생: {e}")
            print("다음 게임으로 넘어갑니다.")

        # 탐험 확률(epsilon) 감소 (탐험 → 활용 전환)
        agent.epsilon *= agent.epsilon_decay
        print(f"게임 {i + 1} 종료. 업데이트된 epsilon: {agent.epsilon:.4f}")

        # 진행상황 저장 - 주기적으로 가중치 저장
        if (i + 1) % 10 == 0:
            agent.save_weights(f"td_learning_progress_{i + 1}_games.json")
            print(f"중간 진행상황 저장: {i + 1}/{num_games} 게임 완료")

    # 학습 후 최종 가중치를 파일에 저장
    agent.save_weights("td_learning_weights.json")

    # 베스트 가중치도 따로 저장
    if agent.best_win_ratio > 0:
        agent.save_weights("best_weights_final.json", agent.best_weights)
        print(f"학습 완료! 최고 승률 {agent.best_win_ratio:.2f}의 가중치를 best_weights_final.json 파일에 저장하였습니다.")

    print("학습 완료! 최종 가중치를 td_learning_weights.json 파일에 저장하였습니다.")