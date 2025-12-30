import torch
import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Импортируем ядро симулятора
# Важно: Neuralbiocore_U_for_GPU.py должен быть в той же папке
try:
    from Neuralbiocore_U_for_GPU import ConsciousnessSimulator, PhysicsConfig, BioChemistry
except ImportError:
    raise ImportError("Не найден файл Neuralbiocore_U_for_GPU.py. Убедитесь, что он находится рядом.")

# ==========================================
# 1. ТОКЕНИЗАТОР (Русский язык)
# ==========================================

class SimpleRussianTokenizer:
    """
    Превращает слова в индексы и обратно.
    """
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # Специальные токены:
        # PAD - заполнение, UNK - неизвестное, BOS - начало, EOS - конец, SILENCE - тишина
        self.specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SILENCE>"]
        for i, token in enumerate(self.specials):
            self.token2id[token] = i
            self.id2token[i] = token
            
        self.vocab_size = len(self.specials)
        
        # Базовый словарь (Философско-когнитивный набор)
        initial_vocab = [
            "я", "ты", "мы", "он", "мир", "сознание", "боль", "радость", 
            "вижу", "чувствую", "думаю", "есть", "нет", "быть", "свет", 
            "тьма", "время", "поиск", "смысл", "жизнь", "сон", "пробуждение",
            "красный", "синий", "форма", "объект", "субъект", "действовать",
            "любовь", "страх", "понимаю", "не", "и", "но", "где", "зачем",
            "человек", "разум", "сигнал", "вселенная", "пустота",
            "число", "машина", "код", "система", "энергия"
        ]
        self.add_tokens(initial_vocab)

    def add_tokens(self, tokens: List[str]):
        for token in tokens:
            token = token.lower().strip()
            if token not in self.token2id:
                self.token2id[token] = self.vocab_size
                self.id2token[self.vocab_size] = token
                self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        # Простая токенизация по пробелам
        tokens = text.lower().replace('.', ' .').replace(',', ' ,').replace('?', ' ?').split()
        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]
        # Оборачиваем в BOS (Begin) и EOS (End)
        return [self.token2id["<BOS>"]] + ids + [self.token2id["<EOS>"]]

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            if i in self.id2token:
                t = self.id2token[i]
                if t not in self.specials:
                    tokens.append(t)
        return " ".join(tokens)

# ==========================================
# 2. ИНТЕРФЕЙС ("Уши и Рот")
# ==========================================

class NeuroLinguisticInterface:
    """
    Связывает дискретные символы с непрерывной нейронной активностью.
    """
    def __init__(self, simulator: ConsciousnessSimulator, tokenizer: SimpleRussianTokenizer, embedding_dim: int = 64):
        self.sim = simulator
        self.tokenizer = tokenizer
        
        # Определяем устройство (CPU/GPU) по тому, где живет первый слой мозга
        self.device = simulator.hierarchy.levels[0].layer.V.device
        
        # Размер сенсорного слоя V1 (обычно 1000 нейронов)
        self.sensor_dim = simulator.hierarchy.levels[0].layer.N 
        self.emb_dim = embedding_dim
        
        # Эмбеддинги (Словарь векторов)
        self.embeddings = torch.nn.Embedding(tokenizer.vocab_size, embedding_dim).to(self.device)
        
        # Проекция на V1 (W_sensory)
        # Коэффициент 0.015 подобран экспериментально, чтобы не перегружать нейроны
        self.W_sensor_projection = torch.randn(self.sensor_dim, embedding_dim).to(self.device) * 0.015
        
    def token_to_sensor_input(self, token_id: int) -> torch.Tensor:
        """
        УШИ: Превращает ID токена в электрический ток для сенсоров.
        """
        with torch.no_grad():
            token_tensor = torch.tensor([token_id], device=self.device)
            emb = self.embeddings(token_tensor).squeeze(0)
            
            # Input = W * Embedding
            sensor_signal = torch.mv(self.W_sensor_projection, emb)
            
            # Добавляем стохастичность (шум восприятия)
            sensor_signal += torch.randn_like(sensor_signal) * 0.005
            
            return sensor_signal

    def neural_state_to_logits(self, neural_prediction_mu: torch.Tensor) -> torch.Tensor:
        """
        РОТ: Превращает предсказание мозга (V1 mu) обратно в вероятности слов.
        Использует транспонированную матрицу проекции (Inverse Model).
        """
        # Emb_pred = W.T * Neural_State
        # [FIX] Cast to match projection matrix dtype (float32 vs float64)
        neural_prediction_mu = neural_prediction_mu.to(self.W_sensor_projection.dtype)
        
        emb_pred = torch.mv(self.W_sensor_projection.T, neural_prediction_mu)
        
        # Logits = Embeddings * Emb_pred (поиск ближайшего вектора слова)
        logits = torch.mv(self.embeddings.weight, emb_pred)
        return logits

# ==========================================
# 3. АГЕНТ (Управление обучением)
# ==========================================

class TextAgent:
    def __init__(self):
        self.tokenizer = SimpleRussianTokenizer()
        
        # Инициализируем мозг с топологией Small-World (лучше для ассоциаций)
        print("Инициализация нейросети...")
        self.brain = ConsciousnessSimulator(use_small_world=True)
        
        self.interface = NeuroLinguisticInterface(self.brain, self.tokenizer)
        
        # Время предъявления одного слова (в секундах)
        # 50 мс достаточно для первичной обработки, но не перегружает
        self.token_duration = 0.050 
        self.dt = self.brain.phys_cfg.dt
        
        # Порог паники: если Свободная Энергия выше, сбрасываем активность
        self.panic_threshold = 4_500_000.0 

        # Список понятий с негативной окраской (Semantic Aversion)
        self.negative_concepts = {
            "боль", "страх", "плохо", "тьма", "враг", "ненавидеть", 
            "смерть", "ужас", "опасно", "разрушать", "ломать"
        }
        
        print(f"Агент готов. Device: {self.interface.device}")

    def force_calm(self, severity: float = 1.0):
        """
        Принудительный гомеостаз v2.0 (Fix Sparse Update).
        """
        with torch.no_grad():
            # 1. Химия: Снижаем нейромодуляторы
            self.brain.chemistry.dopamine.mul_(0.5 * (1.0 - severity))
            self.brain.chemistry.norepinephrine.mul_(0.5 * (1.0 - severity))
            
            # 2. Электрика: Гасим активность
            for unit in self.brain.hierarchy.levels:
                unit.layer.I_ext.zero_()
                unit.layer.V.mul_(0.1) 
                unit.prediction_error.zero_()
                
                # === ЛЕЧЕНИЕ ЭПИЛЕПСИИ ===
                if severity > 0.5:
                    # Усиливаем терапию: 20% забывания вместо 5%
                    decay = 0.8 
                    
                    # Функция для лечения одного синапса
                    def heal_synapse(synapse):
                        if synapse is None: return
                        
                        # 1. Ослабляем веса
                        if hasattr(synapse, 'W_dense'):
                            synapse.W_dense.mul_(decay)
                            
                            # 2. CLAMP: Жесткое ограничение максимального веса
                            # Срезаем все, что выросло выше 1.5 (предохранитель)
                            synapse.W_dense.clamp_(max=1.5)
                            
                            # 3. ВАЖНО: Синхронизируем Sparse матрицу, если она используется
                            if hasattr(synapse, 'is_sparse') and synapse.is_sparse:
                                synapse.W_sparse = synapse.W_dense.to_sparse_csr()
                        else:
                            # Если старая версия без dense/sparse разделения
                            synapse.W.mul_(decay)
                            synapse.W.clamp_(max=1.5)

                    # Применяем к входам (bottom-up) и ожиданиям (top-down)
                    heal_synapse(unit.synapse_bottom_up)
                    heal_synapse(unit.synapse_top_down)
            
            # 3. Сброс GWT
            self.brain.gwt.broadcast_signal.zero_()
            self.brain.gwt.active_coalitions = []
            
            if severity > 0.5:
                print("   [Homeostasis] ВЕСА СБРОШЕНЫ И СИНХРОНИЗИРОВАНЫ (Sparse Sync).")

    def apply_negative_reinforcement(self, severity: float = 1.0):
        """
        Принудительное ОТРИЦАТЕЛЬНОЕ ПОДКРЕПЛЕНИЕ (Anti-Hebbian).
        Наказывает синапсы, которые были активны в момент ошибки/боли.
        """
        print(f"   ☠️ NEGATIVE REINFORCEMENT (Severity: {severity:.2f})")
        
        with torch.no_grad():
            # 1. Химия СТРЕССА
            # Дофамин в ноль (нет награды), Норадреналин в максимум (паника/запоминание негатива)
            self.brain.chemistry.dopamine.mul_(0.0)
            self.brain.chemistry.norepinephrine.fill_(0.8 + 0.2 * severity)
            
            # 2. Anti-Hebbian Learning для всех уровней
            for level_idx, unit in enumerate(self.brain.hierarchy.levels):
                
                # Функция наказания синапса
                def punish_synapse(synapse):
                    if synapse is None: return
                    
                    # Получаем следы активности (Eligibility Traces)
                    # trace_pre: [N_pre], trace_post: [N_post]
                    # Co-activity ~ OuterProduct(post, pre)
                    
                    # Пытаемся достать следы. Если их нет, используем текущие спайки как proxy (менее точно)
                    # В Neuralbiocore_U_for_GPU.py это trace_pre и trace_post
                    if not hasattr(synapse, 'trace_pre') or not hasattr(synapse, 'trace_post'):
                        return

                    # Вычисляем матрицу совпадений (кто виноват?)
                    # [N_post, 1] * [1, N_pre] -> [N_post, N_pre]
                    eligibility = torch.ger(synapse.trace_post, synapse.trace_pre)
                    
                    # Наказываем: W = W - severity * learning_rate * eligibility
                    punishment_strength = 0.5 * severity # Сила наказания
                    
                    # Применяем к Dense матрице
                    if hasattr(synapse, 'W_dense'):
                        # W -= punishment
                        synapse.W_dense.sub_(eligibility * punishment_strength)
                        synapse.W_dense.clamp_(min=0.0) # Не даем весам стать отрицательными
                        
                        # Sync Sparse
                        if hasattr(synapse, 'is_sparse') and synapse.is_sparse:
                            synapse.W_dense.masked_fill_(~synapse.mask, 0.0) # Mask restore
                            synapse.W_sparse = synapse.W_dense.to_sparse_csr()
                    elif hasattr(synapse, 'W'):
                        synapse.W.sub_(eligibility * punishment_strength)
                        synapse.W.clamp_(min=0.0)

                # Наказываем связи
                punish_synapse(unit.synapse_bottom_up)
                punish_synapse(unit.synapse_top_down)
                
                # Сбрасываем активность нейронов (GABA Flush)
                unit.layer.V.fill_(self.brain.phys_cfg.v_rest)
                unit.layer.is_dead.fill_(False) # Воскрешаем, если умерли от шока
                unit.layer.ATP.fill_(0.5) # Даем энергию на восстановление

    def listen_and_learn(self, text: str, epochs: int = 1) -> bool:
        """
        Возвращает True, если обучение прошло успешно.
        Возвращает False, если случилась паника.
        """
        token_ids = self.tokenizer.encode(text)
        steps_per_token = int(self.token_duration / self.dt)
        
        print(f"\n--- Обучение фразе: '{text}' ---")
        
        # Легкая настройка перед фразой
        self.force_calm(severity=0.5)
        
        for epoch in range(epochs):
            total_free_energy = 0.0
            panic_mode = False
            
            for t, token_id in enumerate(token_ids):
                if panic_mode: break
                
                sensory_signal = self.interface.token_to_sensor_input(token_id)
                self.brain.body.sensory_input = sensory_signal
                
                current_token_fe = 0.0
                for _ in range(steps_per_token):
                    self.brain.step(self.dt)
                    fe = self.brain.hierarchy.get_global_free_energy()
                    current_token_fe += fe
                
                avg_token_fe = current_token_fe / steps_per_token
                total_free_energy += avg_token_fe
                
                # Check for Semantic Aversion (Негативная окраска слова)
                token_str = self.tokenizer.id2token.get(token_id, "")
                if token_str in self.negative_concepts:
                    print(f"   ⚠️ Semantic Aversion: '{token_str}' detected.")
                    # Используем МЯГКОЕ отрицательное подкрепление (Low Severity)
                    # Чтобы не травмировать сеть, а лишь сформировать отвращение
                    self.apply_negative_reinforcement(severity=0.2) 

                # Проверка порога
                if avg_token_fe > self.panic_threshold:
                    token_str = self.tokenizer.id2token.get(token_id, "???")
                    print(f"   ! ШОК на слове '{token_str}': FE={avg_token_fe:.0f}")
                    panic_mode = True
                    break

            if panic_mode:
                print(f"   !!! ВКЛЮЧЕНИЕ ПРОТОКОЛА ЗАЩИТЫ.")
                # self.force_calm(severity=1.0) 
                self.apply_negative_reinforcement(severity=1.5) # NEW WAY: Punish active pathways
                return False # <--- Вернули False при ошибке
            
            # Логгирование (только если не паника)
            avg_epoch_fe = total_free_energy / len(token_ids)
            print(f"Epoch {epoch+1} | Avg FE: {avg_epoch_fe:.0f} | Phi: {self.brain.gwt.phi_current:.2f}")
            
            # Пауза
            silence = self.interface.token_to_sensor_input(self.tokenizer.token2id["<SILENCE>"])
            self.brain.body.sensory_input = silence
            for _ in range(30): self.brain.step(self.dt)
            
        return True # <--- ВАЖНО! Эта строка должна быть ЗДЕСЬ, вне цикла for

    def generate_text(self, prompt: str, max_length: int = 8, temperature: float = 0.8) -> str:
        """
        Генерация через Активный Вывод (Active Inference).
        """
        print(f"\n--- Генерация (Prompt: '{prompt}') ---")
        
        # Легкое успокоение перед речью
        self.force_calm(severity=0.1)
        
        prompt_ids = self.tokenizer.encode(prompt)
        steps_per_token = int(self.token_duration / self.dt)
        
        # 1. Priming: Загружаем контекст (слушаем промпт)
        for token_id in prompt_ids:
            if token_id == self.tokenizer.token2id["<EOS>"]: continue
            sensory_signal = self.interface.token_to_sensor_input(token_id)
            self.brain.body.sensory_input = sensory_signal
            for _ in range(steps_per_token):
                self.brain.step(self.dt)

        # 2. Generation Loop: Говорим
        generated_ids = []
        # Начинаем с последнего слова промпта
        current_input_id = prompt_ids[-2] if len(prompt_ids) > 1 else prompt_ids[0]
        
        # Список последних токенов для избегания повторов
        history_window = [current_input_id]
        
        for _ in range(max_length):
            # Эхо-сигнал (Self-feedback)
            # Вместо тишины подаем слабое "эхо" последнего слова.
            # Это имитирует слуховую петлю обратной связи (мы слышим, что говорим).
            echo_signal = self.interface.token_to_sensor_input(current_input_id)
            self.brain.body.sensory_input = echo_signal * 0.2
            
            # Даем мозгу подумать (Generative process)
            for _ in range(steps_per_token):
                self.brain.step(self.dt)
            
            # Читаем прогноз V1 (mu) - чего мозг ожидает услышать дальше?
            v1_prediction = self.brain.hierarchy.levels[0].mu
            
            # Превращаем прогноз в слова
            logits = self.interface.neural_state_to_logits(v1_prediction)
            
            # === REPETITION PENALTY (Наказание за повторы) ===
            # Временное подавление уже сказанных слов
            for past_token in history_window[-3:]: # Смотрим на 3 последних слова
                logits[past_token] -= 5.0 # Сильно понижаем вероятность
            
            # Блокируем технические токены при генерации
            for bad_token in ["<UNK>", "<PAD>", "<BOS>"]:
                logits[self.tokenizer.token2id[bad_token]] = -float('inf')
            
            # Сэмплируем
            probs = torch.softmax(logits / temperature, dim=0)
            
            # Защита от NaN (если веса совсем упали)
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / len(probs)
            
            next_token_id = torch.multinomial(probs, 1).item()
            
            if next_token_id == self.tokenizer.token2id["<EOS>"]:
                break
                
            generated_ids.append(next_token_id)
            current_input_id = next_token_id
            history_window.append(next_token_id)

        return self.tokenizer.decode(generated_ids)

    def save_brain(self, filename="brain_dump.pt"):
        """Сохраняет полное состояние агента (Веса + Химия + Токенизатор)"""
        print(f"\n💾 Сохранение памяти в '{filename}'...")
        
        # 1. Собираем веса иерархии
        hierarchy_state = []
        for level in self.brain.hierarchy.levels:
            # Функция для безопасного извлечения весов
            def get_w(syn):
                if syn is None: return None
                if hasattr(syn, 'W_dense'): return syn.W_dense.cpu()
                if hasattr(syn, 'W'): return syn.W.cpu()
                return None

            level_data = {
                'W_bu': get_w(level.synapse_bottom_up),
                'W_td': get_w(level.synapse_top_down)
            }
            hierarchy_state.append(level_data)
            
        # 2. Собираем химию
        chemistry_state = {
            'dopamine': self.brain.chemistry.dopamine.cpu(),
            'norepinephrine': self.brain.chemistry.norepinephrine.cpu()
        }
        
        # 3. Собираем словарь (Interface)
        # Нам нужно сохранить эмбеддинги и сам словарь слов
        interface_state = {
            'embeddings': self.interface.embeddings.state_dict(),
            'projection': self.interface.W_sensor_projection.cpu(),
            'token2id': self.tokenizer.token2id,
            'id2token': self.tokenizer.id2token,
            'vocab_size': self.tokenizer.vocab_size
        }
        
        # Сохраняем все в один файл
        torch.save({
            'hierarchy': hierarchy_state,
            'chemistry': chemistry_state,
            'interface': interface_state
        }, filename)
        print("✅ Память успешно сохранена.")

    def load_brain(self, filename="brain_dump.pt"):
        """Загружает состояние агента"""
        import os
        if not os.path.exists(filename):
            print(f"⚠️ Файл сохранения '{filename}' не найден. Начинаем с нуля.")
            return

        print(f"\n📂 Загрузка памяти из '{filename}'...")
        checkpoint = torch.load(filename, map_location=self.interface.device)
        
        # 1. Восстанавливаем словарь
        saved_interface = checkpoint['interface']
        self.tokenizer.token2id = saved_interface['token2id']
        self.tokenizer.id2token = saved_interface['id2token']
        self.tokenizer.vocab_size = saved_interface['vocab_size']
        
        # Пересоздаем слой эмбеддингов под новый размер
        self.interface.emb_dim = saved_interface['embeddings']['weight'].shape[1]
        self.interface.embeddings = torch.nn.Embedding(self.tokenizer.vocab_size, self.interface.emb_dim).to(self.interface.device)
        self.interface.embeddings.load_state_dict(saved_interface['embeddings'])
        self.interface.W_sensor_projection = saved_interface['projection'].to(self.interface.device)
        
        # 2. Восстанавливаем веса иерархии
        for i, level_data in enumerate(checkpoint['hierarchy']):
            if i >= len(self.brain.hierarchy.levels): break
            
            level = self.brain.hierarchy.levels[i]
            
            # Helper function for safe loading
            def load_w(syn, data):
                if syn is None or data is None: return
                
                device = self.interface.device
                if hasattr(syn, 'W_dense'):
                    syn.W_dense.data = data.to(device)
                    # Sync sparse if needed
                    if getattr(syn, 'is_sparse', False):
                        syn.W_sparse = syn.W_dense.to_sparse_csr()
                elif hasattr(syn, 'W'):
                    syn.W.data = data.to(device)

            # Bottom-Up
            load_w(level.synapse_bottom_up, level_data.get('W_bu'))
            
            # Top-Down
            load_w(level.synapse_top_down, level_data.get('W_td'))

        # 3. Восстанавливаем химию
        self.brain.chemistry.dopamine.data = checkpoint['chemistry']['dopamine'].to(self.interface.device)
        self.brain.chemistry.norepinephrine.data = checkpoint['chemistry']['norepinephrine'].to(self.interface.device)
        
        print(f"✅ Агент 'вспомнил' прошлую жизнь. Vocab: {self.tokenizer.vocab_size}")

# ==========================================
# 4. ЗАПУСК ДЕМО
# ==========================================

def run_education_session():
    # Создаем агента
    agent = TextAgent()
    
    # Корпус текстов для обучения
    dataset = [
        "Я мыслю сознание",
        "Боль есть сигнал",
        "Радость свет жизнь",
        "Я чувствую время",
        "Мы видим мир",
        "Сознание есть поиск",
        "Свет и тьма",
        "Я есть субъект"
    ]
    
    start_time = time.time()
    
    # Обучение
    # ИЗМЕНЕНИЕ 1: Увеличиваем количество эпох
    # Дадим ему "зубрить" материал подольше.
    print("Начинаем углубленное обучение...")
    for i, phrase in enumerate(dataset):
        agent.listen_and_learn(phrase, epochs=6) # Было 2, ставим 6
        
    print(f"\nОбучение завершено за {time.time() - start_time:.2f} сек.")
    
    # Тест
    print("\n=== ТЕСТ ГЕНЕРАЦИИ (Диалог) ===")
    prompts = ["Я", "Сознание", "Боль", "Свет"]
    
    # ИЗМЕНЕНИЕ 2: Снижаем температуру при генерации
    # Это сделает его более "сосредоточенным" и уберет бред типа "синий".
    for p in prompts:
        # Temperature 0.5 (было 0.7-0.8). 
        # Чем ниже, тем строже логика.
        response = agent.generate_text(p, temperature=0.5)
        print(f"User: {p}")
        print(f"Agent: {response}")

if __name__ == "__main__":
    run_education_session()