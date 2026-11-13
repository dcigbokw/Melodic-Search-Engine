import math
import random

class DFSSolver:
    def __init__(self, matrix_1st, matrix_2nd, matrix_3rd, validator_fn, tonic_pc=0):
        self.matrix_1st = matrix_1st
        self.matrix_2nd = matrix_2nd
        self.matrix_3rd = matrix_3rd
        self.is_valid_transition = validator_fn
        self.tonic_pc = tonic_pc
        self.fallback_options = [c for c in matrix_1st.keys() if isinstance(c, tuple) and c[0] != -1]
        
    def solve(self, start_sequence, num_to_add, strictness=1.0, markov_order=2, 
              temperature=1.0, max_drafts=1, seed=None, range_expansion=0, cadence_pref="none"):
        
        successful_drafts = []
        stats = {
            "nodes_entered": 0, 
            "candidate_evaluations": 0, 
            "rule_rejections": 0, 
            "out_of_range": 0, 
            "cadence_violations": 0
        }
        RANGES = {0: (60, 81), 1: (53, 74), 2: (48, 69), 3: (40, 62)}
        
        def dfs(path, visited):
            if len(successful_drafts) >= max_drafts: return
            
            stats["nodes_entered"] += 1
            current_depth = len(path) - len(start_sequence)
            
            if current_depth == num_to_add:
                successful_drafts.append(path.copy())
                return
                
            opts_dict = {}
            target_order = markov_order
            
            while target_order >= 1 and not opts_dict:
                if target_order == 3 and len(path) >= 3:
                    opts_dict = self.matrix_3rd.get((path[-3], path[-2], path[-1]), {})
                elif target_order == 2 and len(path) >= 2:
                    opts_dict = self.matrix_2nd.get((path[-2], path[-1]), {})
                elif target_order == 1:
                    opts_dict = self.matrix_1st.get(path[-1], {})
                target_order -= 1
                
            valid_keys = [k for k in opts_dict.keys() if isinstance(k, tuple) and k[0] != -1]
            
            # Use deterministic state-derived RNG for benchmarking, or standard RNG for API
            local_rng = random.Random(hash(tuple(path)) ^ seed) if seed is not None else random
            
            temp = max(0.1, temperature)
            opts = sorted(valid_keys, key=lambda k: math.log(local_rng.random() + 1e-9) / (opts_dict.get(k, 0.1) ** (1.0 / temp)), reverse=True)
            
            if len(opts) < 5 and self.fallback_options:
                opts.extend(local_rng.sample(self.fallback_options, min(20, len(self.fallback_options))))
                
            for cand in opts:
                stats["candidate_evaluations"] += 1
                cand_state = cand[-1] if isinstance(cand, tuple) and isinstance(cand[0], tuple) else cand
                
                h = hash(tuple(path + [cand_state]))
                if h in visited: continue
                visited.add(h)
                
                out_of_range = False
                if cand_state[0] != -1:
                    for v in range(4):
                        min_pitch, max_pitch = RANGES[v]
                        if not (min_pitch - range_expansion <= cand_state[v] <= max_pitch + range_expansion):
                            out_of_range = True
                            break
                if out_of_range: 
                    stats["out_of_range"] += 1
                    continue

                cadence_failed = False
                if cand_state[0] != -1 and cadence_pref != "none":
                    cand_index_in_ext = len(path) - len(start_sequence)
                    is_penultimate = (cand_index_in_ext == num_to_add - 2)
                    is_ultimate = (cand_index_in_ext == num_to_add - 1)
                    bass_pc = cand_state[3] % 12
                    
                    if cadence_pref == "authentic":
                        if is_penultimate and bass_pc != (self.tonic_pc + 7) % 12: cadence_failed = True 
                        if is_ultimate and bass_pc != self.tonic_pc: cadence_failed = True 
                    elif cadence_pref == "half":
                        if is_ultimate and bass_pc != (self.tonic_pc + 7) % 12: cadence_failed = True 
                    elif cadence_pref == "plagal":
                        if is_penultimate and bass_pc != (self.tonic_pc + 5) % 12: cadence_failed = True 
                        if is_ultimate and bass_pc != self.tonic_pc: cadence_failed = True 

                if cadence_failed:
                    stats["cadence_violations"] += 1
                    continue

                passes_rules = self.is_valid_transition(path[-1], cand_state, self.tonic_pc)
                if passes_rules or local_rng.random() > strictness:
                    path.append(cand_state)
                    dfs(path, visited)
                    path.pop()
                else:
                    stats["rule_rejections"] += 1

        dfs(start_sequence.copy(), set())
        return successful_drafts, stats