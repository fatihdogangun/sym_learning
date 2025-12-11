
import os
import sys
import subprocess
import argparse
import time
import multiprocessing
import gc

from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

import environment
import utils
from learn_rules import tensor_to_pddl_problem
from models import load_ckpt
from parse import determinize_domain, extract_actions



THRESHOLD = 0.05
NUM_DOMAINS = 200  


def setup_paths(model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downward_path = os.path.join(script_dir, "..", "downward", "fast-downward.py")
    domain_path = os.path.join(script_dir, "..", "save", model_name, "domain.pddl")
    prob_domain_path = os.path.join(script_dir, "..", "save", model_name, "domain_prob.pddl")
    return {
        "downward": os.path.abspath(downward_path),
        "domain": os.path.abspath(domain_path),
        "prob_domain": os.path.abspath(prob_domain_path),
    }


def run_planner(domain_path, problem_path, downward_path, timeout=60):

    try:
        result = subprocess.run(
            ["python", downward_path, domain_path, problem_path, "--search", "astar(ff())"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
        sas_plan_path = "sas_plan"
        if os.path.exists(sas_plan_path):
            with open(sas_plan_path, "r") as f:
                plan_lines = f.readlines()
            plan = [line.strip()[1:-1] for line in plan_lines if line.startswith("(")]
            os.remove(sas_plan_path)
            return plan
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def parse_action(action_str, n_objects):

    parts = action_str.split()
    action_name = parts[0]
    obj_args = [int(p.replace("obj", "")) for p in parts[1:]]
    
    name_parts = action_name.split("_")
    from_dy = int(name_parts[1])
    to_dy = int(name_parts[3])
    
    from_obj = obj_args[0]
    to_obj = obj_args[1] if len(obj_args) > 1 else obj_args[0]
    
    return [from_obj, to_obj, 0, from_dy, 0, to_dy, 1, 1]


def execute_plan(env, plan, save_images=False, image_dir=None):

    images = []
    n_objects = len(env.obj_dict)
    
    for i, action_str in enumerate(plan):
        action = parse_action(action_str, n_objects)
        if save_images:
            state, effect, imgs = env.step(*action, get_images=True)
            images.extend(imgs)
            if image_dir:
                for j, img in enumerate(imgs):
                    Image.fromarray(img).save(os.path.join(image_dir, f"step_{i}_{j}.png"))
        else:
            state, effect = env.step(*action)
    
    return env.state(), images


def generate_problem(env, model, n_actions, seed):

    np.random.seed(seed)
    env.reset_objects()
    
    initial_state = torch.tensor(env.state(), dtype=torch.float)
    
  
    for _ in range(n_actions):
        action = env.full_random_action()
        env.step(*action)
    
    goal_state = torch.tensor(env.state(), dtype=torch.float)
    
 
    z_init, r_init, z_goal, r_goal = utils.state_to_problem(initial_state, goal_state, model)
    
    n_obj = initial_state.shape[0]
    graph = np.ones((n_obj, n_obj), dtype=int)
    
    problem_pddl = tensor_to_pddl_problem(
        z_init.cpu().numpy(), r_init.cpu().numpy(),
        z_goal.cpu().numpy(), r_goal.cpu().numpy(),
        graph, graph
    )
    
    return initial_state, goal_state, problem_pddl


def check_success(final_state, goal_state, threshold=0.05):
 
    errors = np.linalg.norm(final_state[:, :3] - goal_state[:, :3], axis=1)
    return np.all(errors < threshold)


def run_single_test(test_idx, n_actions, model_name, paths, pregenerated_domains, 
                    base_seed, num_objects, out_dir, save_images, result_queue):

    seed = base_seed + test_idx
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_ckpt(model_name, tag="best")
    model.to(device)
    model.freeze()
    
    env = environment.BlocksWorld_v4(gui=0, min_objects=num_objects, max_objects=num_objects)
    
 
    np.random.seed(seed)
    initial_state, goal_state, problem_pddl = generate_problem(env, model, n_actions, seed)
    
 
    test_dir = os.path.join(out_dir, f"n{n_actions}", f"test_{test_idx}")
    os.makedirs(test_dir, exist_ok=True)
    problem_path = os.path.join(test_dir, "problem.pddl")
    with open(problem_path, "w") as f:
        f.write(problem_pddl)
    
    env.reset_objects()
    np.random.seed(seed)
    env.reset_objects()
    
    results = {
        "deterministic": {"success": 0, "fail": 0, "no_plan": 0},
        "probabilistic": {"success": 0, "fail": 0, "no_plan": 0},
    }
    

    det_plan = run_planner(paths["domain"], problem_path, paths["downward"])
    
    if det_plan is None:
        results["deterministic"]["no_plan"] = 1
    else:
       
        np.random.seed(seed)
        env.reset_objects()
        
        img_dir = os.path.join(test_dir, "det_exec") if save_images else None
        if img_dir:
            os.makedirs(img_dir, exist_ok=True)
        
        final_state, _ = execute_plan(env, det_plan, save_images, img_dir)
        
        if check_success(final_state, goal_state.numpy(), THRESHOLD):
            results["deterministic"]["success"] = 1
        else:
            results["deterministic"]["fail"] = 1
    

    prob_success = False
    best_plan = None
    
    for domain_idx, domain_path in enumerate(pregenerated_domains):
        plan = run_planner(domain_path, problem_path, paths["downward"], timeout=30)
        
        if plan is not None:
         
            np.random.seed(seed)
            env.reset_objects()
            
            final_state, _ = execute_plan(env, plan, save_images=False)
            
            if check_success(final_state, goal_state.numpy(), THRESHOLD):
                prob_success = True
                best_plan = plan
                break
    
    if prob_success:
        results["probabilistic"]["success"] = 1
        if save_images:
            img_dir = os.path.join(test_dir, "prob_exec")
            os.makedirs(img_dir, exist_ok=True)
            np.random.seed(seed)
            env.reset_objects()
            execute_plan(env, best_plan, save_images=True, image_dir=img_dir)
    elif best_plan is None:
        results["probabilistic"]["no_plan"] = 1
    else:
        results["probabilistic"]["fail"] = 1
    
 
    del env
    gc.collect()
    
    result_queue.put(results)


def save_results(stats, out_path, num_tests, max_actions):
 
    with open(out_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Planning Success Rates\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'Actions':<10} {'Det Success':<15} {'Prob Success':<15}\n")
        f.write("-" * 40 + "\n")
        
        for n in range(1, max_actions + 1):
            det_success = stats[n]["deterministic"]["success"] / num_tests * 100
            prob_success = stats[n]["probabilistic"]["success"] / num_tests * 100
            f.write(f"{n:<10} {det_success:<15.1f} {prob_success:<15.1f}\n")
    
    print(f"Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser("Evaluate planning performance.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Model name")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("-o", "--objects", type=int, default=3, help="Number of objects")
    parser.add_argument("-p", "--processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("-t", "--tests", type=int, default=100, help="Number of tests per action count")
    parser.add_argument("-k", "--max_actions", type=int, default=5, help="Maximum action count")
    parser.add_argument("--save_images", action="store_true", help="Save execution images")
    args = parser.parse_args()
    

    paths = setup_paths(args.name)
    out_dir = os.path.join("../results", args.name, f"seed_{args.seed}", f"{args.objects}obj")
    os.makedirs(out_dir, exist_ok=True)
    
  
    if not os.path.exists(paths["domain"]):
        print(f"Error: Domain file not found: {paths['domain']}")
        print("Run learn_rules.py first to generate PDDL domains.")
        sys.exit(1)
    
    if not os.path.exists(paths["downward"]):
        print(f"Error: Fast Downward not found: {paths['downward']}")
        print("Please install Fast Downward planner.")
        sys.exit(1)
    
    print(f"Pre-generating {NUM_DOMAINS} determinized domains...")
    with open(paths["prob_domain"], "r") as f:
        prob_template = f.read().replace(":probabilistic-effects", "")
        action_blocks = extract_actions(prob_template)
    
    pregen_dir = os.path.join(out_dir, "determinized_domains")
    os.makedirs(pregen_dir, exist_ok=True)
    pregenerated_domains = []
    
    pregenerated_domains.append(paths["domain"])
    
    for i in range(NUM_DOMAINS - 1):
        det_content = determinize_domain(action_blocks, prob_template, seed=i * 11)
        det_path = os.path.join(pregen_dir, f"domain_{i+1}.pddl")
        with open(det_path, "w") as f:
            f.write(det_content)
        pregenerated_domains.append(det_path)
    
    print("Domain generation complete.")
    
    overall_stats = {
        n: {
            "deterministic": {"success": 0, "fail": 0, "no_plan": 0},
            "probabilistic": {"success": 0, "fail": 0, "no_plan": 0},
        }
        for n in range(1, args.max_actions + 1)
    }
    
    for n_actions in range(1, args.max_actions + 1):
        print(f"\nRunning {args.tests} tests for k={n_actions} actions...")
        
        tasks = list(range(args.tests))
        results = []
        active_processes = {}
        
        with tqdm(total=args.tests, desc=f"k={n_actions}") as pbar:
            while tasks or active_processes:
             
                while tasks and len(active_processes) < args.processes:
                    test_idx = tasks.pop(0)
                    result_queue = multiprocessing.Queue()
                    
                    p = multiprocessing.Process(
                        target=run_single_test,
                        args=(test_idx, n_actions, args.name, paths, pregenerated_domains,
                              args.seed, args.objects, out_dir, args.save_images, result_queue)
                    )
                    p.start()
                    active_processes[p] = (test_idx, result_queue, time.time())
                
              
                for p in list(active_processes):
                    test_idx, result_queue, start_time = active_processes[p]
                    p.join(timeout=0.1)
                    
                    if not p.is_alive():
                        try:
                            result = result_queue.get_nowait()
                            results.append(result)
                        except:
                            pass
                        del active_processes[p]
                        pbar.update(1)
                    elif time.time() - start_time > 300:  
                        print(f"\nTest {test_idx} timed out.")
                        p.terminate()
                        p.join()
                        del active_processes[p]
                        pbar.update(1)
                
                time.sleep(0.1)
        

        for result in results:
            for planner in ["deterministic", "probabilistic"]:
                for key in ["success", "fail", "no_plan"]:
                    overall_stats[n_actions][planner][key] += result[planner][key]
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\n{'Actions':<10} {'Det Success %':<15} {'Prob Success %':<15}")
    print("-" * 40)
    
    for n in range(1, args.max_actions + 1):
        det_success = overall_stats[n]["deterministic"]["success"] / args.tests * 100
        prob_success = overall_stats[n]["probabilistic"]["success"] / args.tests * 100
        print(f"{n:<10} {det_success:<15.1f} {prob_success:<15.1f}")
    
    results_path = os.path.join(out_dir, "results.txt")
    save_results(overall_stats, results_path, args.tests, args.max_actions)
    
    subprocess.run(["pkill", "-f", "downward"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
