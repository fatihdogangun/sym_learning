import time
import os
import argparse
import subprocess
import multiprocessing as mp

import torch


def collect(script, num, t, folder, idx, n_min=None, n_max=None):

    cmd = ["python", script, "-N", num, "-T", t, "-o", folder, "-i", idx]
    if n_min is not None:
        cmd += ["-n_min", str(n_min)]
    if n_max is not None:
        cmd += ["-n_max", str(n_max)]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect interaction data in parallel.")
    parser.add_argument("-s", help="Collection script path", type=str, required=True)
    parser.add_argument("-d", help="Data output folder", type=str, required=True)
    parser.add_argument("-N", help="Number of samples per process", type=int, required=True)
    parser.add_argument("-T", help="Interactions per episode", type=int, required=True)
    parser.add_argument("-p", help="Number of parallel processes", type=int, required=True)
    parser.add_argument("-n_min", help="Minimum number of objects", type=int, default=2)
    parser.add_argument("-n_max", help="Maximum number of objects", type=int, default=4)
    args = parser.parse_args()

    args.d = os.path.join("..", args.d)
    if not os.path.exists(args.d):
        os.makedirs(args.d)

    
    procs = []
    start = time.time()
    for i in range(args.p):
        p = mp.get_context("spawn").Process(
            target=collect, 
            args=[args.s, str(args.N), str(args.T), args.d, str(i), args.n_min, args.n_max]
        )
        p.start()
        procs.append(p)

 
    for i in range(args.p):
        procs[i].join()
    
    end = time.time()
    elapsed = end - start
    
    print(f"Collected {args.p * args.N} samples in {elapsed:.2f} seconds. "
          f"Rate: {args.p * args.N / elapsed:.1f} samples/sec")
    

    print("Merging data from all processes...")
    keys = ["action", "effect", "mask", "state", "post_state"]
    for key in keys:
        field = []
        for i in range(args.p):
            field.append(torch.load(os.path.join(args.d, f"{key}_{i}.pt")))
        field = torch.cat(field, dim=0)
        torch.save(field, os.path.join(args.d, f"{key}.pt"))
        
 
        for i in range(args.p):
            os.remove(os.path.join(args.d, f"{key}_{i}.pt"))
    
    print("Done.")
