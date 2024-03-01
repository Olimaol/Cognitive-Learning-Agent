from CompNeuroPy import run_script_parallel

if __name__ == "__main__":
    args_list = [[f"{sim_id}", "0"] for sim_id in range(1, 61)]
    run_script_parallel(
        script_path="run_cla_Training.py",
        n_jobs=4,
        args_list=args_list,
    )