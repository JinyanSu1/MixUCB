## Training noisy bandit settings.

import generate_toy_data
import run_linucb
import run_perfect_expert
import plot_quick_linucb

def main():
    noise_settings = [0.1, 0.2, 0.5]
    for i,setting in enumerate(noise_settings):
        # Generate datasets
        args = generate_toy_data.parser.parse_args(['--T', '1000', \
        '--noise_std', str(setting), '--seed', '42', \
        '--output_file', f'simulation_data_toy20241009_noise{setting}.pkl'])
        generate_toy_data.main(args)
        # Run LinUCB on those datasets (with default settings)
        alpha = 0.1
        lambda_ = 0.001
        args = run_linucb.parser.parse_args(['--T', '1000', \
        '--lambda', str(lambda_), '--alpha', str(alpha), \
        '--pickle_file', f'simulation_data_toy20241009_noise{setting}.pkl',
        "--setting_id", f"{1+i}"])
        run_linucb.main(args)
        # Run PerfectExpert on those datasets
        args = run_perfect_expert.parser.parse_args(['--T', '1000', \
        '--pickle_file', f'simulation_data_toy20241009_noise{setting}.pkl', \
        "--setting_id", f"{1+i}"])
        run_perfect_expert.main(args)
        # Finally, plot the results
        args = plot_quick_linucb.parser.parse_args(["--setting_id", f"{1+i}"])
        plot_quick_linucb.main(args)

if __name__=="__main__":
    main()
