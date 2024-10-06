    # Fixed parameters
    temperature = 0.1
    alpha = 1

    # Variable parameters.
    generator = [(1,5000),(1,6000),(1,7000)] + [(10,beta) for beta in [1000, 2000, 4000, 8000, 16000]]
