def add_noise(df, n_noise):
    for i in range(n_noise):
        df[f'noise_{i}'] = np.random.normal(-2,2,len(df))