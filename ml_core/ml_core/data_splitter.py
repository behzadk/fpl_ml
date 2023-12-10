from sklearn.model_selection import train_test_split


def randomly_split_data(df, test_frac):
    if test_frac == 1.0:
        return df, df[0:0]

    elif test_frac == 0.0:
        return df[0:0], df

    df_a, df_b = train_test_split(df, test_size=test_frac, shuffle=True)

    return df_a, df_b
