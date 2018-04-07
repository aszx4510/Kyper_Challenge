import pandas as pd


def convert_nominal(data_path):
    df = pd.read_csv(data_path)
    print(df.columns.values)
    # ['Sex' ' Length' ' Diameter' ' Height' ' Whole weight' ' Shucked weight' ' Viscera weight' ' Shell weight' ' Rings']

    sex = df['Sex']
    one_hot = pd.get_dummies(sex)  # convert to one-hot encoding

    result = pd.concat([one_hot, df.drop(['Sex'], axis=1)], axis=1)
    return result


if __name__ == '__main__':
    convert_nominal(data_path='data/data.txt')
