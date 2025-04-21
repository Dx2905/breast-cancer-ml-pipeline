
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path):
    headers = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
               'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
               'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
               'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
               'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
               'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
               'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    df = pd.read_csv(csv_path, header=None)
    df.columns = headers

    X = df.iloc[:, 2:].values
    Y = df.iloc[:, 1].values
    Y = LabelEncoder().fit_transform(Y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, df.columns[2:].tolist()
