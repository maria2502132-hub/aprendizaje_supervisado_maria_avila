from train import y_test 
import os
import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_models():
    print("=" * 60)
    print("Iniciando el entrenamiento del pipeline del ML")
    print("=" * 60)

    data_path= 'data/titanic_clean.csv'
    if not os.path.exits(data_path):
        raise FileNotFoundError(f"No s encontro el archivo en {data_path}.")
    
    print(f"Cargado datos desde: {data_path}...")
    df=pd.read_csv(data_path)
    
    x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df[['Survived']]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print(f"registro de entrenamiento: {len(x_train)}")
    print(f"registro de test: {len(x_test)}")
    print("=" * 50)
    
    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_cols = ['Sex', 'Embarked']

    print(f"definiendo el pipeline...")
    print(f">>> numerico (standardsacaler): {numerical_cols}")
    print(f">>> categorico (onehotencoder): {categorical_cols}")
    print("=" * 50)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6))
    ])

    pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipeline_rf.fit(x_train, y_train)
    y_pred_rf = pipeline_rf.predict(x_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"Reporte de clasificacion (random forest):")
    print(classification_report(y_test, y_pred_rf))
    print("=" * 50)

    print("Entrenando logistic regression...")
    pipeline_lr.fit(x_train, y_train)
    y_pred_lr = pipeline_lr.predict(x_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    print(f"Logistic regression Accuracy: {acc_lr:.4f}")
    print(f"Reporte de clasificacion (logistic regression):")
    print(classification_report(y_test, y_pred_lr))
    print("=" * 50)

    print("Guardando los mejores modelos...")
    joblib.dump(pipeline_rf, 'models/pipeline_rf_titanic.pkl')
    joblib.dump(pipeline_lr, 'models/pipeline_lr_titanic.pkl')
    print("Modelos guardados exitosamente")
    print("=" * 50)
    print("Proceso completado exitosamente")

if __name__ == '__main__':
    train_models