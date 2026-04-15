# ==========================================
# PRACTICA APRENDIZAJE SUPERVISADO
# Clasificación MNIST
# Modelos: SVM, KNN, Random Forest, Gradient Boosting
# ==========================================

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# ==========================================
# CARGAR DATASET
# ==========================================

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

print("Forma de los datos:", X.shape)
print("Forma de etiquetas:", y.shape)


# ==========================================
# MOSTRAR IMAGEN DE EJEMPLO
# ==========================================

imagen = X[6000].reshape(28,28)
etiqueta = y[6000]

print("Etiqueta de la imagen:", etiqueta)

plt.imshow(imagen, cmap="gray")
plt.show()


# ==========================================
# DIVISION TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=5
)


# ==========================================
# REDUCCION DEL DATASET (PARA RAPIDEZ)
# ==========================================

X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]


# ==========================================
# FUNCION PARA EVALUAR MODELOS
# ==========================================

def evaluar(nombre, y_real, pred):

    print("\nModelo:", nombre)

    print("Accuracy:", accuracy_score(y_real, pred))

    print("Precision:", precision_score(y_real, pred, average="weighted"))

    print("Recall:", recall_score(y_real, pred, average="weighted"))

    print("F1-score:", f1_score(y_real, pred, average="weighted"))


# ==========================================
# MODELO 1: SVM
# ==========================================

print("\nEntrenando SVM...")

svm = LinearSVC(max_iter=3000)

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("Resultados SVM")

print(classification_report(y_test, y_pred_svm))

evaluar("SVM", y_test, y_pred_svm)


# ==========================================
# MODELO 2: KNN
# ==========================================

print("\nEntrenando KNN...")

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("Resultados KNN")

print(classification_report(y_test, y_pred_knn))

evaluar("KNN", y_test, y_pred_knn)


# ==========================================
# MODELO 3: RANDOM FOREST
# ==========================================

print("\nEntrenando Random Forest...")

rf = RandomForestClassifier(n_estimators=50)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Resultados Random Forest")

print(classification_report(y_test, y_pred_rf))

evaluar("Random Forest", y_test, y_pred_rf)


# ==========================================
# MODELO 4: GRADIENT BOOSTING
# ==========================================

print("\nEntrenando Gradient Boosting...")

gb = GradientBoostingClassifier(n_estimators=30)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

print("Resultados Gradient Boosting")

print(classification_report(y_test, y_pred_gb))

evaluar("Gradient Boosting", y_test, y_pred_gb)


# ==========================================
# MATRIZ DE CONFUSION DEL SVM
# ==========================================

print("\nMostrando matriz de confusión SVM...")

cfn_mat = confusion_matrix(y_test, y_pred_svm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cfn_mat,
    display_labels=svm.classes_
)

disp.plot()

plt.show()


# ==========================================
# ENTRENAMIENTO FINAL DEL MEJOR MODELO
# (SVM normalmente es el mejor en MNIST)
# ==========================================

print("\nEntrenando modelo final con TODO el dataset...")

modelo_final = LinearSVC(max_iter=3000)

modelo_final.fit(X, y)

print("Modelo final entrenado correctamente")