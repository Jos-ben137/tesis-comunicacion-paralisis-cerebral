#!/usr/bin/env python3
"""
clasificador.py — Entrenamiento SVM + MLP y generación de métricas
Uso    : python clasificador.py --csv features.csv --output modelo_svm.pkl
Salida : modelo_svm.pkl + resultados/ (tabla, confusión, ROC, importancia)

Autores: José Antonio Benavides Ortega — FIME UdC
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings, time, json

warnings.filterwarnings("ignore")

GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD   = "\033[1m"
CLASES = ["alegria", "neutral", "dolor"]
PALETTE = {"alegria": "#a6e3a1", "neutral": "#89b4fa", "dolor": "#f38ba8"}


# ── Carga de datos ─────────────────────────────────────────────────────────────
def cargar_datos(csv_path: Path):
    assert csv_path.exists(), f"No encontré {csv_path}"
    df = pd.read_csv(csv_path)
    print(f"{CYAN}Dataset: {len(df):,} muestras | {df.shape[1]-1} features{RESET}")

    feature_cols = [c for c in df.columns if c != "label"]
    assert len(feature_cols) == 68, \
        f"Se esperan 68 features, encontré {len(feature_cols)}"

    X = df[feature_cols].values.astype(np.float32)
    y_str = df["label"].str.lower().values

    le = LabelEncoder()
    le.classes_ = np.array(CLASES)
    y = le.transform(y_str)

    for cls in CLASES:
        n = (y_str == cls).sum()
        print(f"  {cls}: {n:,}")
    return X, y, feature_cols, le


# ── SVM ───────────────────────────────────────────────────────────────────────
def entrenar_svm(X, y, cv):
    print(f"\n{BOLD}{'─'*55}")
    print(f"  SVM — kernel RBF — GridSearchCV k={cv.n_splits}")
    print(f"{'─'*55}{RESET}")

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(kernel="rbf", probability=True, random_state=42))])
    grid = {"svm__C":     [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.001, 0.01, 0.1]}

    print("  GridSearch: 4×4 = 16 combinaciones × 5 folds = 80 fits")
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro",
                      n_jobs=-1, verbose=0)
    gs.fit(X, y)
    best = gs.best_estimator_
    print(f"  Mejores params: {gs.best_params_}")
    print(f"  Mejor F1-macro (CV): {gs.best_score_:.4f}")
    print(f"  Tiempo: {time.time()-t0:.1f}s")

    y_pred = cross_val_predict(best, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(best, X, y, cv=cv, method="predict_proba")
    best.fit(X, y)
    return best, y_pred, y_prob, gs.best_params_


# ── MLP ───────────────────────────────────────────────────────────────────────
def entrenar_mlp(X, y, cv):
    print(f"\n{BOLD}{'─'*55}")
    print(f"  MLP — 68→256→128→3 — Adam — early stopping")
    print(f"{'─'*55}{RESET}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=1e-3,
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15,
            random_state=42, verbose=False,
        )),
    ])
    t0 = time.time()
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    pipe.fit(X, y)
    print(f"  Tiempo: {time.time()-t0:.1f}s")
    return pipe, y_pred, y_prob


# ── Métricas ──────────────────────────────────────────────────────────────────
def metricas(y, y_pred, y_prob, nombre, le) -> dict:
    cls_names = list(le.classes_)
    report = classification_report(y, y_pred, target_names=cls_names,
                                   output_dict=True, zero_division=0)
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob, multi_class="ovr",
                            average="macro", labels=[0,1,2])
    except Exception:
        auc = float("nan")

    print(f"\n  {GREEN}{nombre}{RESET}")
    print(f"  {'Clase':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'n':>7}")
    print(f"  {'─'*45}")
    for cls in cls_names:
        r = report[cls]
        print(f"  {cls:<12} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1-score']:>8.4f} {int(r['support']):>7,}")
    print(f"  {'─'*45}")
    print(f"  Accuracy: {acc:.4f}  |  AUC-ROC macro: {auc:.4f}")

    return {"modelo": nombre, "accuracy": round(acc,4),
            "precision_macro": round(report["macro avg"]["precision"],4),
            "recall_macro":    round(report["macro avg"]["recall"],4),
            "f1_macro":        round(report["macro avg"]["f1-score"],4),
            "auc_roc":         round(auc,4) if not np.isnan(auc) else "N/A",
            **{f"f1_{cls}": round(report[cls]["f1-score"],4) for cls in cls_names}}


# ── Figuras ───────────────────────────────────────────────────────────────────
def plot_confusion(y, y_pred, le, nombre, path):
    cls_names = list(le.classes_)
    cm      = confusion_matrix(y, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [PALETTE[c] for c in cls_names]
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=cls_names, yticklabels=cls_names,
                linewidths=0.5, ax=ax)

    for i in range(len(cls_names)):
        for j in range(len(cls_names)):
            ax.text(j+0.5, i+0.38, f"{cm_norm[i,j]:.1%}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm_norm[i,j] > 0.5 else "#1e1e2e")
            ax.text(j+0.5, i+0.65, f"n={cm[i,j]}",
                    ha="center", va="center", fontsize=8,
                    color="#6c7086")

    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(f"Matriz de Confusión — {nombre}", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {GREEN}✓ {path.name}{RESET}")


def plot_roc(y, prob_svm, prob_mlp, le, path):
    from sklearn.preprocessing import label_binarize
    cls_names = list(le.classes_)
    y_bin = label_binarize(y, classes=[0,1,2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, (cls, ax) in enumerate(zip(cls_names, axes)):
        for prob, lbl, ls in [(prob_svm,"SVM","-"),(prob_mlp,"MLP","--")]:
            fpr, tpr, _ = roc_curve(y_bin[:,i], prob[:,i])
            auc = roc_auc_score(y_bin[:,i], prob[:,i])
            color = PALETTE[cls] if lbl == "SVM" else "#9399b2"
            ax.plot(fpr, tpr, ls, label=f"{lbl} (AUC={auc:.3f})",
                    linewidth=2, color=color)
        ax.plot([0,1],[0,1],":",color="#585b70",linewidth=1)
        ax.set_title(f"{cls.capitalize()}", fontsize=11)
        ax.set_xlabel("FPR", fontsize=9)
        if i == 0: ax.set_ylabel("TPR", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Curvas ROC — One-vs-Rest (k=5 CV)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {GREEN}✓ {path.name}{RESET}")


def plot_importancia(model, X, y, feature_names, path):
    print("  Calculando importancia por permutación...")
    res = permutation_importance(model, X, y, n_repeats=8,
                                 scoring="f1_macro", random_state=42,
                                 n_jobs=-1)
    imp = res.importances_mean
    std = res.importances_std
    order = np.argsort(imp)[::-1][:20]   # top 20

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_names[i] for i in order[::-1]],
            imp[order[::-1]], xerr=std[order[::-1]],
            color="#89b4fa", ecolor="#6c7086", capsize=3, height=0.6)
    ax.set_xlabel("Importancia media (F1-macro drop)", fontsize=10)
    ax.set_title("Top-20 Features por Permutación — SVM", fontsize=11)
    ax.axvline(0, color="#585b70", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {GREEN}✓ {path.name}{RESET}")


def latex_tabla(resultados, path):
    cls_cols = " & ".join([f"\\textbf{{F1-{c.capitalize()}}}" for c in CLASES])
    tex = (
        "% Generado por clasificador.py — pegar en §7.3.2\n"
        "\\begin{table}[H]\n"
        "  \\centering\n"
        "  \\caption{Comparativa de clasificadores ligeros — SVM vs MLP}\n"
        "  \\label{tab:resultados_clasificadores}\n"
        "  \\begin{tabular}{lccccc}\n"
        "    \\toprule\n"
        f"    \\textbf{{Modelo}} & \\textbf{{Accuracy}} & \\textbf{{F1-macro}} "
        f"& \\textbf{{AUC-ROC}} & {cls_cols} \\\\\n"
        "    \\midrule\n"
    )
    for r in resultados:
        cls_f1 = " & ".join([str(r.get(f"f1_{c}", "—")) for c in CLASES])
        tex += (f"    {r['modelo']} & {r['accuracy']} & {r['f1_macro']} "
                f"& {r['auc_roc']} & {cls_f1} \\\\\n")
    tex += (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\begin{tablenotes}\n"
        "    \\small\n"
        "    \\item Validación cruzada estratificada k=5. "
        "Métricas macro-averaged sobre conjunto de prueba.\n"
        "  \\end{tablenotes}\n"
        "\\end{table}\n"
    )
    path.write_text(tex)
    print(f"  {GREEN}✓ {path.name}{RESET}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clasificador FER — SVM + MLP")
    parser.add_argument("--csv",    required=True, help="features.csv de entrada")
    parser.add_argument("--output", default="modelo_svm.pkl",
                        help="Archivo .pkl para guardar el SVM entrenado")
    args = parser.parse_args()

    csv_path    = Path(args.csv)
    output_pkl  = Path(args.output)
    out_dir     = csv_path.parent / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{CYAN}{'='*55}")
    print(f"  Clasificador FER — SVM + MLP")
    print(f"{'='*55}{RESET}\n")

    # Datos
    X, y, feature_names, le = cargar_datos(csv_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Modelos
    svm_model, svm_pred, svm_prob, svm_params = entrenar_svm(X, y, cv)
    mlp_model, mlp_pred, mlp_prob             = entrenar_mlp(X, y, cv)

    # Métricas
    res_svm = metricas(y, svm_pred, svm_prob, "SVM", le)
    res_mlp = metricas(y, mlp_pred, mlp_prob, "MLP", le)

    # Guardar modelo SVM
    with open(output_pkl, "wb") as f:
        pickle.dump({"model": svm_model, "le": le, "params": svm_params}, f)
    print(f"\n  {GREEN}✓ Modelo serializado: {output_pkl}{RESET}")

    # Figuras
    print(f"\n{BOLD}  Generando figuras...{RESET}")
    plot_confusion(y, svm_pred, le, "SVM", out_dir / "matriz_confusion_svm.png")
    plot_confusion(y, mlp_pred, le, "MLP", out_dir / "matriz_confusion_mlp.png")
    plot_roc(y, svm_prob, mlp_prob, le,    out_dir / "roc_curves.png")
    plot_importancia(svm_model, X, y, feature_names,
                     out_dir / "feature_importance.png")

    # Tablas
    print(f"\n{BOLD}  Exportando tablas...{RESET}")
    resultados = [res_svm, res_mlp]
    pd.DataFrame(resultados).to_csv(out_dir / "tabla_resultados.csv", index=False)
    print(f"  {GREEN}✓ tabla_resultados.csv{RESET}")
    latex_tabla(resultados, out_dir / "tabla_resultados.tex")
    (out_dir / "resultados.json").write_text(
        json.dumps({"modelos": resultados, "params_svm": svm_params}, indent=2))
    print(f"  {GREEN}✓ resultados.json{RESET}")

    # Resumen
    print(f"\n{GREEN}{BOLD}{'='*55}")
    print(f"  ✓ Listo — resultados en: {out_dir}")
    print(f"{'='*55}{RESET}")
    print(f"\n  Para el documento (§7.3.2):")
    print(f"    ① tabla_resultados.tex  → reemplaza [TABLA DE RESULTADOS]")
    print(f"    ② matriz_confusion_svm.png → reemplaza [IMAGEN: matriz_confusion]")
    print(f"    ③ roc_curves.png        → figura adicional")
    print()


if __name__ == "__main__":
    main()
