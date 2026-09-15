"""
Feature engineering — Base de Datos de Créditos
================================================

Pipeline construido a partir de la plantilla de referencia, adaptado a las
variables y decisiones tomadas durante el EDA (transformacion_eda.ipynb),
en su estado final (después de eliminar 'puntaje' y 'saldo_principal'):

- 'puntaje' se elimina por fuga de información confirmada (100% de los
  registros con el valor de relleno 95.227787 pagan a tiempo).
- 'saldo_principal' se elimina por alta redundancia con 'saldo_total'
  (correlación de 0.913 confirmada en el EDA, sección 3.3). Se conserva
  'saldo_total' por ser la medida más completa del saldo del crédito.
- Solo se conserva 'relacion_cuota_salario' como atributo derivado de
  capacidad de pago (las otras dos razones se descartaron por VIF > 100).
- 'edad_cliente', 'salario_cliente', 'total_otros_prestamos',
  'puntaje_datacredito' y 'saldo_total' llevan las mismas reglas de
  negocio ya validadas en el EDA (sección 2.5): corrección de edad
  (+100 de digitación), corrección de escala (/1000) en salario y otros
  préstamos, corrección de puntaje_datacredito fuera del rango oficial
  150-950 (el valor 0 se conserva, es el sentinela de "sin score"; los
  demás valores fuera de rango se tratan como error de captura), y
  winsorización del percentil superior de saldo_total.
- 'saldo_mora' y 'saldo_mora_codeudor' NO se eliminan automáticamente: son
  zero-inflated, no necesariamente altos en nulos, pero sí candidatos a
  fuga de información temporal (son saldos medidos en el momento del
  corte de datos, es decir, cuando el resultado ya podría conocerse).
  Se deja un chequeo explícito (ver `revisar_leakage_temporal`) para
  confirmarlo con los datos reales antes de decidir, tal como se hizo
  con 'puntaje'.
- Los nulos remanentes tras el EDA (`puntaje_datacredito`, `saldo_mora`,
  `saldo_total`, `saldo_mora_codeudor`, `promedio_ingresos_datacredito`,
  `tendencia_ingresos`, y el `relacion_cuota_salario` derivado) se
  imputan explícitamente en este pipeline (ver `Imputacion`), ya que el
  EDA los dejó documentados pero sin imputar (esa decisión se tomó
  deliberadamente para esta etapa de feature engineering).
- `tendencia_ingresos` llega MEZCLADA en los datos crudos: algunas filas
  traen la etiqueta ('Creciente'/'Decreciente'/'Estable') y otras el
  valor numérico sin categorizar. Se replica aquí la misma recodificación
  del EDA (ver `RecodificarTendenciaIngresos`) — esta clase no existía en
  la versión anterior del script y es indispensable: sin ella, decenas de
  categorías numéricas espurias llegarían al `OrdinalEncoder`.

Nota de compatibilidad: el pipeline puede recibir tanto el CSV crudo
original (fechas 'd/m/Y H:M', decimales con coma) como el dataset ya
limpio exportado del EDA (`BD_creditos_modelo.csv`/`.pkl`) — en ambos
casos produce el mismo resultado, ya que `CorreccionTipos` y
`RecodificarTendenciaIngresos` no alteran datos que ya están correctos.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# 1. Transformadores
# ---------------------------------------------------------------------------

class CorreccionTipos(BaseEstimator, TransformerMixin):
    """Replica la sección 2.4 del EDA: asegura que cada columna tenga el
    tipo de dato correcto antes de que el resto del pipeline opere sobre
    ella. Es el primer paso del pipeline porque `ReglasNegocio` e
    `Imputacion` necesitan columnas numéricas/fecha ya bien tipadas
    (comparaciones, medianas, etc. no funcionan sobre texto).

    Se aplica siempre con `errors='coerce'`, así que si el pipeline se usa
    sobre un dataset ya limpio (por ejemplo, el `BD_creditos_modelo.pkl`
    exportado desde el EDA, que ya viene con los tipos correctos), este
    paso no daña nada: simplemente confirma los tipos ya correctos.
    """

    COLS_FECHA = ("fecha_prestamo",)
    COLS_NUMERICAS = (
        "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
        "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
        "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
        "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
        "creditos_sectorCooperativo", "creditos_sectorReal",
        "promedio_ingresos_datacredito",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.COLS_FECHA:
            if col in X.columns:
                # Formato exacto usado en el EDA (d/m/Y H:M). Sin especificarlo,
                # pandas infiere un único formato para toda la columna y descarta
                # como NaT cualquier fila que no encaje (p. ej. día > 12).
                X[col] = pd.to_datetime(X[col], format="%d/%m/%Y %H:%M", errors="coerce")

        for col in self.COLS_NUMERICAS:
            if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
                # No usar `dtype == object`: pandas puede leer texto como dtype
                # 'str' (StringDtype) en vez de 'object', y esa comparación no
                # lo detecta, dejando las comas decimales sin convertir.
                X[col] = X[col].astype(str).str.replace(",", ".", regex=False)
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X


class RecodificarTendenciaIngresos(BaseEstimator, TransformerMixin):
    """Replica la recodificación de 'tendencia_ingresos' hecha en el EDA
    (sección 2.4): la columna cruda viene MEZCLADA — algunas filas ya
    traen la etiqueta ('Creciente'/'Decreciente'/'Estable') y otras traen
    el valor numérico del delta de ingresos, nunca categorizado:
        valor numérico < 0  -> 'Decreciente'
        valor numérico == 0 -> 'Estable'
        valor numérico > 0  -> 'Creciente'
        ya viene como texto  -> se deja igual

    Debe ejecutarse ANTES de `ToCategory` / `Imputacion` (la moda) y antes
    de `ToDF` (el OrdinalEncoder): si se casteara a category o se
    codificara con los valores numéricos aún sueltos, aparecerían decenas
    de categorías espurias en vez de las 3 esperadas.
    """

    ETIQUETAS_VALIDAS = {"Creciente", "Decreciente", "Estable"}

    def fit(self, X, y=None):
        return self

    def _recodificar_valor(self, valor):
        if pd.isna(valor):
            return np.nan
        texto = str(valor).strip()
        if texto in self.ETIQUETAS_VALIDAS:
            return texto
        numero = pd.to_numeric(texto.replace(",", "."), errors="coerce")
        if pd.isna(numero):
            return np.nan
        if numero < 0:
            return "Decreciente"
        elif numero == 0:
            return "Estable"
        else:
            return "Creciente"

    def transform(self, X):
        X = X.copy()
        if "tendencia_ingresos" in X.columns:
            X["tendencia_ingresos"] = X["tendencia_ingresos"].apply(self._recodificar_valor)
        return X


class EliminarDuplicados(BaseEstimator, TransformerMixin):
    """Elimina registros duplicados. Se usa dos veces en el pipeline base:
    al inicio (duplicados originales) y después de descartar columnas
    irrelevantes (al quitar columnas pueden aparecer registros idénticos
    que antes no lo eran)."""

    def __init__(self, subset=None):
        self.subset = subset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates(subset=self.subset).copy()


class ColumnasNulos(BaseEstimator, TransformerMixin):
    """Elimina columnas cuyo porcentaje de nulos supera un umbral.

    Se calcula el umbral SOLO con los datos de entrenamiento (fit) y se
    aplica igual en transform. Con el dataset final del EDA, ninguna
    columna supera el 40% (la más alta es 'tendencia_ingresos' con ~27%),
    así que este paso no elimina nada por defecto; se deja como
    salvaguarda ante datos nuevos con peor calidad.
    """

    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        pct_nulos = X.isnull().mean()
        self.cols_to_drop_ = pct_nulos[pct_nulos > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class ReglasNegocio(BaseEstimator, TransformerMixin):
    """Aplica las reglas de validación / corrección de valores fuera de
    rango ya documentadas y confirmadas en el EDA (sección 2.5):

    - edad_cliente >= 120 -> error de digitación (+100), se corrige restando 100
    - salario_cliente / total_otros_prestamos > umbral_escala -> error de
      escala (dígitos de más / unidad distinta), se corrige dividiendo /1000
    - puntaje_datacredito fuera del rango oficial [150, 950]: el 0 se
      conserva (sentinela de "sin score / sin información"); los demás
      valores fuera de rango (negativos, o entre 0 y 150 exclusive) se
      tratan como error de captura y se marcan como NaN, para ser
      imputados más adelante en `Imputacion`
    - saldo_total: winsorización al percentil superior, para moderar la
      cola extrema sin descartar los registros
    """

    def __init__(self, umbral_escala=50_000_000, percentil_winsor=0.995,
                 rango_puntaje_datacredito=(150, 950)):
        self.umbral_escala = umbral_escala
        self.percentil_winsor = percentil_winsor
        self.rango_puntaje_datacredito = rango_puntaje_datacredito
        self.limite_saldo_total_ = None

    def fit(self, X, y=None):
        X_temp = self._corregir_edad_escala_puntaje(X.copy())
        if "saldo_total" in X_temp.columns:
            self.limite_saldo_total_ = X_temp["saldo_total"].quantile(self.percentil_winsor)
        return self

    def _corregir_edad_escala_puntaje(self, X):
        if "edad_cliente" in X.columns:
            mask_edad = X["edad_cliente"] >= 120
            X.loc[mask_edad, "edad_cliente"] = X.loc[mask_edad, "edad_cliente"] - 100

        for col in ("salario_cliente", "total_otros_prestamos"):
            if col in X.columns:
                X[col] = X[col].astype(float)
                mask_escala = X[col] > self.umbral_escala
                X.loc[mask_escala, col] = X.loc[mask_escala, col] / 1000

        if "puntaje_datacredito" in X.columns:
            rango_min, rango_max = self.rango_puntaje_datacredito
            mask_negativo = X["puntaje_datacredito"] < 0
            mask_intermedio = (X["puntaje_datacredito"] > 0) & (X["puntaje_datacredito"] < rango_min)
            mask_alto = X["puntaje_datacredito"] > rango_max
            # El sentinela 0 ("sin score") se conserva intacto
            X.loc[mask_negativo | mask_intermedio | mask_alto, "puntaje_datacredito"] = np.nan
        return X

    def transform(self, X):
        X = self._corregir_edad_escala_puntaje(X.copy())
        if "saldo_total" in X.columns and self.limite_saldo_total_ is not None:
            X["saldo_total"] = X["saldo_total"].clip(upper=self.limite_saldo_total_)
        return X


class Imputacion(BaseEstimator, TransformerMixin):
    """Imputa los nulos remanentes tras `ReglasNegocio` (el EDA los dejó
    documentados pero sin imputar; esa decisión se tomó deliberadamente
    para esta etapa de feature engineering):

    - Numéricas -> mediana, calculada solo con datos de entrenamiento.
    - Categóricas -> moda, calculada solo con datos de entrenamiento.

    El sentinela 0 de `puntaje_datacredito` ("sin score") NO se imputa:
    ya se conserva tal cual desde `ReglasNegocio`, solo se imputan los
    NaN genuinos.
    """

    def __init__(self,
                 cols_mediana=("saldo_mora", "saldo_total", "saldo_mora_codeudor",
                               "puntaje_datacredito", "promedio_ingresos_datacredito",
                               "edad_cliente"),
                 cols_moda=("tendencia_ingresos",)):
        self.cols_mediana = cols_mediana
        self.cols_moda = cols_moda
        self.medianas_ = {}
        self.modas_ = {}

    def fit(self, X, y=None):
        self.medianas_ = {
            col: X[col].median() for col in self.cols_mediana if col in X.columns
        }
        self.modas_ = {}
        for col in self.cols_moda:
            if col in X.columns:
                moda = X[col].mode(dropna=True)
                if not moda.empty:
                    self.modas_[col] = moda.iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col, mediana in self.medianas_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mediana)
        for col, moda in self.modas_.items():
            if col in X.columns:
                X[col] = X[col].fillna(moda)
        return X


class Outliers(BaseEstimator, TransformerMixin):
    """Filtra los registros que, tras aplicar `ReglasNegocio`, siguen
    fuera del rango válido de negocio para edad_cliente (18 a 100 años).
    A diferencia de la plantilla original (que solo filtraba < 100), aquí
    se valida el rango completo porque la corrección de +100 ya se hizo
    en el paso anterior."""

    def __init__(self, edad_min=18, edad_max=100):
        self.edad_min = edad_min
        self.edad_max = edad_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "edad_cliente" not in X.columns:
            return X.copy()
        mask = X["edad_cliente"].between(self.edad_min, self.edad_max) | X["edad_cliente"].isna()
        return X[mask].copy()


class NuevasVariables(BaseEstimator, TransformerMixin):
    """Crea los atributos derivados que sobrevivieron al análisis de
    multicolinealidad (VIF): solo 'relacion_cuota_salario'.
    'relacion_deuda_ingreso' y 'relacion_capital_salario' NO se crean,
    porque tenían VIF > 100 frente a esta (redundantes).

    También descompone 'fecha_prestamo' en variables más útiles para el
    modelo (un timestamp crudo no aporta nada a un modelo tabular). Estas
    dos son nuevas respecto al EDA (que no las incluyó en su export), y
    se agregan aquí como parte propia de esta etapa de feature engineering:
    - mes_desembolso: estacionalidad del otorgamiento
    - antiguedad_meses: meses transcurridos entre el desembolso y la
      fecha de corte de los datos (madurez del crédito)
    """

    def __init__(self):
        self.mediana_relacion_ = None
        self.mediana_antiguedad_ = None
        self.fecha_corte_ = None

    def _construir(self, X):
        X = X.copy()

        if {"cuota_pactada", "salario_cliente"}.issubset(X.columns):
            salario_seguro = X["salario_cliente"].replace(0, np.nan)
            X["relacion_cuota_salario"] = X["cuota_pactada"] / salario_seguro

        if "fecha_prestamo" in X.columns:
            fecha = pd.to_datetime(X["fecha_prestamo"], errors="coerce")
            # 0 = sentinela "mes desconocido" (fecha_prestamo no parseable / NaT).
            # Se evita el dtype Int64 nullable: OneHotEncoder no maneja pd.NA de
            # forma confiable en el ordenamiento de categorías.
            X["mes_desembolso"] = fecha.dt.month.fillna(0).astype(int)
            fecha_corte = self.fecha_corte_ if self.fecha_corte_ is not None else fecha.max()
            X["antiguedad_meses"] = (
                (fecha_corte.year - fecha.dt.year) * 12 + (fecha_corte.month - fecha.dt.month)
            )
        return X

    def fit(self, X, y=None):
        if "fecha_prestamo" in X.columns:
            self.fecha_corte_ = pd.to_datetime(X["fecha_prestamo"], errors="coerce").max()
        X_temp = self._construir(X)
        if "relacion_cuota_salario" in X_temp.columns:
            self.mediana_relacion_ = X_temp["relacion_cuota_salario"].median()
        if "antiguedad_meses" in X_temp.columns:
            self.mediana_antiguedad_ = X_temp["antiguedad_meses"].median()
        return self

    def transform(self, X):
        X = self._construir(X)
        if "relacion_cuota_salario" in X.columns and self.mediana_relacion_ is not None:
            # NaN producido por salario_cliente == 0 (división segura) -> mediana
            X["relacion_cuota_salario"] = X["relacion_cuota_salario"].fillna(self.mediana_relacion_)
        if "antiguedad_meses" in X.columns and self.mediana_antiguedad_ is not None:
            # NaN producido por fecha_prestamo no parseable (NaT) -> mediana,
            # mismo patrón que mes_desembolso (sentinela 0) y relacion_cuota_salario
            X["antiguedad_meses"] = X["antiguedad_meses"].fillna(self.mediana_antiguedad_)
        return X


class DiscretizarAtributos(BaseEstimator, TransformerMixin):
    """Discretiza atributos continuos en rangos de negocio, cuando sea
    apropiado (por ejemplo, para un scorecard interpretable). No se aplica
    por defecto en `pipeline_basemodel` (los modelos de árboles/lineales
    ya manejan bien la variable continua); se deja lista por si el
    proyecto la necesita más adelante."""

    def __init__(self, columna, bins, labels, nueva_columna=None):
        self.columna = columna
        self.bins = bins
        self.labels = labels
        self.nueva_columna = nueva_columna or f"{columna}_rango"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columna in X.columns:
            X[self.nueva_columna] = pd.cut(X[self.columna], bins=self.bins, labels=self.labels)
        return X


class ToCategory(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X


class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    """Descarta columnas que no deben llegar al modelo:
    - 'puntaje': fuga de información confirmada (ver docstring del módulo)
    - 'saldo_principal': redundante con 'saldo_total' (correlación 0.913,
      ver docstring del módulo) — decisión tomada al final del EDA
    - 'fecha_prestamo': ya se descompuso en mes_desembolso / antiguedad_meses
      en NuevasVariables, el timestamp crudo ya no aporta nada
    """

    def __init__(self, cols_to_drop=("puntaje", "saldo_principal", "fecha_prestamo")):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=list(self.cols_to_drop), errors="ignore")


class EliminarCategorias(BaseEstimator, TransformerMixin):
    """Elimina registros de categorías inválidas/residuales dentro de una
    columna categórica (por ejemplo, un código de tipo_credito que no
    corresponde a ningún producto real). Se deja parametrizada y sin
    categorías por defecto porque el EDA de este dataset no identificó
    ninguna categoría inválida; si aparece una al revisar
    df[col].value_counts(), se agrega aquí."""

    def __init__(self, target_col=None, cats_to_drop=()):
        self.target_col = target_col
        self.cats_to_drop = cats_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.target_col or self.target_col not in X.columns or not self.cats_to_drop:
            return X.copy()
        return X[~X[self.target_col].isin(self.cats_to_drop)].copy()


class AgregarTarget(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="Pago_atiempo"):
        self.target_col = target_col
        self._y = None

    def fit(self, X, y=None):
        self._y = pd.Series(y, index=getattr(X, "index", None), name=self.target_col) if y is not None else None
        return self

    def transform(self, X):
        if self._y is None:
            return X
        X = X.copy()
        X[self.target_col] = self._y.reindex(X.index)
        return X


class ToDF(BaseEstimator, TransformerMixin):
    """Aplica el escalado/encoding final y devuelve un DataFrame (en vez
    del array de numpy que entrega ColumnTransformer por defecto), para
    poder seguir trabajando con nombres de columna.

    Separa las categóricas en NOMINALES (OneHotEncoder, sin orden) y
    ORDINALES (OrdinalEncoder, respetando el orden de negocio)."""

    def __init__(self, numeric_features, nominal_features, ordinal_features, ordinal_categories):
        self.numeric_features = numeric_features
        self.nominal_features = nominal_features
        self.ordinal_features = ordinal_features
        self.ordinal_categories = ordinal_categories
        self.ct_ = None

    def fit(self, X, y=None):
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        ordinal_enc = OrdinalEncoder(
            categories=self.ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("nom", ohe, self.nominal_features),
                ("ord", ordinal_enc, self.ordinal_features),
            ]
        )
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.ct_.transform(X)
        try:
            feat_names = self.ct_.get_feature_names_out()
        except AttributeError:
            feat_names = []
            for name, trans, cols in self.ct_.transformers_:
                if name == "remainder" and trans == "drop":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    feat_names.extend(trans.get_feature_names_out(cols))
                else:
                    feat_names.extend(cols)
        return pd.DataFrame(Xt, columns=feat_names, index=X.index)


def revisar_leakage_temporal(df, cols=("saldo_mora", "saldo_mora_codeudor", "saldo_total"),
                              target_col="Pago_atiempo"):
    """Mismo chequeo que se usó para confirmar la fuga de 'puntaje':
    compara la tasa del target entre 'tiene saldo > 0' y 'no tiene', para
    cada columna. Si alguna muestra una asociación casi perfecta (como el
    100%/0% que se vio en 'puntaje'), es señal de fuga temporal y debería
    excluirse de `numeric_features` más abajo.

    'saldo_principal' se excluyó de esta lista por defecto: ya se elimina
    del set de modelado por redundancia con 'saldo_total' (ver
    `ColumnasIrrelevantes`), independientemente de lo que arroje este
    chequeo de fuga.

    Antes de entrenar, revisar:
        revisar_leakage_temporal(df)
    """
    resultados = {}
    for col in cols:
        if col in df.columns:
            resultados[col] = pd.crosstab(df[col] > 0, df[target_col], normalize="index")
    return resultados


# ---------------------------------------------------------------------------
# 2. Pipeline base (limpieza + ingeniería de atributos, sin escalado/encoding)
# ---------------------------------------------------------------------------
# NOTA sobre el orden: 'reglas_negocio' corre ANTES que 'imputacion' a
# propósito. Las reglas de negocio marcan como NaN ciertos valores que
# antes no lo eran (p. ej. puntaje_datacredito fuera de rango); si la
# imputación corriera primero, esos valores inválidos se colarían en el
# cálculo de la mediana y en el propio dato, en vez de ser tratados como
# nulos e imputados correctamente.

pipeline_basemodel = Pipeline(steps=[
    ("correccion_tipos", CorreccionTipos()),
    ("recodificar_tendencia", RecodificarTendenciaIngresos()),
    ("duplicados_inicial", EliminarDuplicados()),
    ("eliminar_nulos", ColumnasNulos(threshold=0.4)),
    ("reglas_negocio", ReglasNegocio(umbral_escala=50_000_000, percentil_winsor=0.995)),
    ("imputacion", Imputacion()),
    ("outliers", Outliers(edad_min=18, edad_max=100)),
    ("nuevas_variables", NuevasVariables()),
    ("to_category", ToCategory(cols=["tipo_credito", "tipo_laboral", "tendencia_ingresos"])),
    ("columnas_irrelevantes", ColumnasIrrelevantes(cols_to_drop=("puntaje", "saldo_principal", "fecha_prestamo"))),
    ("eliminar_categorias", EliminarCategorias(target_col=None, cats_to_drop=())),
    ("duplicados_final", EliminarDuplicados()),
])


# ---------------------------------------------------------------------------
# 3. Pipeline ML (base + escalado/encoding final)
# ---------------------------------------------------------------------------

# Numéricas: originales (menos 'puntaje' y 'saldo_principal', eliminadas) + derivadas
numeric_features = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "relacion_cuota_salario", "antiguedad_meses",
]
# * Si al revisar leakage temporal se encuentra algo, se debe de eliminar de las variables numéricas inmediatamente anteriores (arriba de este comentario)

# Nominales (sin orden) -> OneHotEncoder
nominal_features = ["tipo_credito", "tipo_laboral", "mes_desembolso"]

# Ordinales (con orden de negocio) -> OrdinalEncoder
ordinal_features = ["tendencia_ingresos"]
# Las categorías deben coincidir EXACTAMENTE (incluida la mayúscula inicial)
# con las que quedaron tras el EDA: df['tendencia_ingresos'].cat.categories
# -> ['Creciente', 'Decreciente', 'Estable']
ordinal_categories = [["Decreciente", "Estable", "Creciente"]]

pipeline_ml = Pipeline(steps=[
    ("basemodel", pipeline_basemodel),
    ("preprocessor", ToDF(
        numeric_features=numeric_features,
        nominal_features=nominal_features,
        ordinal_features=ordinal_features,
        ordinal_categories=ordinal_categories,
    )),
])