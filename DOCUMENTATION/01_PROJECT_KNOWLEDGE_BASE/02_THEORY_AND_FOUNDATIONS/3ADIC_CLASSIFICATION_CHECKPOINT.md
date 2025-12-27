# 3-Adic Classification Checkpoint

> **Fecha:** 2025-12-27
> **Propósito:** Documentar todos los usos potenciales de matemáticas 3-ádicas para clasificación
> **Estado:** Checkpoint de investigación activo

---

## Resumen Ejecutivo

Las matemáticas 3-ádicas proporcionan un framework único para clasificación jerárquica basado en:
- **Ultrametricidad:** d(a,c) ≤ max(d(a,b), d(b,c))
- **Estructura jerárquica natural:** 3→9→27→81→243...
- **Compatibilidad biológica:** Codones tienen 3 posiciones (wobble)

---

## 1. CLASIFICACIÓN DE SECUENCIAS BIOLÓGICAS

### 1.1 Clasificación de Codones (64 clases)
```
Aplicación: Mapear 64 codones a espacio 3-ádico
Método: padic_distance(codon1, codon2, p=3)
Clasificador: Codones sinónimos se agrupan naturalmente
Validación: 20 aminoácidos emergen de clustering 3-ádico
```

**Código existente:** `src/utils/padic_shift.py:PAdicCodonAnalyzer`

### 1.2 Clasificación de Aminoácidos (20 clases)
```
Aplicación: Clasificar aminoácidos por propiedades 3-ádicas
Método: synonymous_padic_spread(amino_acid)
Hipótesis: Aminoácidos hidrofóbicos vs hidrofílicos se separan
Estado: Por validar experimentalmente
```

### 1.3 Clasificación de Mutaciones
```
Aplicación: Clasificar impacto de mutaciones
Categorías:
  - Silenciosa (mismo aminoácido, distancia 3-ádica baja)
  - Conservativa (aminoácido similar, distancia media)
  - Radical (aminoácido diferente, distancia alta)
  - Nonsense (stop codon)
Método: padic_distance(codon_wt, codon_mut)
```

---

## 2. CLASIFICACIÓN EN ENFERMEDADES

### 2.1 Resistencia a Drogas HIV (4 clases)
```
Clases: NRTI, NNRTI, PI, INSTI
Hallazgo validado:
  - NRTI: d=6.05 (sitio activo más restringido)
  - INSTI: d=5.16 (alta restricción)
  - NNRTI: d=5.34 (moderada)
  - PI: d=3.60 (más flexible)
Aplicación: Predecir clase de resistencia desde secuencia
```

**Código:** `DOCUMENTATION/.../HIV_PADIC_ANALYSIS/`

### 2.2 Riesgo Autoinmune (Binario: En zona/Fuera de zona)
```
Aplicación: Goldilocks Zone Classification
Parámetros:
  - center = 0.5 (zona óptima)
  - width = 0.15 (ancho gaussiano)
Clases:
  - IN_ZONE: Riesgo autoinmune (cross-reactivo)
  - OUT_ZONE: Sin riesgo (muy similar o muy diferente)
```

**Código:** `src/core/padic_math.py:is_in_goldilocks_zone()`

### 2.3 Severidad de Long COVID (Escala continua → discretizada)
```
Aplicación: Clasificar variantes de Spike por potencial de microcoágulos
Método: compute_goldilocks_score(spike_distance)
Discretización: Low/Medium/High risk
```

### 2.4 Progresión de Enfermedades de Expansión de Repeticiones
```
Aplicación: Clasificar conteo de repeticiones
Clases:
  - Normal (n < threshold)
  - Pre-mutación (threshold ≤ n < pathogenic)
  - Patogénico (n ≥ pathogenic)
Método: padic_distance detecta cruce de umbral
```

---

## 3. CLASIFICACIÓN INMUNOLÓGICA

### 3.1 Clasificación de Epítopos (Multi-clase)
```
Clases:
  - CTL (Citotóxico T Linfocito)
  - B-cell (Anticuerpo)
  - bnAb (Broadly Neutralizing Antibody)
  - Helper T-cell
Método: Embedding en espacio hiperbólico 3-ádico
Datos disponibles: LANL 2,115 epítopos HLA-restringidos
```

### 3.2 Clasificación de Respuesta Inmune (4 clases)
```
Clases (del framework Nobel Immune):
  - Tolerancia (muy similar a self)
  - Activación (zona óptima)
  - Anergía (fuera de zona)
  - Deleción (muy diferente)
Método: Umbrales de distancia 3-ádica
```

**Código:** `src/validation/nobel_immune.py`

### 3.3 Predicción de Binding MHC
```
Aplicación: Clasificar péptidos por afinidad MHC
Hipótesis: Distancia 3-ádica correlaciona con IC50
Clases: Strong binder / Weak binder / Non-binder
Umbral típico: IC50 < 500nM = binder
```

---

## 4. CLASIFICACIÓN CRISPR

### 4.1 Especificidad Off-Target (Binario)
```
Aplicación: Clasificar sitios como on-target vs off-target
Método: padic_distance con pesos de posición
  - Seed region (11-20 bp): peso alto
  - Non-seed (1-10 bp): peso bajo
```

**Código:** `src/analysis/crispr/padic_distance.py`

### 4.2 Riesgo de Off-Target (Escala)
```
Clases:
  - Safe (distancia 3-ádica alta)
  - Caution (distancia media)
  - Dangerous (distancia baja = muy similar)
```

---

## 5. CLASIFICACIÓN DE ESTRUCTURAS PROTEICAS

### 5.1 Clasificación de Rotámeros
```
Aplicación: Clasificar conformaciones de cadena lateral
Número de clases: Variable por aminoácido
Método: Distancia 3-ádica entre estados rotaméricos
Estado: Script en desarrollo
```

**Script:** `scripts/ingest/ingest_pdb_rotamers.py`

### 5.2 Clasificación de Plegamiento
```
Aplicación: Clasificar dominios por fold
Hipótesis: Folds similares tienen embedding 3-ádico cercano
Clases: CATH/SCOP hierarchy
```

---

## 6. CLASIFICACIÓN FILOGENÉTICA

### 6.1 Clasificación de Variantes Virales
```
Aplicación: Clasificar variantes (ej: SARS-CoV-2 Alpha, Delta, Omicron)
Método: Distancia 3-ádica captura divergencia jerárquica
Ventaja: Ultrametricidad natural para árboles filogenéticos
```

### 6.2 Clasificación de Clados
```
Aplicación: Asignar secuencia a clado
Método: Nearest neighbor en espacio 3-ádico
Validación: Comparar con asignación tradicional
```

---

## 7. CLASIFICACIÓN DE TARGETS TERAPÉUTICOS

### 7.1 Druggability (Binario)
```
Aplicación: Clasificar proteínas como druggable/non-druggable
Hipótesis: Sitios de binding tienen signature 3-ádica
Features: padic_embedding de bolsillo de binding
```

### 7.2 Clasificación de Mecanismo de Acción
```
Aplicación: Clasificar drogas por mecanismo
Clases: Inhibidor, Agonista, Antagonista, Modulador alostérico
Método: Distancia 3-ádica entre target y ligando
```

---

## 8. IDEAS EXPERIMENTALES (Por Implementar)

### 8.1 Clasificador k-NN 3-Ádico
```python
class PAdicKNN:
    """k-Nearest Neighbors usando distancia 3-ádica."""

    def __init__(self, k=5, p=3):
        self.k = k
        self.p = p

    def fit(self, X_indices, y):
        self.X = X_indices
        self.y = y

    def predict(self, query_idx):
        distances = [padic_distance(query_idx, x, self.p) for x in self.X]
        k_nearest = np.argsort(distances)[:self.k]
        return mode(self.y[k_nearest])
```

### 8.2 Random Forest con Features 3-Ádicos
```python
def extract_padic_features(sequence):
    """Extraer features 3-ádicas de secuencia."""
    return {
        'padic_digits': padic_digits(seq_to_index(sequence)),
        'valuation': padic_valuation(seq_to_index(sequence)),
        'hierarchical_embedding': compute_hierarchical_embedding(sequence),
        'goldilocks_score': compute_goldilocks_score(distance_to_reference),
    }
```

### 8.3 Neural Network con P-Adic Loss
```python
class PAdicClassifier(nn.Module):
    """Clasificador con regularización 3-ádica."""

    def __init__(self, n_classes):
        self.encoder = HyperbolicEncoder()
        self.classifier = nn.Linear(latent_dim, n_classes)
        self.padic_loss = PAdicRankingLoss()

    def forward(self, x, indices):
        z = self.encoder(x)
        logits = self.classifier(z)
        ranking_loss = self.padic_loss(z, indices)
        return logits, ranking_loss
```

### 8.4 Clustering Jerárquico 3-Ádico
```python
def padic_hierarchical_clustering(indices, p=3):
    """Clustering usando ultrametricidad 3-ádica."""
    distance_matrix = padic_distance_matrix(indices, p)
    # La ultrametricidad garantiza árboles consistentes
    linkage = scipy.cluster.hierarchy.linkage(
        distance_matrix,
        method='complete'  # max distance = ultrametric
    )
    return linkage
```

---

## 9. MÉTRICAS DE EVALUACIÓN ESPECÍFICAS

### 9.1 P-Adic Accuracy
```python
def padic_accuracy(y_true, y_pred, indices):
    """Accuracy ponderada por distancia 3-ádica."""
    correct = (y_true == y_pred)
    weights = 1.0 / (1.0 + padic_distance_vectorized(indices, reference))
    return (correct * weights).sum() / weights.sum()
```

### 9.2 Hierarchical F1
```python
def hierarchical_f1(y_true, y_pred, class_hierarchy):
    """F1 que considera jerarquía de clases 3-ádica."""
    # Errores entre clases cercanas penalizan menos
    pass
```

---

## 10. DATASETS DISPONIBLES PARA VALIDACIÓN

| Dataset | Registros | Clases | Archivo |
|---------|-----------|--------|---------|
| Stanford HIVDB | 7,154 | 4 (drug classes) | Procesado |
| LANL CTL Epitopes | 2,115 | Multi (HLA types) | Procesado |
| CATNAP bnAb | 189,879 | Binary (neutralizing) | Procesado |
| V3 Coreceptor | 2,935 | Binary (CCR5/CXCR4) | Procesado |
| Codon Table | 64 | 20 (amino acids) | Built-in |

---

## 11. PRÓXIMOS PASOS

### Inmediatos (Esta semana)
- [ ] Implementar PAdicKNN básico
- [ ] Validar en dataset de codones (64→20)
- [ ] Benchmark vs k-NN euclidiano

### Corto plazo (Este mes)
- [ ] Clasificador de resistencia HIV
- [ ] Clasificador de epítopos
- [ ] Integrar con pipeline de training

### Largo plazo
- [ ] Paper: "3-Adic Classification in Bioinformatics"
- [ ] Librería standalone: `padic-classify`
- [ ] Aplicación web para clasificación

---

## 12. REFERENCIAS

### Código Existente
- `src/core/padic_math.py` - Operaciones core
- `src/losses/padic/` - Loss functions para training
- `src/utils/padic_shift.py` - Encoding de secuencias
- `src/analysis/immunology/padic_utils.py` - Goldilocks zone

### Documentación
- `DOCUMENTATION/.../P_ADIC_GENOMICS/` - Análisis HIV completo
- `DOCUMENTATION/.../MATHEMATICAL_FOUNDATIONS.md` - Teoría

### Literatura
- Kozyrev 2006: P-adic Analysis Methods
- Wong 1975: Coevolution Theory (wobble hypothesis)

---

## Notas de Checkpoint

**Lo que sabemos que funciona:**
1. Distancia 3-ádica correlaciona con fitness cost en HIV (r=-0.89)
2. Goldilocks zone predice riesgo autoinmune
3. Triplet ranking mejora embeddings jerárquicos
4. Codones sinónimos se agrupan en espacio 3-ádico

**Lo que necesita validación:**
1. Generalización a otras enfermedades
2. Comparación cuantitativa vs métodos clásicos
3. Escalabilidad a datasets grandes

**Ideas especulativas:**
1. El código genético es un código corrector de errores 3-ádico
2. La evolución optimiza distancia 3-ádica, no Hamming
3. Proteínas con función similar tienen embedding 3-ádico cercano

---

*Este documento es un checkpoint vivo. Actualizar conforme se validen ideas.*
