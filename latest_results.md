# Confronto sintetico degli scheduler carbon-aware

## Setup comune

- **Job:** 23.560
- **Cluster:** 880 nodi
- **Job predictions:** `data/job_predictions/test_predictions.parquet`
- **Carbon actual:** `data/carbon_intensity/actual/actual.json`
- **Forecast CI:** `data/carbon_intensity/snapshots/test_ridge_direct.json`
- **Energia totale:** 67.52 MWh per tutti i run
- **Baseline:** EASY con job predictions

## Tabella comparativa

| Scheduler                    | Job info   | Carbon info   | Delay policy        | Emissioni (tCO2e) |    vs EASY | Wait medio | Slowdown medio | Peak (MW) | Valutazione                             |
| ---------------------------- | ---------- | ------------- | ------------------- | ----------------: | ---------: | ---------: | -------------: | --------: | --------------------------------------- |
| **EASY**                     | Prediction | —             | —                   |        **17.475** |   baseline |  **458 s** |       **3.00** | **0.844** | Baseline realistica                     |
| **Carbon fixed**             | Prediction | Actual futura | 6 h                 |            17.373 | **−0.58%** |    7,923 s |         463.07 |     0.855 | Saving positivo, QoS troppo penalizzato |
| **Carbon fixed forecast**    | Prediction | Forecast      | 6 h                 |            17.450 | **−0.14%** |    6,898 s |         472.45 | **0.905** | Poco saving, QoS e peak peggiori        |
| **Duration-scaled**          | Prediction | Actual futura | 1× runtime predetto |        **17.229** | **−1.41%** |    1,248 s |       **2.70** | **0.843** | Miglior risultato sperimentale          |
| **Duration-scaled forecast** | Prediction | Forecast      | 1× runtime predetto |            17.508 | **+0.19%** |      880 s |           3.44 |     0.868 | Realistico, ma carbon saving negativo   |

## Lettura rapida

### Miglior risultato carbon

**Duration-scaled + job predictions + actual CI**

- emissioni: **17.229 tCO2e**
- saving vs EASY: **1.41%**
- waiting medio: **1.248 s**
- bounded slowdown medio: **2.70**
- peak power: **0.843 MW**

È la policy che mostra il miglior compromesso tra carbon e QoS, ma utilizza ancora la CI futura reale e quindi rappresenta un benchmark con informazione privilegiata.

### Configurazione realistica completa

**Duration-scaled + job predictions + carbon forecast**

- emissioni: **17.508 tCO2e**
- variazione vs EASY: **+0.19%**
- waiting medio: **880 s**
- bounded slowdown medio: **3.44**
- peak power: **0.868 MW**

È la configurazione più vicina a un uso online reale, ma con `max_delay_fraction=1` non produce ancora un beneficio carbon.

### Fixed-delay con forecast

**Carbon fixed + job predictions + forecast**

- emissioni: **17.450 tCO2e**
- saving vs EASY: **0.14%**
- waiting medio: **6.898 s**
- bounded slowdown medio: **472.45**
- peak power: **0.905 MW**

Ottiene un piccolo saving, ma il costo QoS è sproporzionato e il peak power aumenta sensibilmente.

## Conclusione

| Aspetto                                            | Scheduler migliore                  |
| -------------------------------------------------- | ----------------------------------- |
| **Emissioni minime**                               | Duration-scaled + actual CI         |
| **QoS più vicino a EASY tra i carbon-aware**       | Duration-scaled forecast            |
| **Miglior compromesso sperimentale**               | Duration-scaled + actual CI         |
| **Scenario più realistico**                        | Duration-scaled forecast            |
| **Scheduler da evitare come candidato production** | Fixed-delay 6 h                     |
| **Problema principale da risolvere**               | Robustezza agli errori del forecast |

Il risultato centrale è che **la struttura duration-scaled funziona bene**, ma il beneficio carbon non sopravvive ancora quando la CI futura reale viene sostituita dal forecast.

Il prossimo passo naturale è uno sweep di:

```text
max_delay_fraction = 0.25, 0.50, 0.75, 1.00
```

eventualmente combinato con un limite assoluto:

```text
delay = min(alpha × predicted_runtime, max_delay_absolute)
```
