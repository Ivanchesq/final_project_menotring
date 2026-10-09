# House Prices и Titanic

Один пайплайн на два соревнования Kaggle. Датасет переключается в `config.py`. `python main.py` читает сырые CSV, готовит признаки, подбирает модели и пишет сабмит.

## Запуск

Нужен Python 3.11+ и CSV в папках ниже.

```text
data/house_prices/train.csv
data/house_prices/test.csv
data/titanic/train.csv
data/titanic/test.csv
```

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Время запуска зависит от `N_TRIALS`, `DL_N_TRIALS` и `CV_FOLDS`: каждая попытка заново учится на каждом фолде.

## Переменные `config.py`

Задача:

| Поле | Что делает |
| --- | --- |
| `DATASET` | `"house_prices"` или `"titanic"`. От него зависят обработка, метрика и модели |
| `USE_DL` | `True` — в поиск идёт сеть. `False` — LightGBM, CatBoost и GradientBoosting |
| `RANDOM_STATE` | Общий seed для разбиения фолдов и моделей |
| `CV_FOLDS` | Число фолдов. Для домов это `KFold`, для титаника `StratifiedKFold` |
| `TOP_N` | Сколько лучших попыток записать в `results/<dataset>_topN.csv` |

Пути и цели. Менять их нужно, только если файлы лежат не в `data/`.

| Поле | Что делает |
| --- | --- |
| `TRAIN_DIR_*`, `TEST_DIR_*` | Сырые CSV |
| `PROCESSED_DIR_*` | Куда сохраняются обработанные таблицы |
| `SUBMISSION_DIR_*` | Куда пишется сабмит |
| `TARGET_COL_TITANIC` | Колонка `Survived` |
| `TARGET_COL_HOUSES_PRICES` | Колонка `SalePrice`. В обучении она уже в `log1p` |
| `NUMERIC_COLS_TITANIC` | Числовые колонки титаника, которые масштабирует логистическая регрессия: `Age`, `Fare`, `FamilySize` |
| `MODEL_PATH` | Файл ансамбля, по умолчанию `best_model.joblib` |
| `RESULTS_DIR` | Папка таблиц с метриками |

Поиск моделей. Пара в скобках — нижняя и верхняя граница. Learning rate и `alpha` перебираются в логарифмической шкале.

| Поле | Что перебирается |
| --- | --- |
| `N_TRIALS` | Число попыток для линейных моделей домов и для бустингов |
| `REGULARIZER_SEARCH_SPACE` | Штраф `l1` / `l2` / `elasticnet`, сила `alpha` и `l1_ratio` для ElasticNet |
| `LGBM_SEARCH_SPACE` | Число деревьев, глубина, learning rate |
| `CAT_SEARCH_SPACE` | Число итераций, глубина, learning rate |
| `GRADIENT_BOOST_SEARCH_SPACE` | Число деревьев, глубина, learning rate, `subsample` |
| `LOGREG_SEARCH_SPACE` | `C` логистической регрессии титаника. Больше `C` — слабее штраф |

Сеть. Эти поля используются, когда `USE_DL = True`.

| Поле | Что делает |
| --- | --- |
| `DL_N_TRIALS` | Число попыток Optuna для сети, отдельно от `N_TRIALS` |
| `DL_SEARCH_SPACE` | Слои, ширина, dropout, learning rate, размер батча, максимум эпох, активация (`relu` или `gelu`), оптимизатор (`adam`, `adamw`, `sgd`) |
| `DL_HIDDEN_STEP` | Шаг ширины скрытого слоя внутри диапазона `hidden` |
| `DL_LR_LOG` | `True` — learning rate перебирается в логарифмической шкале |
| `DL_DEFAULT` | Сеть, если учить её без Optuna. Число эпох берётся из `EPOCH` |
| `EPOCH` | Верхняя граница эпох для `DL_DEFAULT` |
| `DL_MONITOR_FRACTION` | Доля обучающей части фолда, по которой следят за ошибкой. В метрику фолда она не входит |
| `DL_PATIENCE` | Сколько эпох без улучшения ждать до ранней остановки |
| `DL_LR_PATIENCE` | Сколько эпох без улучшения ждать, прежде чем `ReduceLROnPlateau` уменьшит шаг |
| `DL_LR_FACTOR` | Во сколько раз уменьшить learning rate |
| `DL_MIN_DELTA` | Насколько должна упасть ошибка, чтобы эпоха считалась улучшением |
| `DL_SGD_MOMENTUM` | Momentum, если выбран SGD |
| `DL_THRESHOLD` | Порог вероятности класса 1 для титаника |

После запуска появляются:

- `best_model.joblib` — ансамбль
- `data/house_prices/submission.csv` или `data/titanic/submission.csv`
- `results/<dataset>_models.csv` — лучшая метрика каждого типа модели
- `results/<dataset>_top5.csv` — пять лучших попыток

## Что делает код

`process_data_house_price.py` и `process_data_titanic.py` заполняют пропуски статистиками train и собирают признаки. Для домов из train убираются две аномальные строки с большим `GrLivArea` и низкой ценой. `SalePrice` переводится в `log1p`.

Линейные модели и сеть получают one-hot, ординальные шкалы и target encoding района внутри кросс-валидации. Бустинги видят ординальные колонки как числа по порядку из описания данных. Номинальные колонки остаются категориями у LightGBM и CatBoost. `GradientBoostingRegressor` получает их коды.

В конце лучшая модель каждого типа учится на всём train. Для домов их прогнозы усредняются. Для титаника усредняются вероятности класса 1, порог — `DL_THRESHOLD`.

## Результаты

Цифры ниже — 5 фолдов, `shuffle=True`, `random_state=42`. Это фиксированные настройки, не полный перебор из `main.py`. Optuna может выбрать другие параметры. Сеть в эту проверку не входила: её RMSE или accuracy печатает `main.py`, когда `USE_DL = True`.

Дома, RMSE на логарифме цены. Предсказание средним даёт 0.3996.

| Модель | Настройки | RMSE |
| --- | --- | --- |
| ElasticNet | alpha=0.001, l1_ratio=0.5 | 0.1118 |
| Lasso | alpha=0.001 | 0.1121 |
| Ridge | alpha=10 | 0.1129 |
| CatBoost | 200 итераций, глубина 6, lr=0.05 | 0.1189 |
| GradientBoosting | 150 деревьев, глубина 3, lr=0.05, subsample=0.8 | 0.1201 |
| LightGBM | 200 деревьев, глубина 6, lr=0.05 | 0.1245 |

Титаник, accuracy. Те же настройки бустингов, логистическая регрессия — бейзлайн проекта.

| Модель | Accuracy |
| --- | --- |
| CatBoost | 0.8395 |
| LightGBM | 0.8384 |
| GradientBoosting | 0.8372 |
| LogisticRegression | 0.8002 |
