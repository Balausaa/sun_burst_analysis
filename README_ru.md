## AI spacewhether / AI космическая погода – документация (RU)

### Общее описание

AI spacewhether – это демонстрационный (MVP) веб‑сервис для **раннего предупреждения о космической погоде**.  
Он анализирует динамические спектры солнечных радиовсплесков (CSV: `time,frequency,intensity`), обнаруживает **радиовсплески II типа** и оценивает **риск для спутников** на основе гибридной архитектуры:

- модуль **CNN + LSTM** для анализа радиовсплесков II типа;
- модуль **Surya** (foundation‑модель от NASA/IBM) для прогноза космической погоды;
- гибридный **risk‑engine**, который комбинирует оба источника информации в итоговый уровень риска: `LOW / MEDIUM / HIGH / EXTREME`.

Система состоит из:

- **Backend**: FastAPI + PyTorch
- **Frontend**: Streamlit + Plotly
- **AI ядро**:
  - ResNet18 (torchvision, предобученная на ImageNet) как CNN‑экстрактор признаков;
  - однослойный LSTM для моделирования временной эволюции;
  - Surya‑модель (`nasa-ibm-ai4science/Surya-1.0`) как источник «физически осмысленных» признаков (вероятность вспышки, скорость солнечного ветра).

\> Surya загружается локально (веса лежат в `models/surya.366m.v1.pt`) и используется для инференса.

---

### Структура проекта

- `backend/`
  - `main.py` – FastAPI приложение и REST‑эндпоинты.
  - `models/`
    - `cnn_model.py` – ResNet18‑экстрактор + линейная голова для вероятности всплеска II типа.
    - `lstm_model.py` – LSTM для оценки «штормовой» вероятности.
    - `surya_model.py` – обёртка над Surya: загрузка локальных весов + лёгкая голова для вывода `flare_probability` и `solar_wind_speed`.
  - `processing/`
    - `signal_processor.py` – pipeline CNN+LSTM для радиовсплесков II типа (спектрограмма → признаки → вероятности).
    - `surya_processor.py` – pipeline Surya (CSV → спектрограмма → энтропия/дрейф → Surya‑выходы).
    - `entropy.py` – расчёт энтропии сигнала.
  - `hybrid_analyzer.py` – комбинированный анализатор, который объединяет:
    - результаты модуля II типа;
    - результаты модуля Surya;
    - считает итоговый `final_risk_level`.
  - `risk_engine.py` – risk‑engine для:
    - «штормовой» вероятности (старый модуль);
    - Surya‑выходов (geomagnetic_risk);
  - `synthetic_generator.py` – генератор синтетических динамических спектров.
- `frontend/`
  - `app.py` – Streamlit‑дашборд (UI).
- `data/`
  - `example_signal.csv` – пример CSV.
- `datas/`
  - `low_risk_example.csv` – синтетический пример для **LOW**.
  - `medium_risk_example.csv` – пример для **MEDIUM**.
  - `high_risk_example.csv` – пример для **HIGH**.
  - `extreme_risk_example.csv` – пример для **EXTREME**.
- `models/`
  - `surya.366m.v1.pt` – веса Surya (скачиваются вручную, **не кладутся в git**).
- `requirements.txt`
- `README.md` – подробная документация на английском.
- `README_ru.md` – эта документация.

---

### Tech stack

- **Язык**: Python 3.11+
- **Backend**:
  - FastAPI
  - Uvicorn
  - Pydantic v2
- **ML / численные библиотеки**:
  - PyTorch + torchvision
  - NumPy
  - SciPy
  - pandas
- **Surya / Hugging Face**:
  - `huggingface-hub` (для загрузки весов, но в текущей конфигурации – только `local_files_only`)
- **Frontend**:
  - Streamlit
  - Plotly
  - `streamlit-plotly-events` (для кликабельного графика истории)

---

### Подготовка окружения на новом компьютере

1. **Установите Python 3.11+**  
   Рекомендуется официальная сборка Python для Windows или Linux.

2. **Склонируйте репозиторий** (или перенесите папку проекта):

```bash
git clone <url_вашего_репозитория>
cd solar_burst   # или d:\solar_burst, если копируете напрямую
```

3. **Создайте виртуальное окружение и активируйте его** (Windows / PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. **Установите зависимости**:

```bash
pip install -r requirements.txt
```

5. **Скачайте веса Surya и положите их в папку `models/`**:

- Перейдите на страницу модели Surya на Hugging Face:  
  `https://huggingface.co/nasa-ibm-ai4science/Surya-1.0`
- Скачайте файл весов `surya.366m.v1.pt`.
- Создайте в корне проекта папку `models` (если её ещё нет) и поместите файл туда:

```text
<project_root>/
  models/
    surya.366m.v1.pt
```

> Файл `models/surya.366m.v1.pt` **игнорируется git‑ом** (см. `.gitignore`),  
> так что он не будет загружаться в репозиторий.

Surya‑обёртка (`backend/models/surya_model.py`) автоматически:

- сначала проверяет наличие локального файла `models/surya.366m.v1.pt`;
- если файл найден – использует его;
- в противном случае пытается найти веса в локальном HF‑кэше (`local_files_only=True`).

---

### Как запустить backend (API) на новом компьютере

1. **Убедитесь, что venv активирован**:

```bash
.venv\Scripts\activate
```

2. **(Рекомендуется) установить PYTHONPATH**, чтобы `backend` импортировался из корня:

Windows PowerShell:

```bash
$env:PYTHONPATH="."
uvicorn backend.main:app --reload
```

Linux/macOS:

```bash
export PYTHONPATH=.
uvicorn backend.main:app --reload
```

3. **Проверьте, что API работает**:

- Откройте в браузере: `http://127.0.0.1:8000/health` – должно вернуть JSON:

```json
{"status": "ok", "service": "HelioGuard AI"}
```

4. **Основные эндпоинты**:

- `POST /predict` – комбинированный анализ (Surya + Type II + hybrid risk):

  - **Request (multipart/form-data)**:

    - поле `csv_file`: файл CSV с колонками `time,frequency,intensity`;
    - опционально: `satellite_id`, `session_id`.

  - Пример запроса (PowerShell / curl):

    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" `
      -F "csv_file=@datas/high_risk_example.csv" `
      -F "satellite_id=HELIO-SAT-1" `
      -F "session_id=test-high"
    ```

  - Пример ответа:

    ```json
    {
      "timestamp": "2026-02-25T10:15:00Z",
      "satellite_id": "HELIO-SAT-1",
      "signal_metrics": {
        "entropy": 1.907,
        "drift_rate_mhz_s": 1.5,
        "max_intensity": 0.497,
        "mean_intensity": 0.031
      },
      "type_ii_probability": 0.82,
      "solar_wind_speed_kms": 538,
      "geomagnetic_risk": "MEDIUM",
      "final_risk_level": "HIGH",
      "recommendation": "Elevated risk. Review mission timelines, avoid high-risk maneuvers, and prepare contingency plans.",
      "event_history": [
        {
          "timestamp": "...",
          "satellite_id": "HELIO-SAT-1",
          "session_id": "test-high",
          "type_ii_probability": 0.65,
          "final_risk_level": "MEDIUM"
        }
      ]
    }
    ```

- `POST /analyze` – старый эндпоинт только для модуля II типа (принимает матрицу `signal_data`).
- `POST /generate-synthetic` – генерация синтетических спектров.

---

### Как запустить Streamlit‑дашборд

1. В активированном venv установите `PYTHONPATH` (как для backend):

```bash
$env:PYTHONPATH="."
streamlit run frontend/app.py
```

2. Откройте в браузере адрес, который напишет Streamlit (обычно `http://localhost:8501`).

3. Функции UI:

- Загрузка CSV (`time,frequency,intensity`);
- Построение динамического спектра (heatmap Plotly);
- Блок **«Анализ радиовсплеска»**:
  - gauge‑индикатор `Type II Probability`;
  - текстовая скорость дрейфа (МГц/с).
- Блок **«Прогноз космической погоды»**:
  - gauge‑индикатор `Вероятность вспышки`;
  - `Скорость солнечного ветра (км/с)`;
  - `Геомагнитный риск` (на основе Surya);
  - `Итоговый риск для спутника` (hybrid risk engine).
- **Последнее событие** – детальный вывод всех метрик.
- **История событий** – таблица + интерактивный график:
  - клик по точке на графике показывает все детали соответствующего события.

---

### Как это работает внутри (кратко)

1. **Предобработка CSV** (`surya_processor.py` / `signal_processor.py`):

   - CSV → таблица → pivot по `time` и `frequency` → матрица `[T, F]`;
   - нормализация, медианный фильтр;
   - вычисление:
     - энтропии (структурированность сигнала),
     - скорости дрейфа (максимум по частоте во времени),
     -统计 метрик интенсивности (max/mean).

2. **Модуль Surya** (`surya_model.py`, `surya_processor.py`):

   - Спектрограмма подаётся в лёгкую «surrogate»‑голову,
   - Выходы:
     - `flare_probability` – вероятность солнечной вспышки;
     - `solar_wind_speed` – оценка скорости солнечного ветра.

3. **Модуль Type II** (`signal_processor.py`, `cnn_model.py`, `lstm_model.py`):

   - Спектрограмма → 3‑канальное изображение 224×224 → ResNet18 (замороженный);
   - Вектор признаков → линейная голова + sigmoid → базовая вероятность II типа;
   - Признаки `[features, entropy, drift, duration]` → LSTM → «штормовая» вероятность;
   - Гибридная эвристика дополнительно использует энтропию и дрейф.

4. **Гибридный risk‑engine** (`hybrid_analyzer.py`):

   - Нормализует:
     - `type_ii_probability`,
     - `flare_probability`,
     - `drift_rate_mhz_s` (0–30 МГц/с),
     - `solar_wind_speed_kms` (300–900 км/с).
   - Считает итоговый score:

     \[
     score = 0.4 \cdot \text{TypeII}^{1.3} +
             0.3 \cdot \text{Flare}^{1.3} +
             0.2 \cdot \text{Drift}^{1.1} +
             0.1 \cdot \text{Wind}^{1.1}
     \]

   - Маппинг:
     - `< 0.20` → LOW
     - `< 0.45` → MEDIUM
     - `< 0.70` → HIGH
     - `>= 0.70` → EXTREME

   - Возвращает `final_risk_level` и текстовую рекомендацию.

---

### Git и большие файлы

- В `.gitignore` настроено:

  - игнорировать файл `models/surya.366m.v1.pt`;
  - при этом папка `models/` остаётся в репозитории (можно добавить туда README, скрипты и т.п.);
  - игнорируются кэши (`__pycache__/`, `.pytest_cache/`, `.streamlit/`, `.cache/`).

Таким образом:

- вы можете безопасно хранить код и конфигурацию в git;
- каждый разработчик/оператор **локально** скачивает веса Surya и кладёт их в `models/`;
- репозиторий остаётся лёгким и переносимым.

