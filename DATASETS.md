# Погодные датасеты — полный справочник

Данный документ описывает все погодные датасеты, упомянутые в ТЗ хакатона MifiHack и в контекстных материалах, способы их скачивания и ключевые ссылки, включая Google-ресурсы (WeatherBench2) и все вторичные ссылки.

---

## 🔗 Google-ссылки и WeatherBench2-экосистема

| # | Ссылка | Описание |
|---|--------|----------|
| 1 | [WeatherBench (оригинал)](https://sites.research.google/gr/weatherbench/) | Первая версия облачного интерфейса Google для сравнения погодных моделей (2022). |
| 2 | [WeatherBench2 (статья)](https://arxiv.org/pdf/2308.15560) | Научная статья: "WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models" (2023). |
| 3 | [WeatherBench2 (GitHub)](https://github.com/google-research/weatherbench2) | Основной репозиторий: код, пайплайны тестирования и метрики оценки качества моделей. |
| 4 | [Детерминированные scores (web)](https://sites.research.google/gr/weatherbench/deterministic-scores/) | Web-интерфейс сравнения качества детерминированных моделей прогноза погоды. |
| 5 | [Вероятностные scores (web)](https://sites.research.google/gr/weatherbench/probabilistic-scores/) | Web-интерфейс сравнения качества вероятностных моделей прогноза погоды. |
| 6 | [WeatherBench2 — Google Cloud Storage (данные)](https://console.cloud.google.com/storage/browser/weatherbench2) | Прямой доступ к данным WeatherBench2 в Google Cloud Storage (Zarr-формат). |
| 7 | `gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr` | **Главный рекомендуемый датасет хакатона**: готовая ERA5 Zarr (0.25°, 6-часовые кадры, 1959–2023, с derived-переменными). |

---

## 🌍 Европейские организации — источники данных

| # | Ссылка | Организация | Описание |
|---|--------|-------------|----------|
| 1 | [ECMWF](https://www.ecmwf.int/) | ECMWF | Европейский центр среднесрочных прогнозов погоды. Основной источник ERA5, IFS HRES. |
| 2 | [Copernicus](https://www.copernicus.eu/) | Copernicus | Европейская программа наблюдения Земли. |
| 3 | [Climate Data Store (CDS)](https://cds.climate.copernicus.eu/) | Copernicus CDS | Централизованное хранилище погодных датасетов ECMWF/Copernicus. Открытая платформа для выбора, настройки и выгрузки данных. |

---

## 🌐 Международные организации

| # | Ссылка | Организация | Описание |
|---|--------|-------------|----------|
| 1 | [WMO](https://wmo.int/) | WMO | Всемирная метеорологическая организация. |
| 2 | [UKMO / MetOffice](https://www.metoffice.gov.uk/) | UKMO | Метеорологическая служба Великобритании. |
| 3 | [NOAA](https://www.noaa.gov/) | NOAA | Национальное управление океанических и атмосферных исследований США. |

---

## 📦 Конкретные датасеты — описание и способы скачивания

### 1. ERA5 hourly data on pressure levels

- **Ссылка:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview
- **Тип:** Реанализ (reanalysis) — прогнозы численных моделей + ассимиляция спутников/радаров/станций/буев.
- **Что это:** Погодные параметры на 37 атмосферных слоях (уровнях давления) с часовым шагом.
- **Переменные:** T (температура), U/V (компоненты ветра), Z (геопотенциал), Q (удельная влажность), R (отн. влажность), W (вертикальная скорость).
- **Пространственное разрешение:** Глобальная сетка 0.25° (721×1440).
- **Временной охват:** С 1950-х по настоящее время, часовой шаг.
- **Формат:** GRIB / NetCDF (через CDS API).
- **Уровни:** 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 гПа (и более).

#### Как скачать (CDS API):

```bash
# 1. Установить CDS API
pip install cdsapi

# 2. Зарегистрироваться на https://cds.climate.copernicus.eu/ и получить API-ключ
# 3. Создать ~/.cdsapirc:
#    url: https://cds.climate.copernicus.eu/api
#    key: <ваш-uid>:<ваш-api-key>
```

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['temperature', 'u_component_of_wind', 'v_component_of_wind',
                      'geopotential', 'specific_humidity'],
        'pressure_level': ['1000', '925', '850', '700'],
        'year': ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
        'month': ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'],
        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
        'grid': [0.25, 0.25],
    },
    'era5_pressure_levels.nc'
)
```

---

### 2. ERA5 hourly data on single levels

- **Ссылка:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
- **Тип:** Реанализ.
- **Что это:** Наземные (приземные) погодные параметры с часовым шагом.
- **Переменные:** t2m (температура 2м), mslp (давление на уровне моря), u10/v10 (ветер 10м), tp (осадки), sst (температура поверхности моря), tcc (облачность), tcwv (водяной пар в колонке) и многие другие.
- **Пространственное разрешение:** Глобальная сетка 0.25°.
- **Временной охват:** С 1950-х по настоящее время, часовой шаг.

#### Как скачать (CDS API):

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['2m_temperature', 'mean_sea_level_pressure',
                      '10m_u_component_of_wind', '10m_v_component_of_wind',
                      'total_precipitation', 'sea_surface_temperature',
                      'total_cloud_cover', 'total_column_water_vapour'],
        'year': ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
        'month': ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'],
        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
        'grid': [0.25, 0.25],
    },
    'era5_single_levels.nc'
)
```

---

### 3. ERA5-Land hourly data

- **Ссылка:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview
- **Тип:** Реанализ.
- **Что это:** Детализированный реанализ наземных параметров только по суше.
- **Пространственное разрешение:** Сетка 0.1° (более мелкая, чем стандартный ERA5).
- **Временной охват:** С 1950-х по настоящее время, часовой шаг.
- **Применение:** Полезен для задач, требующих высокого пространственного разрешения над сушей (гидрология, сельское хозяйство).

#### Как скачать:

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': ['2m_temperature', 'total_precipitation'],
        'year': '2020',
        'month': '01',
        'day': ['01', '02', '03'],
        'time': ['00:00', '01:00', '02:00'],
        'format': 'netcdf',
    },
    'era5_land.nc'
)
```

---

### 4. E-OBS (In-situ Gridded Observations — Europe)

- **Ссылка:** https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-europe?tab=overview
- **Тип:** Наблюдения (интерполированные).
- **Что это:** Датасет наблюдений температур и осадков по Европе, интерполированный на регулярную сетку.
- **Пространственное разрешение:** 0.1° (только Европа).
- **Временной шаг:** Суточный.
- **Временной охват:** С 1950 года по настоящее время.
- **Применение:** Валидация моделей на реальных наблюдениях (а не реанализе).

#### Как скачать:

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'insitu-gridded-observations-europe',
    {
        'product_type': 'ensemble_mean',
        'variable': ['mean_temperature', 'precipitation_amount'],
        'grid_resolution': '0.1deg',
        'period': 'full_period',
        'version': '28.0e',
        'format': 'tgz',
    },
    'eobs.tar.gz'
)
```

---

### 5. CERRA (Copernicus European Regional ReAnalysis)

- **Ссылка:** https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-pressure-levels?tab=overview
- **Тип:** Региональный реанализ.
- **Что это:** Реанализ атмосферных погодных параметров по Европе с мезомасштабным разрешением.
- **Пространственное разрешение:** 5.5 км (мезомасштаб).
- **Применение:** Полезен для тестирования моделей понижения масштаба (downscaling).

#### Как скачать:

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-cerra-pressure-levels',
    {
        'variable': 'temperature',
        'pressure_level': '850',
        'year': '2020',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
    },
    'cerra_pressure.nc'
)
```

---

### 6. IFS HRES (ECMWF High-Resolution Forecast)

- **Ссылка:** https://www.ecmwf.int/en/forecasts/datasets/set-i
- **Тип:** Прогнозы численной модели.
- **Что это:** Результаты прогнозирования численной модели ECMWF IFS — SOTA (State Of The Art) среди численных моделей погоды.
- **Пространственное разрешение:** Сетка 0.1°.
- **Временной шаг:** 6-часовой.
- **Слои:** Наземные и атмосферные слои.
- **Доступ:** Через ECMWF, требуется лицензия/регистрация. Некоторые подмножества доступны через CDS.

#### Как скачать:

Доступ через ECMWF MARS API или CDS (подмножества):

```python
# Через CDS — набор "set-i" может быть частично доступен
import cdsapi
c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-complete',  # или специфический dataset для IFS HRES
    {
        'class': 'od',
        'type': 'fc',
        'stream': 'oper',
        'expver': '1',
        'date': '2020-01-01',
        'time': '00:00:00',
        'step': '6',
        'param': '130.128/129.128',
        'grid': '0.1/0.1',
        'format': 'netcdf',
    },
    'ifs_hres.nc'
)
```

---

### 7. GFS025 (Global Forecast System, NOAA)

- **Ссылка:** https://rda.ucar.edu/datasets/d084001/
- **Тип:** Прогнозы численной модели.
- **Что это:** Датасет на основе прогнозов численной модели GFS (Global Forecast System) от NOAA.
- **Пространственное разрешение:** 0.25°.
- **Доступ:** Требуется регистрация на NCAR RDA (бесплатно для исследований).

#### Как скачать:

1. Зарегистрироваться на https://rda.ucar.edu/
2. Скачать через веб-интерфейс или скрипт:

```bash
# Через wget с авторизацией (после получения cookies)
wget --load-cookies ~/.rda_cookies -r -np -nH \
  https://data.rda.ucar.edu/d084001/2020/20200101/gfs_4_20200101_0000_006.grb2
```

```python
# Альтернативно — через Python REST API NCAR
# См. https://github.com/ncar/rda-apps-clients
```

---

### 8. CMIP5 / CMIP6 (Coupled Model Intercomparison Project)

- **Ссылка:** https://pcmdi.llnl.gov/CMIP6/
- **Тип:** Климатические проекции.
- **Что это:** Региональные датасеты климатических проекций от множества климатических моделей по всему миру.
- **CMIP5:** Предыдущее поколение (2010-е), около 60 моделей.
- **CMIP6:** Текущее поколение, более 100 моделей, более высокое разрешение.
- **Сценарии:** historical, SSP1-2.6, SSP2-4.5, SSP5-8.5.
- **Временной охват:** 1850–2100 (зависит от сценария).
- **Применение:** Изучение долгосрочных климатических трендов, а не краткосрочных прогнозов.

#### Как скачать:

```bash
# 1. Установить synda (рекомендуется для массовой загрузки CMIP)
pip install synda

# 2. Или через wget с ESGF (Earth System Grid Federation)
# Поиск и скачивание через https://esgf-node.llnl.gov/search/cmip6/
```

```python
# Пример через wget (после получения URL с ESGF):
# wget -c <url-с-ESGF-ноды>
```

---

### 9. CRA5 (Compressed Representation of ERA5)

- **Ссылка:** https://github.com/taohan10200/CRA5
- **Тип:** Сжатое представление реанализа (через VAEformer).
- **Что это:** Экстремальное сжатие ERA5 с помощью модели VAEformer. Один из ориентиров хакатона.
- **Степень сжатия:** Экстремальная (существенно выше целевого диапазона 32–64×).
- **Применение:** Референсная точка для сравнения качества сжатия.

---

### 10. Aurora (Microsoft Weather Foundation Model)

- **Ссылка:** https://github.com/microsoft/aurora
- **Тип:** Foundation-модель прогноза погоды.
- **Что это:** Модель от Microsoft Research: 3D Perceiver Encoder → 3D Swin Transformer UNet → 3D Perceiver Decoder.
- **Архитектура:** Работает с произвольными погодными параметрами, масштабами и числом атмосферных слоёв.
- **Применение:** Источник архитектурного вдохновения для хакатона.

---

## 📥 Рекомендуемый способ получения данных для хакатона

### Основной (быстрый) способ — WeatherBench2 Zarr

**Датасет:** `gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`

Этот датасет **уже подготовлен** специально для ML/DL задач:
- Регулярная сетка 0.25° (721×1440, включая полюса)
- Шестичасовые кадры (00/06/12/18 UTC)
- Уже содержит derived-переменную `total_precipitation_6hr` (корректно извлечённую из накопленных осадков)
- Все необходимые pressure levels
- Доступ **анонимный** (без регистрации)

```python
import xarray as xr
import gcsfs
import zarr

# 1. Подключение к Google Cloud Storage (анонимно)
fs = gcsfs.GCSFileSystem(token='anon')

# 2. Открытие Zarr-датасета
ds = xr.open_zarr(
    fs.get_mapper(
        'weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
    ),
    consolidated=True
)

print(ds)
# Вывод: размерности, переменные, координаты

# 3. Выбор нужных переменных и временного диапазона
# 28-канальный набор для хакатона
surface_vars = ['2m_temperature', 'mean_sea_level_pressure',
                '10m_u_component_of_wind', '10m_v_component_of_wind',
                'total_precipitation_6hr', 'sea_surface_temperature',
                'total_column_water_vapour', 'total_cloud_cover']

pressure_vars = ['temperature', 'u_component_of_wind', 'v_component_of_wind',
                 'geopotential', 'specific_humidity']

# Выбор периода 2014–2021
ds_subset = ds.sel(time=slice('2014', '2021'))

# 4. Скачивание в локальный Zarr (опционально)
ds_subset.to_zarr('era5_28ch_0p25_6h.zarr')
```

```bash
# Или через gsutil (если установлен Google Cloud SDK):
gsutil -m cp -r \
  gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr \
  ./era5_wb2.zarr
```

### Альтернативный способ — Copernicus CDS

Если нужен полный контроль над переменными и сеткой:

```bash
pip install cdsapi
```

Скрипты для скачивания см. выше в разделах ERA5 pressure levels и ERA5 single levels.

> **Важно:** `tp6h` требует корректного извлечения из накопленных осадков — в WeatherBench2 это уже сделано.

---

## 🔑 Регистрация и лицензии

| Датасет | Требуется регистрация | Стоимость |
|---------|----------------------|-----------|
| WeatherBench2 (GCS) | Нет (анонимный доступ) | Бесплатно |
| ERA5 (CDS) | Да (https://cds.climate.copernicus.eu/) | Бесплатно |
| ERA5-Land (CDS) | Да (CDS) | Бесплатно |
| E-OBS (CDS) | Да (CDS) | Бесплатно |
| CERRA (CDS) | Да (CDS) | Бесплатно |
| IFS HRES | Да (ECMWF) | Частично бесплатно / лицензия |
| GFS025 | Да (NCAR RDA) | Бесплатно для исследований |
| CMIP6 | Да (ESGF) | Бесплатно |

---

## 📊 Сводная таблица датасетов

| Датасет | Тип | Разрешение | Шаг | Охват | Область |
|---------|-----|-----------|-----|------|--------|
| **ERA5** | Реанализ | 0.25° | 1 час | 1950–н.в. | Глобально |
| **ERA5-Land** | Реанализ | 0.1° | 1 час | 1950–н.в. | Только суша (глобально) |
| **E-OBS** | Наблюдения | 0.1° | 24 часа | 1950–н.в. | Европа |
| **CERRA** | Региональный реанализ | 5.5 км | 1 час | ~1985–н.в. | Европа |
| **IFS HRES** | Прогноз | 0.1° | 6 часов | Оперативный | Глобально |
| **GFS025** | Прогноз | 0.25° | 6 часов | Оперативный | Глобально |
| **CMIP6** | Климатические проекции | Разное | Месяц/день | 1850–2100 | Глобально |
| **WeatherBench2 ERA5** | Реанализ (Zarr) | 0.25° | 6 часов | 1959–2023 | Глобально |

---

## 🛠 Инструменты для работы с данными

| Инструмент | Назначение | Установка |
|------------|------------|-----------|
| **Xarray** | Многомерные массивы с координатами, NetCDF/Zarr/GRIB | `pip install xarray` |
| **Zarr** | Хранение многомерных массивов в облаке | `pip install zarr` |
| **gcsfs** | Доступ к Google Cloud Storage | `pip install gcsfs` |
| **dask** | Ленивые вычисления, крупные массивы | `pip install dask[complete]` |
| **cdsapi** | Доступ к Copernicus CDS | `pip install cdsapi` |
| **cfgrib** | Чтение GRIB-файлов через xarray | `pip install cfgrib` |
| **MetPy** | Расчёт метеорологических параметров | `pip install metpy` |
| **Cartopy** | Визуализация карт и проекций | `pip install cartopy` |