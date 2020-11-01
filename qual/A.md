
# A. Первый бинарный классификатор
Ограничение времени 	300 секунд
Ограничение памяти 	4Gb
Ввод 	input.txt
Вывод 	params.json
В своих мемуарах в главе «Мой первый бинарный классификатор» один известный ML-инженер упоминает, что свой первый бинарный классификатор он обучал с помощью https://catboost.ai/ со следующими параметрами:
```python
params = { 
    "iterations": 100, 
    "learning_rate": 0.0001, 
    "depth": 1, 
    "l2_leaf_reg": 100.0, 
    "rsm": 0.9, 
    "border_count": 10, 
    "max_ctr_complexity": 3, 
    "random_strength": 40.0, 
    "bagging_temperature": 100.0, 
    "grow_policy": "Depthwise", 
    "min_data_in_leaf": 5, 
    "langevin": true, 
    "diffusion_temperature": 100000 
}
```

Приложите набор новых значений этих параметров, который на отложенной выборке из того же датасета дает на 30% лучшее качество (AUC).
Формат ввода
Архив с датасетом в формате json:
```json
[ 
  {"label": "<label_value_1>", "1": "<feature_value_11>", "2": "<feature_value_12>", ...}, 
  {"label": "<label_value_2>", "1": "<feature_value_21>", "2": "<feature_value_22>", ...}, 
  ... 
]
```

https://yadi.sk/d/DMmse1jOiRxmdQ


# Формат вывода

Указанные выше параметры их новые значения в формате:
```json
{ 
    "iterations": <param_value_1>, 
    "learning_rate": <param_value_2>, 
    ... 
}
```