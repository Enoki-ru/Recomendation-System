# Рекомендательные системы
Всем снова ку!)

Последнюю неделю я изучал нейронные сети, однако понял, что в эти дебри мне лучше не суваться, поэтому давайте пропустим большой курс SkillBox, посвященные этому, и перейдем сразу в систему рекомендаций. Это алгоритм, который использует данные пользователей, для того, чтобы выявлять похожие запросы среди пользователей.
## Как это работает
Допустим, вы оценили фильм определенного жанра в 8/10 баллов. И так несколько фильмов. После этого, среди всех пользователей вы начинаете занимать какое-то пространство, Даже можно сказать вектор ответов. Выглядит это так:

![image.png](rec.png) 

Таким образом можно найти соседей среди ваших ответов, и посмотреть, что нравится им. Если вы оцениваете разные фильмы примерно одинаково, то скорее всего то, что понравится этому человеку, понравится и вам!

## Реализация
В ходе статьи будет использоваться новая для меня библиотека *Surpriselib*. Эта библиотека позволяет быстро анализировать рекомендательные системы.

---
Теперь вроде всё, мы со всем разобрались, так что погнали!



```python
print("Первым делом скачаем библиотеку, написав $ pip install scikit-surprise```")
```

    Первым делом скачаем библиотеку, написав $ pip install scikit-surprise```
    


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv("recdemo.csv", sep=";") #небольшой dataset
```


```python
df # Мне пришлось вручную в эскеле вбивать значения и делать датасет, тк в уроке не нашел этого файла. Поставьте лайк за это если не сложно пахпапх
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_unpivot=pd.melt(df, id_vars=['id'])
df_unpivot.head(10) # преобразует данные в так называемый аккуратный вид
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>A</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>A</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>A</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>A</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>B</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>B</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_unpivot.dropna(inplace=True) # Убрали строки, где есть пустые значения
df_unpivot.columns=['userID', 'itemID', 'rating'] # Переименовали колонки в удобные нам названия
df_unpivot.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userID</th>
      <th>itemID</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>A</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>B</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>B</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>B</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6</td>
      <td>B</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>C</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2</td>
      <td>C</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from surprise import Dataset
from surprise import Reader

print("Reader отвечает за размер шкалы. В данном случае рейтинги у нас от 1 до 5, поэтому так и пишем")

reader = Reader(rating_scale=(1, 5)) # Зададим разброс оценок
data = Dataset.load_from_df(df_unpivot[['userID', 'itemID', 'rating']], reader) #создадим объект, с которым умеет работать библиотека
```

    Reader отвечает за размер шкалы. В данном случае рейтинги у нас от 1 до 5, поэтому так и пишем
    

Далее нужно нам просто разбить датасет, как ранее мы делали, на train и test, для просмотра точности выполнения операции.



```python
trainset= data.build_full_trainset()
testset=trainset.build_anti_testset()
```


```python
testset[0:10] # Номер пользователя, Номер фильма, и средний рейтинг, посчитанный за все значения по элементам массива
```




    [(1, 'F', 3.4642857142857144),
     (2, 'E', 3.4642857142857144),
     (5, 'B', 3.4642857142857144),
     (5, 'F', 3.4642857142857144),
     (7, 'B', 3.4642857142857144),
     (7, 'C', 3.4642857142857144),
     (7, 'E', 3.4642857142857144),
     (3, 'A', 3.4642857142857144),
     (3, 'E', 3.4642857142857144),
     (6, 'A', 3.4642857142857144)]



Давайте быстро вспомним, как работает knn метод (вдруг кто забыл). Создадим сэмпл, на основе которого наша модель обучится. Далее найдем 2х ближайших соседей для точки. Как видите, первые два значения, это расстояния до ближайших соседей, и вторые два значения, это номера этих соседей из списка


```python
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5], [0.3, .5, 0.2], [.2, 1., .5]]
from sklearn.neighbors import NearestNeighbors
knn=NearestNeighbors(n_neighbors=2)
knn.fit(samples)
print(knn.kneighbors([[1.,1.,1.]]))
```

    (array([[0.5       , 0.94339811]]), array([[2, 4]], dtype=int64))
    

Теперь давайте посмотрим на реализацию уже в нашей новой библиотеке для этого метода.


```python
from surprise import KNNBaseline
algo=KNNBaseline(k=1) # по 1 ближайшему соседу.
algo.fit(trainset)
predictions=algo.test(testset) # сделаем прогноз для тестсета
```

    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    


```python
predictions[0:5]
```




    [Prediction(uid=1, iid='F', r_ui=3.4642857142857144, est=1.9577925900897966, details={'actual_k': 1, 'was_impossible': False}),
     Prediction(uid=2, iid='E', r_ui=3.4642857142857144, est=1.0422074099102034, details={'actual_k': 1, 'was_impossible': False}),
     Prediction(uid=5, iid='B', r_ui=3.4642857142857144, est=4.082661845573158, details={'actual_k': 1, 'was_impossible': False}),
     Prediction(uid=5, iid='F', r_ui=3.4642857142857144, est=2.0826618455731576, details={'actual_k': 1, 'was_impossible': False}),
     Prediction(uid=7, iid='B', r_ui=3.4642857142857144, est=3.8701641215302014, details={'actual_k': 1, 'was_impossible': False})]



* uid - id пользователя
* iid - id фильма по которому искался сосед
* est - та оценка, которая нас интересует. 

Давайте перепишем результаты в более удобный формат. Для этого создадим таблицу


```python
import warnings

warnings.filterwarnings('ignore')
# для того, чтобы убрать надоедливое напоминание, что опр функция уйдет в будущей версии pandas
df_unpivot1=df_unpivot.copy()
for i in predictions:
    df_unpivot1 = df_unpivot1.append({'userID':i.uid, 'itemID': i.iid, 'rating': i.est}, ignore_index=True)
```


```python
df_unpivot1.pivot(index='userID', columns='itemID', values='rating')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>itemID</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
    <tr>
      <th>userID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.957793</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.042207</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.311189</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>4.090446</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.072513</td>
      <td>4.197382</td>
      <td>2.072513</td>
      <td>4.072513</td>
      <td>4.000000</td>
      <td>3.976639</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.000000</td>
      <td>4.082662</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.082662</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.469084</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>4.248341</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>3.870164</td>
      <td>1.870164</td>
      <td>2.000000</td>
      <td>3.691628</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3.095874</td>
      <td>1.909554</td>
      <td>2.095874</td>
      <td>4.095874</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



Как видите, теперь в таблице убраны все NaN-ы. Да, это сделали мы)


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Теперь давайте сделаем всё то же самое, но для k=3")
```

    Теперь давайте сделаем всё то же самое, но для k=3
    


```python
algo = KNNBaseline(k=3)
algo.fit(trainset)
# Than predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)
```

    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    


```python
df_unpivot3 = df_unpivot.copy()
for i in predictions:
    df_unpivot3 = df_unpivot3.append({'userID':i.uid, 'itemID': i.iid, 'rating': i.est}, ignore_index=True)
df_unpivot3.pivot(index='userID', columns='itemID', values='rating')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>itemID</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
    <tr>
      <th>userID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.507435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.601745</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.256100</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.527693</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.083864</td>
      <td>4.197382</td>
      <td>2.083864</td>
      <td>4.083864</td>
      <td>4.000000</td>
      <td>3.976639</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.000000</td>
      <td>3.834977</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.013312</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.418235</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.590444</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>3.366011</td>
      <td>1.857717</td>
      <td>2.000000</td>
      <td>3.095321</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.579863</td>
      <td>3.471928</td>
      <td>3.463240</td>
      <td>4.463240</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



Как видите, значения изменились. Так как же нам понять, когда нужно остановиться? Для этого нужно воспользоваться знаниями кросс-платформенной валидации:

---
## Кросс-валидация



```python
from surprise.model_selection import cross_validate
```


```python
cross_validate(algo,data,cv=2,verbose=True)
```

    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Estimating biases using als...
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Evaluating RMSE, MAE of algorithm KNNBaseline on 2 split(s).
    
                      Fold 1  Fold 2  Mean    Std     
    RMSE (testset)    1.5231  1.3452  1.4342  0.0889  
    MAE (testset)     1.1617  1.0102  1.0859  0.0758  
    Fit time          0.00    0.00    0.00    0.00    
    Test time         0.00    0.00    0.00    0.00    
    




    {'test_rmse': array([1.52310128, 1.34522382]),
     'test_mae': array([1.16171853, 1.01016551]),
     'fit_time': (0.0010008811950683594, 0.0),
     'test_time': (0.0, 0.0)}



Как видите, мы использовали всего 2 фолда (хз что это, но очень интересно). Разбиение в этом методе не нужно на тест и трейн. Оно само само, и слава богу



```python
for i in range(1,6):
    algo=KNNBaseline(k=i,verbose=False) # Отключили вывод
    cv=cross_validate(algo,data,measures=['RMSE'],cv=3,verbose=False)
    print(str(i)+'NN:',np.mean(cv['test_rmse']))
```

    1NN: 1.751042465246463
    2NN: 1.7374517019459086
    3NN: 1.5664632445650042
    4NN: 1.3412878123745717
    5NN: 1.4208622735581171
    

Как видите, наибольшее качество у нас при 2 или при 4 соседях

---
## Метод косинусной меры 


```python

```


```python
algo = KNNBaseline(k=5,sim_options= {'name': 'cosine'}, verbose=False)
predictions = algo.fit(trainset).test(testset)
df_unpivot5_cos = df_unpivot.copy()
for i in predictions:
    df_unpivot5_cos = df_unpivot5_cos.append({'userID':i.uid, 'itemID': i.iid, 'rating': i.est}, ignore_index=True)
df_unpivot5_cos.pivot(index='userID', columns='itemID', values='rating')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>itemID</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
    <tr>
      <th>userID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>3.572362</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.927003</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.754335</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.270544</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.134947</td>
      <td>4.197382</td>
      <td>2.134947</td>
      <td>4.134947</td>
      <td>4.000000</td>
      <td>3.976639</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.000000</td>
      <td>3.668772</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.714426</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.939492</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.360864</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>3.442211</td>
      <td>2.951617</td>
      <td>2.000000</td>
      <td>2.802029</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.700881</td>
      <td>3.765123</td>
      <td>3.231273</td>
      <td>4.431273</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



Как видите, сегодня бзе красивых графиков(((
Возможно, позже я буду ориентироваться исключительно на графиках, потому что они довольно интересны и понятны для изучения
