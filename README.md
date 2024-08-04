# Classification_with_an_Academic_Success_Dataset_Kaggle_Playground_Prediction_Competition

# Overview
Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal of this competition is to predict academic risk of students in higher education.

# Evaluation
Submissions are evaluated using the accuracy score.

# Importing Libraries and Dataset


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df_train = pd.read_csv(r"C:\Users\ACER\Downloads\playground-series-s4e6\train.csv", index_col='id')
```


```python
# Encode the target feature
le = LabelEncoder()
df_train['Target'] = le.fit_transform(df_train['Target'])
```


```python
df_train.head()
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
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Previous qualification (grade)</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9238</td>
      <td>1</td>
      <td>1</td>
      <td>126.0</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>12.428571</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>9238</td>
      <td>1</td>
      <td>1</td>
      <td>125.0</td>
      <td>1</td>
      <td>19</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>17</td>
      <td>2</td>
      <td>9254</td>
      <td>1</td>
      <td>1</td>
      <td>137.0</td>
      <td>1</td>
      <td>3</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>16.2</td>
      <td>0.3</td>
      <td>-0.92</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>9500</td>
      <td>1</td>
      <td>1</td>
      <td>131.0</td>
      <td>1</td>
      <td>19</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>11</td>
      <td>7</td>
      <td>12.820000</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>9500</td>
      <td>1</td>
      <td>1</td>
      <td>132.0</td>
      <td>1</td>
      <td>19</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>12</td>
      <td>6</td>
      <td>12.933333</td>
      <td>0</td>
      <td>7.6</td>
      <td>2.6</td>
      <td>0.32</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



# Splitting and Preprocessing


```python
X = df_train.drop(columns = 'Target')
y = df_train['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y, test_size = 0.2)
```


```python
# Pipeline
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

num_feature = ['Previous qualification (grade)',
    'Admission grade',
    'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP','Application order']

cat_feature = ['Marital status',
    'Application mode',
    'Course','Daytime/evening attendance',
    'Previous qualification','Nacionality',"Mother's qualification","Father's qualification","Mother's occupation",
    "Father's occupation",'Displaced',
    'Educational special needs','Debtor','Tuition fees up to date','Gender','Scholarship holder','International']

preprocessor = ColumnTransformer([
    ('numeric', numerical_pipeline, num_feature),
    ('categoric', categorical_pipeline, cat_feature)
])
```

# Modelling


```python
pipeline_cb = Pipeline([
    ('prep', preprocessor),
    ('algo', CatBoostClassifier(verbose = 0))
])
```


```python
pipeline_cb.fit(X_train, y_train)

y_train_pred = pipeline_cb.predict(X_train)
y_test_pred = pipeline_cb.predict(X_test)

y_train_pred = le.inverse_transform(y_train_pred)
y_test_pred = le.inverse_transform(y_test_pred)
y_train_original = le.inverse_transform(y_train)
y_test_original = le.inverse_transform(y_test)

train_f1 = f1_score(y_train_original, y_train_pred, average = 'weighted')
test_f1 = f1_score(y_test_original, y_test_pred, average = 'weighted')

print(f"Training F1 Score: {train_f1}")
print(f"Test F1 Score: {test_f1}")
```

    D:\anaconda3\Lib\site-packages\sklearn\preprocessing\_label.py:153: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    D:\anaconda3\Lib\site-packages\sklearn\preprocessing\_label.py:153: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    Training F1 Score: 0.8528335491831845
    Test F1 Score: 0.8310450566606753
    

# Hyperparameter Tuning


```python
param_grid = {
    'algo__iterations': [100, 200],
    'algo__learning_rate': [0.01, 0.1],
    'algo__depth': [4, 6, 8],
    'algo__l2_leaf_reg': [1, 3, 5],
    'algo__border_count': [32, 64, 128]
}
```


```python
grid_search = GridSearchCV(
    pipeline_cb, param_grid, scoring = make_scorer(f1_score, average = 'weighted'),
    cv = 3, verbose = 2, n_jobs = -1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
```

    Fitting 3 folds for each of 108 candidates, totalling 324 fits
    Best Parameters: {'algo__border_count': 128, 'algo__depth': 8, 'algo__iterations': 200, 'algo__l2_leaf_reg': 3, 'algo__learning_rate': 0.1}
    


```python
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;prep&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Previous qualification &#x27;
                                                   &#x27;(grade)&#x27;,
                                                   &#x27;Admission grade&#x27;,
                                                   &#x27;Age at enrollment&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(credited)&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(enrolled)&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(evaluations)&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(approved)&#x27;,
                                                   &#x27;Curricular units 1st se...
                                                   &#x27;Application mode&#x27;, &#x27;Course&#x27;,
                                                   &#x27;Daytime/evening attendance&#x27;,
                                                   &#x27;Previous qualification&#x27;,
                                                   &#x27;Nacionality&#x27;,
                                                   &quot;Mother&#x27;s qualification&quot;,
                                                   &quot;Father&#x27;s qualification&quot;,
                                                   &quot;Mother&#x27;s occupation&quot;,
                                                   &quot;Father&#x27;s occupation&quot;,
                                                   &#x27;Displaced&#x27;,
                                                   &#x27;Educational special needs&#x27;,
                                                   &#x27;Debtor&#x27;,
                                                   &#x27;Tuition fees up to date&#x27;,
                                                   &#x27;Gender&#x27;,
                                                   &#x27;Scholarship holder&#x27;,
                                                   &#x27;International&#x27;])])),
                (&#x27;algo&#x27;,
                 &lt;catboost.core.CatBoostClassifier object at 0x000001A6D529E610&gt;)])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;prep&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                                  Pipeline(steps=[(&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;Previous qualification &#x27;
                                                   &#x27;(grade)&#x27;,
                                                   &#x27;Admission grade&#x27;,
                                                   &#x27;Age at enrollment&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(credited)&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(enrolled)&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(evaluations)&#x27;,
                                                   &#x27;Curricular units 1st sem &#x27;
                                                   &#x27;(approved)&#x27;,
                                                   &#x27;Curricular units 1st se...
                                                   &#x27;Application mode&#x27;, &#x27;Course&#x27;,
                                                   &#x27;Daytime/evening attendance&#x27;,
                                                   &#x27;Previous qualification&#x27;,
                                                   &#x27;Nacionality&#x27;,
                                                   &quot;Mother&#x27;s qualification&quot;,
                                                   &quot;Father&#x27;s qualification&quot;,
                                                   &quot;Mother&#x27;s occupation&quot;,
                                                   &quot;Father&#x27;s occupation&quot;,
                                                   &#x27;Displaced&#x27;,
                                                   &#x27;Educational special needs&#x27;,
                                                   &#x27;Debtor&#x27;,
                                                   &#x27;Tuition fees up to date&#x27;,
                                                   &#x27;Gender&#x27;,
                                                   &#x27;Scholarship holder&#x27;,
                                                   &#x27;International&#x27;])])),
                (&#x27;algo&#x27;,
                 &lt;catboost.core.CatBoostClassifier object at 0x000001A6D529E610&gt;)])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;prep: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for prep: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;numeric&#x27;,
                                 Pipeline(steps=[(&#x27;scaler&#x27;, StandardScaler())]),
                                 [&#x27;Previous qualification (grade)&#x27;,
                                  &#x27;Admission grade&#x27;, &#x27;Age at enrollment&#x27;,
                                  &#x27;Curricular units 1st sem (credited)&#x27;,
                                  &#x27;Curricular units 1st sem (enrolled)&#x27;,
                                  &#x27;Curricular units 1st sem (evaluations)&#x27;,
                                  &#x27;Curricular units 1st sem (approved)&#x27;,
                                  &#x27;Curricular units 1st sem (grade)&#x27;,
                                  &#x27;Curricular units 1st sem (w...
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 [&#x27;Marital status&#x27;, &#x27;Application mode&#x27;,
                                  &#x27;Course&#x27;, &#x27;Daytime/evening attendance&#x27;,
                                  &#x27;Previous qualification&#x27;, &#x27;Nacionality&#x27;,
                                  &quot;Mother&#x27;s qualification&quot;,
                                  &quot;Father&#x27;s qualification&quot;,
                                  &quot;Mother&#x27;s occupation&quot;, &quot;Father&#x27;s occupation&quot;,
                                  &#x27;Displaced&#x27;, &#x27;Educational special needs&#x27;,
                                  &#x27;Debtor&#x27;, &#x27;Tuition fees up to date&#x27;, &#x27;Gender&#x27;,
                                  &#x27;Scholarship holder&#x27;, &#x27;International&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numeric</label><div class="sk-toggleable__content fitted"><pre>[&#x27;Previous qualification (grade)&#x27;, &#x27;Admission grade&#x27;, &#x27;Age at enrollment&#x27;, &#x27;Curricular units 1st sem (credited)&#x27;, &#x27;Curricular units 1st sem (enrolled)&#x27;, &#x27;Curricular units 1st sem (evaluations)&#x27;, &#x27;Curricular units 1st sem (approved)&#x27;, &#x27;Curricular units 1st sem (grade)&#x27;, &#x27;Curricular units 1st sem (without evaluations)&#x27;, &#x27;Curricular units 2nd sem (credited)&#x27;, &#x27;Curricular units 2nd sem (enrolled)&#x27;, &#x27;Curricular units 2nd sem (evaluations)&#x27;, &#x27;Curricular units 2nd sem (approved)&#x27;, &#x27;Curricular units 2nd sem (grade)&#x27;, &#x27;Curricular units 2nd sem (without evaluations)&#x27;, &#x27;Unemployment rate&#x27;, &#x27;Inflation rate&#x27;, &#x27;GDP&#x27;, &#x27;Application order&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;StandardScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">categoric</label><div class="sk-toggleable__content fitted"><pre>[&#x27;Marital status&#x27;, &#x27;Application mode&#x27;, &#x27;Course&#x27;, &#x27;Daytime/evening attendance&#x27;, &#x27;Previous qualification&#x27;, &#x27;Nacionality&#x27;, &quot;Mother&#x27;s qualification&quot;, &quot;Father&#x27;s qualification&quot;, &quot;Mother&#x27;s occupation&quot;, &quot;Father&#x27;s occupation&quot;, &#x27;Displaced&#x27;, &#x27;Educational special needs&#x27;, &#x27;Debtor&#x27;, &#x27;Tuition fees up to date&#x27;, &#x27;Gender&#x27;, &#x27;Scholarship holder&#x27;, &#x27;International&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">CatBoostClassifier</label><div class="sk-toggleable__content fitted"><pre>&lt;catboost.core.CatBoostClassifier object at 0x000001A6D529E610&gt;</pre></div> </div></div></div></div></div></div>




```python
y_train_pred_best = best_model.predict(X_train)
y_test_pred_best = best_model.predict(X_test)

y_train_pred_best = le.inverse_transform(y_train_pred_best)
y_test_pred_best = le.inverse_transform(y_test_pred_best)
y_train_original_best = le.inverse_transform(y_train)
y_test_original_best = le.inverse_transform(y_test)

train_f1_best = f1_score(y_train_original_best, y_train_pred_best, average = 'weighted')
test_f1_best = f1_score(y_test_original_best, y_test_pred_best, average = 'weighted')

print(f"Tuned Training F1 Score: {train_f1_best}")
print(f"Tuned Test F1 Score: {test_f1_best}")
```

    D:\anaconda3\Lib\site-packages\sklearn\preprocessing\_label.py:153: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    D:\anaconda3\Lib\site-packages\sklearn\preprocessing\_label.py:153: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    Tuned Training F1 Score: 0.8406813717854638
    Tuned Test F1 Score: 0.82881330555337
    

# Evaluation


```python
# Confusion matrix for the best model
cm = confusion_matrix(y_test_original_best, y_test_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```


    
![png](output_17_0.png)
    



```python
# Display feature importance for the best model
catboost_model_best = best_model.named_steps['algo']
feature_importances = catboost_model_best.get_feature_importance()
feature_names = (best_model.named_steps['prep'].named_transformers_['categoric'].named_steps['onehot'].get_feature_names_out().tolist() + 
                 best_model.named_steps['prep'].named_transformers_['numeric'].feature_names_in_.tolist())

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Filter to get top 10 features
top_10_features = feature_importance_df.head(10)

# Plot feature importance for top 10 features
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Importance', y='Feature', data=top_10_features)
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Add labels to the bars
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')

plt.show()
```


    
![png](output_18_0.png)
    


# Apply to New Dataset


```python
df_test = pd.read_csv(r"C:\Users\ACER\Downloads\playground-series-s4e6\test.csv", index_col = 'id')
df_test
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
      <th>Marital status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Previous qualification (grade)</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>...</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>76518</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9500</td>
      <td>1</td>
      <td>1</td>
      <td>141.0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>76519</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9238</td>
      <td>1</td>
      <td>1</td>
      <td>128.0</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.500000</td>
      <td>0</td>
      <td>11.1</td>
      <td>0.6</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>76520</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9238</td>
      <td>1</td>
      <td>1</td>
      <td>118.0</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>11</td>
      <td>5</td>
      <td>11.000000</td>
      <td>0</td>
      <td>15.5</td>
      <td>2.8</td>
      <td>-4.06</td>
    </tr>
    <tr>
      <th>76521</th>
      <td>1</td>
      <td>44</td>
      <td>1</td>
      <td>9147</td>
      <td>1</td>
      <td>39</td>
      <td>130.0</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>14</td>
      <td>5</td>
      <td>11.000000</td>
      <td>0</td>
      <td>8.9</td>
      <td>1.4</td>
      <td>3.51</td>
    </tr>
    <tr>
      <th>76522</th>
      <td>1</td>
      <td>39</td>
      <td>1</td>
      <td>9670</td>
      <td>1</td>
      <td>1</td>
      <td>110.0</td>
      <td>1</td>
      <td>1</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>4</td>
      <td>10.666667</td>
      <td>2</td>
      <td>7.6</td>
      <td>2.6</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>127525</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>171</td>
      <td>1</td>
      <td>1</td>
      <td>128.0</td>
      <td>1</td>
      <td>38</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>15.5</td>
      <td>2.8</td>
      <td>-4.06</td>
    </tr>
    <tr>
      <th>127526</th>
      <td>2</td>
      <td>39</td>
      <td>1</td>
      <td>9119</td>
      <td>1</td>
      <td>19</td>
      <td>133.1</td>
      <td>1</td>
      <td>19</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
    </tr>
    <tr>
      <th>127527</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>171</td>
      <td>1</td>
      <td>1</td>
      <td>127.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>15.5</td>
      <td>2.8</td>
      <td>-4.06</td>
    </tr>
    <tr>
      <th>127528</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>9773</td>
      <td>1</td>
      <td>1</td>
      <td>132.0</td>
      <td>1</td>
      <td>19</td>
      <td>19</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>13.000000</td>
      <td>0</td>
      <td>7.6</td>
      <td>2.6</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>127529</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>171</td>
      <td>1</td>
      <td>1</td>
      <td>129.0</td>
      <td>1</td>
      <td>37</td>
      <td>38</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>7.6</td>
      <td>2.6</td>
      <td>0.32</td>
    </tr>
  </tbody>
</table>
<p>51012 rows × 36 columns</p>
</div>




```python
X_test_new = df_test.copy()
```


```python
X_test_new_transformed = best_model.named_steps['prep'].transform(X_test_new)
```


```python
new_predictions = best_model.named_steps['algo'].predict(X_test_new_transformed)
```


```python
new_predictions = le.inverse_transform(new_predictions)
```

    D:\anaconda3\Lib\site-packages\sklearn\preprocessing\_label.py:153: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    


```python
df_test['Target'] = new_predictions
results = df_test[['Target']].reset_index()
print(results)
```

               id    Target
    0       76518   Dropout
    1       76519  Graduate
    2       76520  Graduate
    3       76521  Graduate
    4       76522  Enrolled
    ...       ...       ...
    51007  127525   Dropout
    51008  127526   Dropout
    51009  127527   Dropout
    51010  127528   Dropout
    51011  127529   Dropout
    
    [51012 rows x 2 columns]
    


```python
results.to_csv(r"C:\Users\ACER\Downloads\Classification with an Academic Success Dataset (CatBoost Model).csv", index=False)
```


```python

```
