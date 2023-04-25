---
library_name: sklearn
tags:
- sklearn
- skops
- text-classification
---

# Model description

This is a logistic regression model trained with GPT-2 embeddings on imdb dataset.
The notebook to generate this model is in this repository and in this [kaggle link](https://www.kaggle.com/code/unofficialmerve/scikit-learn-with-transformers-with-skops/notebook).

## Intended uses & limitations

This model is trained for educational purposes.

## Training Procedure

### Hyperparameters

The model is trained with below hyperparameters.

<details>
<summary> Click to expand </summary>

| Hyperparameter                | Value                                                                                                             |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------|
| memory                        |                                                                                                                   |
| steps                         | [('embedding', HFTransformersLanguage(model_name_or_path='facebook/bart-base')), ('model', LogisticRegression())] |
| verbose                       | False                                                                                                             |
| embedding                     | HFTransformersLanguage(model_name_or_path='facebook/bart-base')                                                   |
| model                         | LogisticRegression()                                                                                              |
| embedding__model_name_or_path | facebook/bart-base                                                                                                |
| model__C                      | 1.0                                                                                                               |
| model__class_weight           |                                                                                                                   |
| model__dual                   | False                                                                                                             |
| model__fit_intercept          | True                                                                                                              |
| model__intercept_scaling      | 1                                                                                                                 |
| model__l1_ratio               |                                                                                                                   |
| model__max_iter               | 100                                                                                                               |
| model__multi_class            | auto                                                                                                              |
| model__n_jobs                 |                                                                                                                   |
| model__penalty                | l2                                                                                                                |
| model__random_state           |                                                                                                                   |
| model__solver                 | lbfgs                                                                                                             |
| model__tol                    | 0.0001                                                                                                            |
| model__verbose                | 0                                                                                                                 |
| model__warm_start             | False                                                                                                             |

</details>

### Model Plot

The model plot is below.

<style>#sk-c17251a9-68a0-4b34-a80a-a89592893866 {color: black;background-color: white;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 pre{padding: 0;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-toggleable {background-color: white;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-estimator:hover {background-color: #d4ebff;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-item {z-index: 1;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-parallel-item:only-child::after {width: 0;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-c17251a9-68a0-4b34-a80a-a89592893866 div.sk-text-repr-fallback {display: none;}</style><div id="sk-c17251a9-68a0-4b34-a80a-a89592893866" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;embedding&#x27;,HFTransformersLanguage(model_name_or_path=&#x27;facebook/bart-base&#x27;)),(&#x27;model&#x27;, LogisticRegression())])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cae9454b-9d5c-424d-bbf8-8256c92c6733" type="checkbox" ><label for="cae9454b-9d5c-424d-bbf8-8256c92c6733" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;embedding&#x27;,HFTransformersLanguage(model_name_or_path=&#x27;facebook/bart-base&#x27;)),(&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0e6ffcb1-dfbc-44ef-9a7c-c15d496369c7" type="checkbox" ><label for="0e6ffcb1-dfbc-44ef-9a7c-c15d496369c7" class="sk-toggleable__label sk-toggleable__label-arrow">HFTransformersLanguage</label><div class="sk-toggleable__content"><pre>HFTransformersLanguage(model_name_or_path=&#x27;facebook/bart-base&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b8eb1956-cf67-40f5-96f2-c2f0a0a41704" type="checkbox" ><label for="b8eb1956-cf67-40f5-96f2-c2f0a0a41704" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>

## Evaluation Results

You can find the details about evaluation process and the evaluation results.



| Metric   |    Value |
|----------|----------|
| f1_score | 0.867535 |

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

```python
[More Information Needed]
```

</details>


```


# Additional Content

## Confusion matrix

![Confusion matrix](confusion_matrix.png)