## Supervised Learning
_Wendy and Miles_

Imagine you have the following dataset, where `temperature` (in F, in case that's relevant) is your target.

| leaf_index | birds_per_hour | weather | wind_speed | uv_index | **temperature** |
| ---------- | -------------- | ------- | ---------- | -------- | --------------- |
| 0.4        | 6              | 1       | 12         | 2        | 65              |
| ...        | ...            | ...     | ...        | ...      | ...             |

* `leaf_index` is the percentage of leaves still on 'normal' trees that lose their leaves in the winter
* `birds_per_hour` is the number of birds that appear at a given birdfeeder

-----

1. Given that `temperature` is the target, what kind of model would we try to fit to this data?
2. What is the name and general form of the cost function for the model you identified in part 1?
3. Imagine you've fitted a model to the data. Now, you're trying to predict new values. What features must you pass to the model, and what do you expect the model to output?
4. Now, you're running your model on test data. Your model achieved a RMSE of 3 on the training data, but when you test your model on separate test data, the RMSE is 34. What happened?
