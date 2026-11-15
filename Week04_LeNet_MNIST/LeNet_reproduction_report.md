# LeNet MNIST Reproduction Report

## Experimental Results

A simplified LeNet architecture was implemented using PyTorch and evaluated on the MNIST handwritten digit dataset.

The model was trained for 5 epochs using the Adam optimizer with a learning rate of 0.001 and a batch size of 128.

The final test accuracy reached **98.61%**, exceeding the target of approximately 98%.

The training loss decreased from 0.4040 to 0.0498, while the test loss decreased from 0.1169 to 0.0466. The results indicate that the model learned effectively over the five training epochs.

## Prediction Examples

Ten randomly selected MNIST test images were visualized. All ten examples were classified correctly.

The prediction results are saved in `results/prediction_examples.png`.

## Misclassification Analysis

Ten misclassified test samples were examined.

| Case | True Label | Predicted Label | Possible Explanation |
|---|---|---|---|
| 1 | 5 | 3 | Curved strokes resemble the digit 3. |
| 2 | 8 | 3 | Incomplete loops make the digit resemble 3. |
| 3 | 4 | 2 | Unusual stroke arrangement resembles 2. |
| 4 | 6 | 0 | The dominant closed loop resembles 0. |
| 5 | 2 | 7 | The upper stroke resembles a handwritten 7. |
| 6 | 5 | 3 | Missing or unclear upper horizontal stroke. |
| 7 | 6 | 0 | The lower loop dominates the digit shape. |
| 8 | 3 | 5 | Unusual upper and lower curves resemble 5. |
| 9 | 8 | 2 | Incomplete loops create a shape resembling 2. |
| 10 | 2 | 7 | The upper structure resembles 7, with unusual lower strokes. |

The examples suggest that unusual handwriting styles, ambiguous stroke connections, and similarities between digit shapes may contribute to classification errors.

These explanations are based on visual inspection rather than direct evidence of the model's internal decision process.

The error examples are saved in `results/error_cases.png`.

## Preliminary Conclusion

The simplified LeNet model successfully achieved the required MNIST classification performance. Both training and test losses decreased throughout training, with no obvious indication of overfitting during the five epochs.

The misclassification examples suggest that visually ambiguous handwriting remains a challenge for the model.