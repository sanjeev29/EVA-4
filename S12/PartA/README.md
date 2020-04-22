# Work on tiny-imagenet-200 Dataset with ResNet18 Architecture

## Goals

* Validation accuracy > 50%
* Epochs <= 50

## Results

* Reached a validation accuracy of 51.28% in 12th epoch
* Best training accuracy - 76.17%
* Best test accuracy - 56.45%
* Model run for 27 epochs

### Accuracy Change Plot

![Image description](https://github.com/sanjeev29/EVA-4/blob/master/S12/PartA/accuracy_change_plot.jpg)


## Issue

'Buffered data was truncated after reaching the output size limit.' issue occured during model training and testing in Google Colab. Model ran only for 27/49 epochs (starting with epoch=0 as 1st epoch)
