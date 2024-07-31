## Inspiration

Cancer is a disease that claims about 10-15 million lives per year. However, these numbers usually don't affect us until it hits close to home. Last year, I lost my maternal grandfather to cancer. If his cyst near the stomach had been detected earlier, he might still be with us today.

## What it does

My project takes an image of a CT scan and predicts the possibility and type of chest cancer.

## How we built it

I trained an EffNetModel on Kaggle's chest CT scan dataset. Initially, the accuracy was not good, so I incorporated data augmentation and a scheduler, which improved the model's performance.

## Challenges we ran into

- Finding the right dataset: A dataset that is too large would train slowly without access to GPUs, while a dataset that is too small might be biased.
- Hosting on Hugging Face.
- Learning to use Gradio.

## Accomplishments that we're proud of

The model achieved about 85% validation accuracy and 78% test accuracy, which can potentially help in the early detection and treatment of chest cancer.

## What we learned

- How to host on Hugging Face.
- Techniques to tackle overfitting and underfitting.

## What's next for Cancer Detector for a Cancer Free World

We aim to fine-tune the model on more datasets to help more people worldwide.
