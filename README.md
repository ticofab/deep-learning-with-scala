# From Zero To Deep Learning With Scala

Building blocks to train your own binary classifier using Scala.

[`ModelSettings.scala`](ModelSettings.scala) contains the actual construction of the neural network (using [Deeplearning4j](https://deeplearning4j.org)), the necessary hyperparameters and a few other settings.

[`ModelEvaluator.scala`](ModelEvaluator.scala) contains logic to load images, train and evaluate the model.

The dataset to run your own experiments is avaiable [here](https://drive.google.com/drive/folders/1-O17UhjdJjtKJ1iAMDBJhaYMiTmhIIAi?usp=sharing).

I gave a few talks about this project. Please have a look at the [slides](https://www.slideshare.net/FabioTiriticco/from-zero-to-deep-learning-with-scala-232229345) (and recording will come soon).

![A plane on a bridge](plane.png)

## License

    Copyright 2020 Fabio Tiriticco - Fabway

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
